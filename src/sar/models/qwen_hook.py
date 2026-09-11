from __future__ import annotations

import types
from dataclasses import dataclass

import torch
from torch import nn

from sar.methods.qacr import QACRRouter


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def qacr_eager_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    *,
    audio_mask: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    **kwargs,
):
    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    if audio_mask is not None and gate is not None:
        if audio_mask.shape != gate.shape:
            raise ValueError("audio_mask and gate shapes must match")
        if audio_mask.shape[1] != attn_weights.shape[-1]:
            raise ValueError("controller key mask length must match attention key length")
        contribution_gate = torch.where(
            audio_mask,
            gate.to(dtype=attn_weights.dtype),
            torch.ones_like(gate, dtype=attn_weights.dtype),
        )
        attn_for_values = attn_weights * contribution_gate[:, None, None, :]
    else:
        attn_for_values = attn_weights

    attn_output = torch.matmul(attn_for_values, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@dataclass
class QACRController:
    router: QACRRouter
    audio_mask: torch.Tensor | None = None
    query_mask: torch.Tensor | None = None
    gate: torch.Tensor | None = None

    def set_masks(self, audio_mask: torch.Tensor, query_mask: torch.Tensor) -> None:
        if audio_mask.dtype is not torch.bool or query_mask.dtype is not torch.bool:
            raise ValueError("audio_mask and query_mask must be boolean")
        if audio_mask.shape != query_mask.shape:
            raise ValueError("audio_mask and query_mask must have identical [batch, seq] shape")
        self.audio_mask = audio_mask
        self.query_mask = query_mask
        self.gate = None

    def compute_prefill_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.audio_mask is None or self.query_mask is None:
            raise RuntimeError("set_masks() must be called before prefill")
        if hidden_states.shape[:2] != self.audio_mask.shape:
            raise ValueError("prefill hidden states must match controller masks")
        qmask = self.query_mask.to(hidden_states.device)
        denom = qmask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden_states.dtype)
        query_repr = (hidden_states * qmask[..., None]).sum(dim=1) / denom
        self.gate = self.router(
            query_repr,
            hidden_states,
            self.audio_mask.to(hidden_states.device),
        )
        return self.gate

    def gate_for_key_length(self, key_length: int) -> torch.Tensor:
        if self.gate is None:
            if self.audio_mask is None:
                raise RuntimeError("controller masks are not initialized")
            base = torch.ones_like(self.audio_mask, dtype=torch.float32)
        else:
            base = self.gate
        if key_length < base.shape[1]:
            return base[:, :key_length]
        if key_length == base.shape[1]:
            return base
        pad = torch.ones(base.shape[0], key_length - base.shape[1], device=base.device, dtype=base.dtype)
        return torch.cat([base, pad], dim=1)

    def audio_mask_for_key_length(self, key_length: int, device=None) -> torch.Tensor:
        if self.audio_mask is None:
            raise RuntimeError("controller masks are not initialized")
        base = self.audio_mask.to(device=device or self.audio_mask.device)
        if key_length < base.shape[1]:
            return base[:, :key_length]
        if key_length == base.shape[1]:
            return base
        pad = torch.zeros(base.shape[0], key_length - base.shape[1], device=base.device, dtype=torch.bool)
        return torch.cat([base, pad], dim=1)


def install_qacr_on_qwen_attention(attn_module: nn.Module, controller: QACRController):
    """Patch one Qwen2.5-Omni Thinker self-attention module in eager mode.

    The implementation mirrors the public Transformers Qwen2_5OmniAttention forward path,
    but replaces the eager value aggregation with QACR contribution gating. Returns a
    callable that restores the original forward method.
    """
    required = ["q_proj", "k_proj", "v_proj", "o_proj", "num_heads", "head_dim", "num_key_value_heads", "num_key_value_groups", "scaling", "config"]
    missing = [name for name in required if not hasattr(attn_module, name)]
    if missing:
        raise TypeError(f"unsupported Qwen attention module; missing {missing}")
    if getattr(attn_module.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("QACR currently requires attn_implementation='eager' for exact contribution routing")

    try:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb
    except ImportError as exc:  # pragma: no cover - GPU optional dependency
        raise RuntimeError("Install sar[gpu] with a Qwen2.5-Omni-capable Transformers version") from exc

    original_forward = attn_module.forward

    def qacr_forward(
        module_self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings=None,
        **kwargs,
    ):
        del output_attentions, use_cache
        bsz, q_len, _ = hidden_states.size()
        if controller.audio_mask is not None and controller.gate is None and q_len == controller.audio_mask.shape[1]:
            controller.compute_prefill_gate(hidden_states)

        query_states = module_self.q_proj(hidden_states)
        key_states = module_self.k_proj(hidden_states)
        value_states = module_self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, module_self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, module_self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, module_self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            raise ValueError("QACR Qwen attention requires position_embeddings from the Thinker decoder")
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            module_self.config.rope_parameters["mrope_section"],
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, module_self.layer_idx)
        key_length = key_states.shape[-2]
        gate = controller.gate_for_key_length(key_length).to(query_states.device, dtype=query_states.dtype)
        amask = controller.audio_mask_for_key_length(key_length, device=query_states.device)
        attn_output, attn_weights = qacr_eager_attention_forward(
            module_self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=module_self.scaling,
            dropout=0.0 if not module_self.training else module_self.attention_dropout,
            audio_mask=amask,
            gate=gate,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = module_self.o_proj(attn_output)
        return attn_output, attn_weights

    attn_module.forward = types.MethodType(qacr_forward, attn_module)

    def restore() -> None:
        attn_module.forward = original_forward

    return restore
