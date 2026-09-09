from __future__ import annotations

import types
from dataclasses import dataclass

import torch
from torch import nn

from sar.methods.daa import daa_eager_attention_forward


def post_audio_consumer_mask(audio_mask: torch.Tensor) -> torch.Tensor:
    if audio_mask.ndim != 2 or audio_mask.dtype is not torch.bool:
        raise ValueError("audio_mask must be boolean [batch, sequence]")
    out = torch.zeros_like(audio_mask)
    for b in range(audio_mask.shape[0]):
        positions = torch.where(audio_mask[b])[0]
        if positions.numel() == 0:
            continue
        out[b, int(positions.max().item()) + 1 :] = True
    return out


@dataclass
class DAAController:
    audio_mask: torch.Tensor | None = None
    allowed_audio_mask: torch.Tensor | None = None
    consumer_mask: torch.Tensor | None = None

    def set_masks(
        self,
        audio_mask: torch.Tensor,
        allowed_audio_mask: torch.Tensor,
        consumer_mask: torch.Tensor,
    ) -> None:
        for name, mask in {
            "audio_mask": audio_mask,
            "allowed_audio_mask": allowed_audio_mask,
            "consumer_mask": consumer_mask,
        }.items():
            if mask.ndim != 2 or mask.dtype is not torch.bool:
                raise ValueError(f"{name} must be boolean [batch, sequence]")
        if audio_mask.shape != allowed_audio_mask.shape or audio_mask.shape != consumer_mask.shape:
            raise ValueError("all DAA masks must share [batch, sequence] shape")
        if torch.any(allowed_audio_mask & ~audio_mask):
            raise ValueError("allowed_audio_mask may only select audio positions")
        self.audio_mask = audio_mask
        self.allowed_audio_mask = allowed_audio_mask
        self.consumer_mask = consumer_mask

    @staticmethod
    def _extend(mask: torch.Tensor, length: int, *, fill: bool) -> torch.Tensor:
        if length < mask.shape[1]:
            return mask[:, :length]
        if length == mask.shape[1]:
            return mask
        pad = torch.full(
            (mask.shape[0], length - mask.shape[1]),
            fill,
            dtype=torch.bool,
            device=mask.device,
        )
        return torch.cat([mask, pad], dim=1)

    def key_masks_for_length(
        self,
        key_length: int,
        device=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.audio_mask is None or self.allowed_audio_mask is None:
            raise RuntimeError("DAA controller masks are not initialized")
        audio = self._extend(self.audio_mask, key_length, fill=False)
        allowed = self._extend(self.allowed_audio_mask, key_length, fill=False)
        if device is not None:
            audio = audio.to(device)
            allowed = allowed.to(device)
        return audio, allowed

    def consumer_mask_for_query_length(self, query_length: int, device=None) -> torch.Tensor:
        if self.consumer_mask is None:
            raise RuntimeError("DAA controller masks are not initialized")
        if query_length == self.consumer_mask.shape[1]:
            mask = self.consumer_mask
        elif query_length < self.consumer_mask.shape[1]:
            # Cached generation normally has q_len=1; new generated tokens consume the focus.
            mask = torch.ones(
                (self.consumer_mask.shape[0], query_length),
                dtype=torch.bool,
                device=self.consumer_mask.device,
            )
        else:
            mask = self._extend(self.consumer_mask, query_length, fill=True)
        return mask.to(device=device or mask.device)


def install_daa_on_qwen_attention(attn_module: nn.Module, controller: DAAController):
    required = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "num_heads",
        "head_dim",
        "num_key_value_heads",
        "num_key_value_groups",
        "scaling",
        "config",
    ]
    missing = [name for name in required if not hasattr(attn_module, name)]
    if missing:
        raise TypeError(f"unsupported Qwen attention module; missing {missing}")
    if getattr(attn_module.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("DAA requires attn_implementation='eager' for exact block masking")

    try:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            apply_multimodal_rotary_pos_emb,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install sar[gpu] with Qwen2.5-Omni Transformers support"
        ) from exc

    original_forward = attn_module.forward

    def daa_forward(
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
        query_states = module_self.q_proj(hidden_states)
        key_states = module_self.k_proj(hidden_states)
        value_states = module_self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, module_self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, module_self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, module_self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            raise ValueError("DAA Qwen attention requires position_embeddings")
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            module_self.config.rope_parameters["mrope_section"],
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                module_self.layer_idx,
            )
        key_len = key_states.shape[-2]
        audio_mask, allowed = controller.key_masks_for_length(
            key_len,
            device=query_states.device,
        )
        consumer = controller.consumer_mask_for_query_length(
            q_len,
            device=query_states.device,
        )
        attn_output, attn_weights = daa_eager_attention_forward(
            module_self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=module_self.scaling,
            dropout=0.0 if not module_self.training else module_self.attention_dropout,
            audio_mask=audio_mask,
            allowed_audio_mask=allowed,
            consumer_mask=consumer,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = module_self.o_proj(attn_output)
        return attn_output, attn_weights

    attn_module.forward = types.MethodType(daa_forward, attn_module)

    def restore() -> None:
        attn_module.forward = original_forward

    return restore


def install_daa_on_qwen_layers(attn_modules: list[nn.Module], controller: DAAController):
    restorers = [
        install_daa_on_qwen_attention(module, controller)
        for module in attn_modules
    ]

    def restore_all() -> None:
        for restore in reversed(restorers):
            restore()

    return restore_all
