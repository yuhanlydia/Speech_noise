from __future__ import annotations

import math

import torch
from torch import nn


def _expand_token_weight(weight: torch.Tensor, attn_probs: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("gate/mask weights must have shape [batch, key_tokens]")
    return weight[:, None, None, :].to(dtype=attn_probs.dtype, device=attn_probs.device)


def decompose_attention_contributions(
    attn_probs: torch.Tensor,
    values: torch.Tensor,
    audio_mask: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return text/non-audio contribution and gated audio contribution.

    Shapes: attn_probs [B,H,Q,K], values [B,H,K,D], audio_mask/gate [B,K].
    """
    if attn_probs.ndim != 4 or values.ndim != 4:
        raise ValueError("attention probabilities and values must be rank-4")
    if attn_probs.shape[0] != values.shape[0] or attn_probs.shape[1] != values.shape[1]:
        raise ValueError("batch/head dimensions must align")
    if attn_probs.shape[-1] != values.shape[-2]:
        raise ValueError("attention key dimension must match value sequence dimension")
    if audio_mask.dtype is not torch.bool:
        raise ValueError("audio_mask must be boolean")
    if audio_mask.shape != gate.shape or audio_mask.shape != (attn_probs.shape[0], attn_probs.shape[-1]):
        raise ValueError("audio_mask and gate must have shape [batch, key_tokens]")

    audio_w = _expand_token_weight(audio_mask.to(attn_probs.dtype) * gate, attn_probs)
    text_w = _expand_token_weight((~audio_mask).to(attn_probs.dtype), attn_probs)
    text_attn = attn_probs * text_w
    audio_attn = attn_probs * audio_w
    text_contrib = torch.matmul(text_attn, values)
    audio_contrib = torch.matmul(audio_attn, values)
    return text_contrib, audio_contrib


def routed_attention_output(
    attn_probs: torch.Tensor,
    values: torch.Tensor,
    audio_mask: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    text_contrib, audio_contrib = decompose_attention_contributions(attn_probs, values, audio_mask, gate)
    return text_contrib + audio_contrib


class QACRRouter(nn.Module):
    def __init__(self, query_dim: int, audio_dim: int, router_dim: int = 32):
        super().__init__()
        if router_dim <= 0:
            raise ValueError("router_dim must be positive")
        self.router_dim = int(router_dim)
        self.query_proj = nn.Linear(query_dim, router_dim, bias=False)
        self.audio_proj = nn.Linear(audio_dim, router_dim, bias=False)

    def forward(
        self,
        query_repr: torch.Tensor,
        audio_repr: torch.Tensor,
        audio_mask: torch.Tensor,
        *,
        identity_override: bool = False,
    ) -> torch.Tensor:
        if audio_mask.dtype is not torch.bool:
            raise ValueError("audio_mask must be boolean")
        if audio_repr.shape[:-1] != audio_mask.shape:
            raise ValueError("audio_mask must match audio_repr")
        if query_repr.shape[0] != audio_repr.shape[0]:
            raise ValueError("query/audio batch sizes must match")
        if identity_override:
            return torch.ones_like(audio_mask, dtype=audio_repr.dtype)
        dtype = self.query_proj.weight.dtype
        q = self.query_proj(query_repr.to(dtype=dtype))
        a = self.audio_proj(audio_repr.to(dtype=dtype))
        scores = (q[:, None, :] * a).sum(dim=-1) / math.sqrt(self.router_dim)
        gates = torch.sigmoid(scores)
        return torch.where(audio_mask, gates, torch.ones_like(gates))

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
