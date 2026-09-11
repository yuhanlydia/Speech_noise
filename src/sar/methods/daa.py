from __future__ import annotations

import re
from typing import Sequence

import torch
from torch import nn

from sar.data.blocks import AcousticBlock, format_block_table


_FOCUS_RE = re.compile(r'<focus_audio\s+blocks\s*=\s*"([^"]+)"\s*/?>', re.IGNORECASE)


def format_scan_prompt(*, duration_s: float, max_blocks: int) -> str:
    return (
        "Listen to the complete audio once and divide it into a small set of semantically coherent, "
        "time-local acoustic blocks. Do not answer any downstream question. "
        f"The audio is {duration_s:.2f} seconds long and you may declare at most {max_blocks} blocks. "
        "Return exactly:\n<audio_blocks>\nB1|START_SECONDS|END_SECONDS|SHORT_LABEL\n...\n</audio_blocks>. "
        "Use non-overlapping chronological spans. Labels should describe audible events or speakers, not relevance."
    )


def format_focus_prompt(query: str, blocks: Sequence[AcousticBlock], *, max_selected: int) -> str:
    return (
        "You have already listened globally. Below are addressable acoustic blocks from that same waveform.\n"
        f"{format_block_table(blocks)}\n\n"
        f"Question: {query}\n"
        f"Declare only the block(s) needed to answer this question, at most {max_selected}. "
        "Do not answer the question yet. Return exactly one tag such as "
        '<focus_audio blocks="B2"> or <focus_audio blocks="B1,B3">.'
    )


def parse_focus_declaration(text: str, *, valid_block_ids: set[str], max_selected: int) -> list[str]:
    if max_selected < 1:
        raise ValueError("max_selected must be >= 1")
    match = _FOCUS_RE.search(text)
    if not match:
        raise ValueError("missing <focus_audio blocks=...> declaration")
    ids = [part.strip() for part in match.group(1).split(",") if part.strip()]
    if not ids:
        raise ValueError("focus declaration selected no blocks")
    if len(ids) > max_selected:
        raise ValueError(f"focus declaration selected {len(ids)} blocks; max_selected={max_selected}")
    if len(set(ids)) != len(ids):
        raise ValueError("focus declaration contains duplicate block ids")
    unknown = set(ids) - valid_block_ids
    if unknown:
        raise ValueError(f"focus declaration contains unknown block ids: {sorted(unknown)}")
    return ids


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def daa_eager_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    *,
    audio_mask: torch.Tensor | None = None,
    allowed_audio_mask: torch.Tensor | None = None,
    consumer_mask: torch.Tensor | None = None,
    **kwargs,
):
    del kwargs
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    if audio_mask is not None or allowed_audio_mask is not None or consumer_mask is not None:
        if audio_mask is None or allowed_audio_mask is None or consumer_mask is None:
            raise ValueError(
                "audio_mask, allowed_audio_mask, and consumer_mask must be provided together"
            )
        if audio_mask.shape != allowed_audio_mask.shape:
            raise ValueError("audio_mask and allowed_audio_mask shapes must match")
        if audio_mask.shape[1] != attn_weights.shape[-1]:
            raise ValueError("audio key mask length must match attention key length")
        if consumer_mask.shape != (attn_weights.shape[0], attn_weights.shape[-2]):
            raise ValueError("consumer_mask must have shape [batch, query_length]")
        excluded_audio = audio_mask & ~allowed_audio_mask
        blocked = consumer_mask[:, None, :, None] & excluded_audio[:, None, None, :]
        attn_weights = attn_weights.masked_fill(
            blocked,
            torch.finfo(attn_weights.dtype).min,
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights,
        p=dropout,
        training=module.training,
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
