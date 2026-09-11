from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence

import torch


_BLOCK_TAG_RE = re.compile(r"<audio_blocks>(.*?)</audio_blocks>", re.IGNORECASE | re.DOTALL)
_BLOCK_LINE_RE = re.compile(
    r"^\s*(B\d+)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*(.+?)\s*$"
)


@dataclass(frozen=True)
class AcousticBlock:
    block_id: str
    start_s: float
    end_s: float
    label: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"B\d+", self.block_id):
            raise ValueError(f"invalid block id: {self.block_id!r}")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("block must satisfy 0 <= start < end")
        if not self.label.strip():
            raise ValueError("block label cannot be empty")


def validate_blocks(blocks: Sequence[AcousticBlock], *, duration_s: float, max_blocks: int) -> None:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if max_blocks < 1:
        raise ValueError("max_blocks must be >= 1")
    if not blocks:
        raise ValueError("at least one acoustic block is required")
    if len(blocks) > max_blocks:
        raise ValueError(f"declared {len(blocks)} blocks but max_blocks={max_blocks}")
    ids = [b.block_id for b in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate block ids are not allowed")

    # The declared block table is DAA's address space. If it omits a large
    # region, the runtime would silently treat undeclared audio as unselected.
    # Require a near-complete partition so segmentation omissions are visible
    # protocol failures rather than hidden sources of apparent improvement.
    coverage_tolerance_s = min(0.25, max(0.05, 0.05 * float(duration_s)))
    if blocks[0].start_s > coverage_tolerance_s:
        raise ValueError("declared blocks do not cover the audio start")

    previous_end = 0.0
    for i, block in enumerate(blocks):
        if block.end_s > duration_s + 1e-6:
            raise ValueError(f"block {block.block_id} ends beyond audio duration")
        if i and block.start_s < previous_end - 1e-6:
            raise ValueError(f"block {block.block_id} overlap detected")
        if i and block.start_s - previous_end > coverage_tolerance_s:
            raise ValueError("declared blocks do not cover the audio continuously")
        previous_end = block.end_s

    if duration_s - blocks[-1].end_s > coverage_tolerance_s:
        raise ValueError("declared blocks do not cover the audio end")


def parse_audio_blocks(text: str, *, duration_s: float, max_blocks: int) -> list[AcousticBlock]:
    match = _BLOCK_TAG_RE.search(text)
    if not match:
        raise ValueError("missing <audio_blocks>...</audio_blocks> declaration")
    blocks: list[AcousticBlock] = []
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line:
            continue
        parsed = _BLOCK_LINE_RE.match(line)
        if not parsed:
            raise ValueError(f"invalid audio block line: {line!r}")
        block_id, start, end, label = parsed.groups()
        blocks.append(AcousticBlock(block_id, float(start), float(end), label.strip()))
    # The scan prompt displays duration to two decimal places. Interpret that
    # exact rounded-up final endpoint as the real audio end; do not broaden
    # validation for arbitrary overflows or change already-valid declarations.
    if blocks:
        last = blocks[-1]
        displayed_end = float(f"{duration_s:.2f}")
        if (
            last.end_s > duration_s + 1e-6
            and last.end_s == displayed_end
            and 0 < displayed_end - duration_s <= 0.005 + 1e-9
            and last.start_s < duration_s
        ):
            blocks[-1] = AcousticBlock(last.block_id, last.start_s, duration_s, last.label)
    validate_blocks(blocks, duration_s=duration_s, max_blocks=max_blocks)
    return blocks


def fixed_temporal_blocks(duration_s: float, *, block_seconds: float, max_blocks: int) -> list[AcousticBlock]:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if block_seconds <= 0:
        raise ValueError("block_seconds must be positive")
    needed = int(math.ceil(duration_s / block_seconds))
    if needed > max_blocks:
        block_seconds = duration_s / max_blocks
        needed = max_blocks
    blocks: list[AcousticBlock] = []
    for idx in range(needed):
        start = idx * block_seconds
        end = min(duration_s, (idx + 1) * block_seconds)
        blocks.append(
            AcousticBlock(
                f"B{idx + 1}",
                round(start, 6),
                round(end, 6),
                f"audio segment {idx + 1}",
            )
        )
    return blocks


def format_block_table(blocks: Sequence[AcousticBlock]) -> str:
    return "\n".join(
        f"{b.block_id}: {b.start_s:.2f}-{b.end_s:.2f}s | {b.label}" for b in blocks
    )


def blocks_to_audio_token_mask(
    audio_mask: torch.Tensor,
    *,
    duration_s: float,
    blocks: Sequence[AcousticBlock],
    selected_ids: Iterable[str],
) -> torch.Tensor:
    if audio_mask.ndim != 2 or audio_mask.dtype is not torch.bool:
        raise ValueError("audio_mask must be boolean [batch, sequence]")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    selected = set(selected_ids)
    block_by_id = {b.block_id: b for b in blocks}
    unknown = selected - set(block_by_id)
    if unknown:
        raise ValueError(f"unknown selected block ids: {sorted(unknown)}")
    out = torch.zeros_like(audio_mask)
    for batch_idx in range(audio_mask.shape[0]):
        positions = torch.where(audio_mask[batch_idx])[0]
        if positions.numel() == 0:
            continue
        centers = (
            torch.arange(positions.numel(), device=positions.device, dtype=torch.float32) + 0.5
        ) / positions.numel() * float(duration_s)
        token_selected = torch.zeros_like(centers, dtype=torch.bool)
        for block_id in selected:
            block = block_by_id[block_id]
            token_selected |= (centers >= block.start_s) & (centers < block.end_s + 1e-9)
        out[batch_idx, positions[token_selected]] = True
    return out
