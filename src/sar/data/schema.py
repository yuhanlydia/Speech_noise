from __future__ import annotations

from collections import defaultdict
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field


class RelevancePairRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str = Field(min_length=1)
    role: Literal["use", "ignore"]
    waveform_path: str = Field(min_length=1)
    waveform_sha256: str = Field(min_length=1)
    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    options: list[str] | None = None
    event_type: str = Field(min_length=1)
    source_id: str | None = None
    source_start_s: float | None = None
    source_end_s: float | None = None
    source_mask_valid: bool = False
    seed: int = 0
    snr_db: float | None = None
    tir_db: float | None = None
    sample_rate: int | None = None
    offset_samples: int | None = None


def validate_pair_records(records: Sequence[RelevancePairRecord]) -> None:
    grouped: dict[str, list[RelevancePairRecord]] = defaultdict(list)
    for record in records:
        grouped[record.pair_id].append(record)
    for pair_id, items in grouped.items():
        roles = sorted(item.role for item in items)
        if len(items) != 2 or roles != ["ignore", "use"]:
            raise ValueError(f"pair {pair_id!r} must contain exactly one use and one ignore record")
        hashes = {item.waveform_sha256 for item in items}
        if len(hashes) != 1:
            raise ValueError(f"pair {pair_id!r} has mismatched waveform hash values")
