from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import RelevancePairRecord, validate_pair_records


def pair_by_id(records: list[RelevancePairRecord]) -> dict[str, dict[str, RelevancePairRecord]]:
    validate_pair_records(records)
    out: dict[str, dict[str, RelevancePairRecord]] = {}
    for record in records:
        out.setdefault(record.pair_id, {})[record.role] = record
    return out


def load_jsonl_records(path: str | Path) -> list[RelevancePairRecord]:
    path = Path(path)
    records: list[RelevancePairRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(RelevancePairRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid relevance record at {path}:{line_no}") from exc
    return records


def write_jsonl_records(path: str | Path, records: Iterable[RelevancePairRecord]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    return path
