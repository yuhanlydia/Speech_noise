from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from sar.data.mixing import sha256_waveform
from sar.data.relevance_pairs import load_jsonl_records
from sar.data.schema import validate_pair_records


def _load_source_rows(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = str(row.get("pair_id", ""))
            if not pair_id:
                raise ValueError(f"missing pair_id at {path}:{line_no}")
            if pair_id in out:
                raise ValueError(f"duplicate pair_id in source manifest: {pair_id}")
            out[pair_id] = row
    return out


def _audio_info(path: str | Path) -> tuple[np.ndarray, int, float]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    audio, sr = sf.read(p, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError(f"unsupported audio rank at {p}: {audio.ndim}")
    duration = float(len(audio) / int(sr)) if sr else 0.0
    return np.asarray(audio, dtype=np.float32), int(sr), duration


def audit_public_mvp(
    source_manifest: str | Path,
    pairs_manifest: str | Path,
    *,
    tolerance_s: float = 2e-3,
) -> dict:
    """Audit generated same-waveform pairs before any model experiment."""
    sources = _load_source_rows(source_manifest)
    records = load_jsonl_records(pairs_manifest)
    errors: list[str] = []
    try:
        validate_pair_records(records)
    except Exception as exc:
        errors.append(f"pair schema: {exc}")

    grouped: dict[str, list] = defaultdict(list)
    for record in records:
        grouped[record.pair_id].append(record)

    source_ids = set(sources)
    pair_ids = set(grouped)
    for missing in sorted(source_ids - pair_ids):
        errors.append(f"{missing}: source row has no pair records")
    for missing in sorted(pair_ids - source_ids):
        errors.append(f"{missing}: pair records have no source row")

    pair_reports: list[dict] = []
    for pair_id in sorted(source_ids & pair_ids):
        src = sources[pair_id]
        items = grouped[pair_id]
        local_errors: list[str] = []
        try:
            target, target_sr, target_dur = _audio_info(src["target_wav"])
            event, event_sr, event_dur = _audio_info(src["event_wav"])
        except Exception as exc:
            errors.append(f"{pair_id}: source audio read failed: {exc}")
            continue

        if target_sr != event_sr:
            local_errors.append(
                f"source sample-rate mismatch target={target_sr} event={event_sr}"
            )
        if not np.isfinite(target).all() or not np.isfinite(event).all():
            local_errors.append("source audio contains NaN/Inf")

        mixed_paths = {str(item.waveform_path) for item in items}
        mixed_hashes = {str(item.waveform_sha256) for item in items}
        if len(mixed_paths) != 1:
            local_errors.append("use/ignore waveform paths differ")
        if len(mixed_hashes) != 1:
            local_errors.append("use/ignore waveform hashes differ")

        mixed = None
        mixed_sr = None
        mixed_dur = None
        if len(mixed_paths) == 1:
            try:
                mixed, mixed_sr, mixed_dur = _audio_info(next(iter(mixed_paths)))
                if not np.isfinite(mixed).all():
                    local_errors.append("mixed audio contains NaN/Inf")
                if float(np.max(np.abs(mixed))) > 1.0001:
                    local_errors.append("mixed audio peak exceeds 1.0")
                recomputed = sha256_waveform(mixed, mixed_sr)
                expected = next(iter(mixed_hashes)) if mixed_hashes else ""
                if recomputed != expected:
                    local_errors.append(
                        f"waveform_sha256 mismatch expected={expected} recomputed={recomputed}"
                    )
            except Exception as exc:
                local_errors.append(f"mixed audio read/hash failed: {exc}")

        sample_rates = {item.sample_rate for item in items if item.sample_rate is not None}
        if len(sample_rates) > 1:
            local_errors.append("use/ignore sample_rate metadata differs")
        if sample_rates and target_sr not in sample_rates:
            local_errors.append(
                f"sample_rate metadata {sorted(sample_rates)} != source rate {target_sr}"
            )
        if mixed_sr is not None and mixed_sr != target_sr:
            local_errors.append(
                f"mixed sample rate {mixed_sr} != source rate {target_sr}"
            )

        offsets = {item.offset_samples for item in items if item.offset_samples is not None}
        starts = {item.source_start_s for item in items if item.source_start_s is not None}
        ends = {item.source_end_s for item in items if item.source_end_s is not None}
        if len(offsets) != 1 or len(starts) != 1 or len(ends) != 1:
            local_errors.append("source interval metadata missing or inconsistent")
            offset = None
            start = None
            end = None
        else:
            offset = int(next(iter(offsets)))
            start = float(next(iter(starts)))
            end = float(next(iter(ends)))
            expected_start = offset / target_sr
            if abs(start - expected_start) > tolerance_s:
                local_errors.append(
                    f"offset/source_start_s mismatch offset={expected_start:.6f}s start={start:.6f}s"
                )
            if abs((end - start) - event_dur) > tolerance_s:
                local_errors.append(
                    f"event span duration mismatch span={end-start:.6f}s event={event_dur:.6f}s"
                )
            if start + tolerance_s < target_dur:
                local_errors.append(
                    f"event starts before target ends start={start:.6f}s target={target_dur:.6f}s"
                )
            if mixed_dur is not None and (
                start < -tolerance_s or end > mixed_dur + tolerance_s or end <= start
            ):
                local_errors.append(
                    f"invalid event interval [{start:.6f},{end:.6f}] for mixed duration {mixed_dur:.6f}s"
                )

        errors.extend(f"{pair_id}: {err}" for err in local_errors)
        pair_reports.append(
            {
                "pair_id": pair_id,
                "ok": not local_errors,
                "target_duration_s": target_dur,
                "event_duration_s": event_dur,
                "mixed_duration_s": mixed_dur,
                "source_start_s": start,
                "source_end_s": end,
                "gap_s": None if start is None else float(start - target_dur),
                "sample_rate": target_sr,
                "errors": local_errors,
            }
        )

    durations = [
        row["mixed_duration_s"]
        for row in pair_reports
        if row["mixed_duration_s"] is not None
    ]
    gaps = [row["gap_s"] for row in pair_reports if row["gap_s"] is not None]
    return {
        "ok": len(errors) == 0,
        "num_pairs": len(pair_reports),
        "num_errors": len(errors),
        "errors": errors,
        "mixed_duration_mean_s": float(np.mean(durations)) if durations else None,
        "mixed_duration_max_s": float(np.max(durations)) if durations else None,
        "gap_mean_s": float(np.mean(gaps)) if gaps else None,
        "pairs": pair_reports,
    }
