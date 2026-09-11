from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from sar.data.schema import RelevancePairRecord, validate_pair_records
from sar.models.base import OptionScores


def load_source_rows(path: str | Path) -> dict[str, dict]:
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
            if not row.get("target_wav") or not row.get("event_wav"):
                raise ValueError(
                    f"{pair_id}: source row requires target_wav and event_wav"
                )
            out[pair_id] = row
    return out


def _score(wrapper, audio_path: str, record) -> OptionScores:
    if not record.options:
        raise ValueError(f"{record.pair_id}/{record.role}: options are required")
    # The public V2 protocol deliberately uses canonical A/B/C/D options. If the
    # canonical scorer exists but fails, propagate that failure instead of hiding a
    # tokenizer/model integration bug behind a slower scoring fallback.
    scorer = getattr(wrapper, "score_single_token_options", None)
    if scorer is not None:
        return scorer(audio_path, record.query, record.options)
    return wrapper.score_options(audio_path, record.query, record.options)


def evaluate_capability_pairs(
    wrapper,
    records: Sequence[RelevancePairRecord],
    source_manifest: str | Path,
) -> list[dict]:
    """Evaluate the two component tasks in isolation before testing relevance switching.

    A pair is eligible only when the same backbone can solve the target-speech task on
    target audio alone and the event task on event audio alone. This prevents ordinary
    capability failure from being mislabelled as a relevance-routing failure.
    """
    records = list(records)
    validate_pair_records(records)
    sources = load_source_rows(source_manifest)
    grouped: dict[str, dict[str, RelevancePairRecord]] = defaultdict(dict)
    for record in records:
        grouped[record.pair_id][record.role] = record

    rows: list[dict] = []
    for pair_id, pair in grouped.items():
        if pair_id not in sources:
            raise ValueError(f"{pair_id}: missing from source manifest")
        ignore = pair["ignore"]
        use = pair["use"]
        source = sources[pair_id]
        target_scores = _score(wrapper, str(source["target_wav"]), ignore)
        event_scores = _score(wrapper, str(source["event_wav"]), use)
        target_ok = target_scores.predicted_option == ignore.answer
        event_ok = event_scores.predicted_option == use.answer
        rows.append(
            {
                "pair_id": pair_id,
                "target_wav": str(source["target_wav"]),
                "event_wav": str(source["event_wav"]),
                "target_only_prediction": target_scores.predicted_option,
                "target_only_answer": ignore.answer,
                "target_only_correct": bool(target_ok),
                "event_only_prediction": event_scores.predicted_option,
                "event_only_answer": use.answer,
                "event_only_correct": bool(event_ok),
                "eligible": bool(target_ok and event_ok),
            }
        )
    return rows


def filter_eligible_records(
    records: Sequence[RelevancePairRecord], capability_rows: Sequence[dict]
) -> list[RelevancePairRecord]:
    eligible = {
        str(row["pair_id"])
        for row in capability_rows
        if bool(row["eligible"])
    }
    return [record for record in records if record.pair_id in eligible]


def summarize_capability_rows(rows: Sequence[dict]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("capability rows cannot be empty")
    n = len(rows)
    target = sum(bool(row["target_only_correct"]) for row in rows)
    event = sum(bool(row["event_only_correct"]) for row in rows)
    eligible = sum(bool(row["eligible"]) for row in rows)
    return {
        "num_pairs": n,
        "target_only_acc": target / n,
        "event_only_acc": event / n,
        "eligible_pairs": eligible,
        "eligibility_rate": eligible / n,
    }


def write_capability_outputs(
    rows: Sequence[dict],
    eligible_records: Iterable[RelevancePairRecord],
    *,
    output_dir: str | Path,
    eligible_manifest: str | Path,
) -> tuple[Path, Path, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "results.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = summarize_capability_rows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    eligible_manifest = Path(eligible_manifest)
    eligible_manifest.parent.mkdir(parents=True, exist_ok=True)
    eligible_manifest.write_text(
        "".join(record.model_dump_json() + "\n" for record in eligible_records),
        encoding="utf-8",
    )
    return rows_path, eligible_manifest, summary
