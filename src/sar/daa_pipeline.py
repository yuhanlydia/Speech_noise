from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from sar.data.blocks import AcousticBlock, fixed_temporal_blocks
from sar.metrics import compute_pair_metrics


class DAAStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _block_overlaps_span(block: AcousticBlock, start_s: float, end_s: float) -> bool:
    return max(block.start_s, start_s) < min(block.end_s, end_s)


def _event_selected(record, blocks: Sequence[AcousticBlock], selected_ids: Sequence[str]) -> bool | None:
    if not getattr(record, "source_mask_valid", False):
        return None
    start_s = getattr(record, "source_start_s", None)
    end_s = getattr(record, "source_end_s", None)
    if start_s is None or end_s is None:
        return None
    selected = set(selected_ids)
    return any(
        block.block_id in selected
        and _block_overlaps_span(block, float(start_s), float(end_s))
        for block in blocks
    )


def run_daa_pair(
    wrapper,
    records,
    *,
    duration_s: float,
    block_strategy: str,
    block_seconds: float,
    max_blocks: int,
    max_focus_blocks: int,
    scan_max_new_tokens: int,
    select_max_new_tokens: int,
    layers: list[int],
) -> list[dict]:
    records = list(records)
    if len(records) != 2 or {r.role for r in records} != {"use", "ignore"}:
        raise ValueError("run_daa_pair expects exactly one use and one ignore record")
    if len({r.waveform_sha256 for r in records}) != 1:
        raise ValueError("DAA pair must use the exact same waveform hash")

    audio_path = records[0].waveform_path
    raw_blocks: str | None = None
    if block_strategy == "declared":
        try:
            blocks, raw_blocks = wrapper.declare_audio_blocks(
                audio_path,
                duration_s=duration_s,
                max_blocks=max_blocks,
                max_new_tokens=scan_max_new_tokens,
            )
        except Exception as exc:
            raise DAAStageError("block_declaration", str(exc)) from exc
    elif block_strategy == "fixed":
        blocks = fixed_temporal_blocks(
            duration_s,
            block_seconds=block_seconds,
            max_blocks=max_blocks,
        )
    else:
        raise ValueError(f"unsupported DAA block_strategy: {block_strategy!r}")

    rows: list[dict] = []
    for record in records:
        try:
            selected_ids, raw_focus = wrapper.select_audio_blocks(
                audio_path,
                record.query,
                blocks,
                max_selected=max_focus_blocks,
                max_new_tokens=select_max_new_tokens,
            )
        except Exception as exc:
            raise DAAStageError(f"focus_{record.role}", str(exc)) from exc

        if not record.options:
            raise ValueError(f"{record.pair_id}/{record.role}: DAA MVP requires canonical options")
        try:
            scores = wrapper.score_single_token_options_daa(
                audio_path,
                record.query,
                record.options,
                blocks,
                selected_ids,
                layers=layers,
            )
        except Exception as exc:
            raise DAAStageError(f"focused_reasoning_{record.role}", str(exc)) from exc

        rows.append(
            {
                "pair_id": record.pair_id,
                "role": record.role,
                "waveform_sha256": record.waveform_sha256,
                "answer": record.answer,
                "prediction": scores.predicted_option,
                "correct": scores.predicted_option == record.answer,
                "selected_blocks": list(selected_ids),
                "event_selected": _event_selected(record, blocks, selected_ids),
                "block_strategy": block_strategy,
                "blocks": [block.__dict__ for block in blocks],
                "raw_block_declaration": raw_blocks,
                "raw_focus_declaration": raw_focus,
            }
        )
    return rows


def summarize_daa_rows(rows: Sequence[dict]) -> dict:
    summary = compute_pair_metrics(rows)
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_pair[str(row["pair_id"])][str(row["role"])] = row

    switchable = []
    for pair in by_pair.values():
        if set(pair) != {"ignore", "use"}:
            continue
        use_sel = pair["use"].get("event_selected")
        ignore_sel = pair["ignore"].get("event_selected")
        if use_sel is None or ignore_sel is None:
            continue
        switchable.append(bool(use_sel) and not bool(ignore_sel))

    summary["selection_switch_acc"] = (
        float(sum(switchable) / len(switchable)) if switchable else None
    )
    summary["selection_labeled_pairs"] = float(len(switchable))

    use_labeled = [
        row for row in rows
        if row.get("role") == "use" and row.get("event_selected") is not None
    ]
    ignore_labeled = [
        row for row in rows
        if row.get("role") == "ignore" and row.get("event_selected") is not None
    ]
    summary["use_event_selection_rate"] = (
        float(sum(bool(r["event_selected"]) for r in use_labeled) / len(use_labeled))
        if use_labeled
        else None
    )
    summary["ignore_event_avoid_rate"] = (
        float(sum(not bool(r["event_selected"]) for r in ignore_labeled) / len(ignore_labeled))
        if ignore_labeled
        else None
    )

    use_selected = [r for r in use_labeled if bool(r["event_selected"])]
    ignore_avoided = [r for r in ignore_labeled if not bool(r["event_selected"])]
    summary["reasoning_acc_given_use_selection"] = (
        float(sum(bool(r["correct"]) for r in use_selected) / len(use_selected))
        if use_selected
        else None
    )
    summary["reasoning_acc_given_ignore_avoidance"] = (
        float(sum(bool(r["correct"]) for r in ignore_avoided) / len(ignore_avoided))
        if ignore_avoided
        else None
    )
    return summary
