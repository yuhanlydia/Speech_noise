from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from sar.data.blocks import AcousticBlock, fixed_temporal_blocks
from sar.metrics import compute_pair_metrics


class DAAStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _event_selected(
    record, blocks: Sequence[AcousticBlock], selected_ids: Sequence[str]
) -> bool | None:
    if not getattr(record, "source_mask_valid", False):
        return None
    start_s = getattr(record, "source_start_s", None)
    end_s = getattr(record, "source_end_s", None)
    if start_s is None or end_s is None:
        return None
    midpoint = (float(start_s) + float(end_s)) / 2.0
    selected = set(selected_ids)
    return any(
        block.block_id in selected and block.start_s <= midpoint < block.end_s
        for block in blocks
    )


def _target_coverage(
    record, blocks: Sequence[AcousticBlock], selected_ids: Sequence[str]
) -> float | None:
    """Fraction of the pre-event target interval retained by selected blocks."""
    start_s = getattr(record, "source_start_s", None)
    if start_s is None or float(start_s) <= 0:
        return None
    target_end = float(start_s)
    selected = set(selected_ids)
    covered = 0.0
    for block in blocks:
        if block.block_id not in selected:
            continue
        covered += max(
            0.0,
            min(block.end_s, target_end) - max(block.start_s, 0.0),
        )
    return float(min(1.0, covered / target_end))


def _oracle_focus(record, blocks: Sequence[AcousticBlock]) -> list[str]:
    """Ground-truth upper-bound focus using the known event interval."""
    if not getattr(record, "source_mask_valid", False):
        raise ValueError("oracle focus requires source_mask_valid=true")
    start_s = getattr(record, "source_start_s", None)
    end_s = getattr(record, "source_end_s", None)
    if start_s is None or end_s is None:
        raise ValueError("oracle focus requires source_start_s/source_end_s")
    start_s = float(start_s)
    end_s = float(end_s)
    if record.role == "use":
        selected = [
            block.block_id
            for block in blocks
            if max(block.start_s, start_s) < min(block.end_s, end_s)
        ]
    elif record.role == "ignore":
        selected = [
            block.block_id
            for block in blocks
            if block.end_s <= start_s + 1e-9
        ]
    else:
        raise ValueError(f"unknown role: {record.role}")
    if not selected:
        raise ValueError(f"oracle focus selected no blocks for role={record.role}")
    return selected


def _oracle_blocks(record, duration_s: float) -> list[AcousticBlock]:
    """Exact temporal upper-bound address space for the oracle-KV control."""
    if not getattr(record, "source_mask_valid", False):
        raise ValueError("oracle blocks require source_mask_valid=true")
    start_s = getattr(record, "source_start_s", None)
    end_s = getattr(record, "source_end_s", None)
    if start_s is None or end_s is None:
        raise ValueError("oracle blocks require source_start_s/source_end_s")
    start = max(0.0, min(float(start_s), duration_s))
    end = max(start, min(float(end_s), duration_s))
    blocks: list[AcousticBlock] = []
    idx = 1
    if start > 1e-6:
        blocks.append(
            AcousticBlock(f"B{idx}", 0.0, start, "pre-event target region")
        )
        idx += 1
    if end > start + 1e-6:
        blocks.append(
            AcousticBlock(f"B{idx}", start, end, "ground-truth event region")
        )
        idx += 1
    if duration_s > end + 1e-6:
        blocks.append(
            AcousticBlock(f"B{idx}", end, duration_s, "post-event region")
        )
    if not blocks:
        raise ValueError("oracle address space is empty")
    return blocks


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
    focus_source: str = "model",
    apply_kv_mask: bool = True,
    min_target_coverage: float = 0.8,
) -> list[dict]:
    records = list(records)
    if len(records) != 2 or {r.role for r in records} != {"use", "ignore"}:
        raise ValueError("run_daa_pair expects exactly one use and one ignore record")
    if len({r.waveform_sha256 for r in records}) != 1:
        raise ValueError("DAA pair must use the exact same waveform hash")
    if not 0.0 <= min_target_coverage <= 1.0:
        raise ValueError("min_target_coverage must be in [0, 1]")

    audio_path = records[0].waveform_path
    raw_blocks: str | None = None
    if focus_source == "oracle":
        try:
            blocks = _oracle_blocks(records[0], duration_s)
        except Exception as exc:
            raise DAAStageError("oracle_blocks", str(exc)) from exc
        raw_blocks = "<oracle_blocks>ground-truth temporal address space</oracle_blocks>"
    elif focus_source == "model":
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
            raise ValueError(
                f"unsupported DAA block_strategy: {block_strategy!r}"
            )
    else:
        raise ValueError(f"unsupported DAA focus_source: {focus_source!r}")

    rows: list[dict] = []
    for record in records:
        if focus_source == "oracle":
            try:
                selected_ids = _oracle_focus(record, blocks)
            except Exception as exc:
                raise DAAStageError(
                    f"oracle_focus_{record.role}", str(exc)
                ) from exc
            raw_focus = f'<oracle_focus blocks="{",".join(selected_ids)}">'
        else:
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
            raise ValueError(
                f"{record.pair_id}/{record.role}: DAA MVP requires canonical options"
            )
        try:
            scores = wrapper.score_single_token_options_daa(
                audio_path,
                record.query,
                record.options,
                blocks,
                selected_ids,
                layers=layers,
                apply_kv_mask=apply_kv_mask,
            )
        except Exception as exc:
            raise DAAStageError(
                f"focused_reasoning_{record.role}", str(exc)
            ) from exc

        event_selected = _event_selected(record, blocks, selected_ids)
        target_coverage = (
            _target_coverage(record, blocks, selected_ids)
            if record.role == "ignore"
            else None
        )
        ignore_selection_valid = None
        if (
            record.role == "ignore"
            and event_selected is not None
            and target_coverage is not None
        ):
            ignore_selection_valid = bool(
                (not event_selected)
                and target_coverage >= min_target_coverage
            )

        rows.append(
            {
                "pair_id": record.pair_id,
                "role": record.role,
                "waveform_sha256": record.waveform_sha256,
                "answer": record.answer,
                "prediction": scores.predicted_option,
                "correct": scores.predicted_option == record.answer,
                "selected_blocks": list(selected_ids),
                "event_selected": event_selected,
                "target_coverage": target_coverage,
                "ignore_selection_valid": ignore_selection_valid,
                "block_strategy": block_strategy,
                "focus_source": focus_source,
                "apply_kv_mask": bool(apply_kv_mask),
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
        ignore_valid = pair["ignore"].get("ignore_selection_valid")
        if use_sel is None or ignore_valid is None:
            continue
        switchable.append(bool(use_sel) and bool(ignore_valid))

    summary["selection_switch_acc"] = (
        float(sum(switchable) / len(switchable)) if switchable else None
    )
    summary["selection_labeled_pairs"] = float(len(switchable))

    use_labeled = [
        row
        for row in rows
        if row.get("role") == "use" and row.get("event_selected") is not None
    ]
    ignore_labeled = [
        row
        for row in rows
        if row.get("role") == "ignore" and row.get("event_selected") is not None
    ]
    summary["use_event_selection_rate"] = (
        float(
            sum(bool(r["event_selected"]) for r in use_labeled)
            / len(use_labeled)
        )
        if use_labeled
        else None
    )
    summary["ignore_event_avoid_rate"] = (
        float(
            sum(not bool(r["event_selected"]) for r in ignore_labeled)
            / len(ignore_labeled)
        )
        if ignore_labeled
        else None
    )

    coverage_rows = [
        row for row in ignore_labeled if row.get("target_coverage") is not None
    ]
    summary["ignore_target_coverage_mean"] = (
        float(
            sum(float(r["target_coverage"]) for r in coverage_rows)
            / len(coverage_rows)
        )
        if coverage_rows
        else None
    )
    valid_ignore = [
        row
        for row in ignore_labeled
        if row.get("ignore_selection_valid") is not None
    ]
    summary["ignore_valid_selection_rate"] = (
        float(
            sum(bool(r["ignore_selection_valid"]) for r in valid_ignore)
            / len(valid_ignore)
        )
        if valid_ignore
        else None
    )

    use_selected = [r for r in use_labeled if bool(r["event_selected"])]
    ignore_avoided = [r for r in ignore_labeled if not bool(r["event_selected"])]
    ignore_valid_selected = [
        r for r in valid_ignore if bool(r["ignore_selection_valid"])
    ]
    summary["reasoning_acc_given_use_selection"] = (
        float(
            sum(bool(r["correct"]) for r in use_selected) / len(use_selected)
        )
        if use_selected
        else None
    )
    summary["reasoning_acc_given_ignore_avoidance"] = (
        float(
            sum(bool(r["correct"]) for r in ignore_avoided)
            / len(ignore_avoided)
        )
        if ignore_avoided
        else None
    )
    summary["reasoning_acc_given_valid_ignore_selection"] = (
        float(
            sum(bool(r["correct"]) for r in ignore_valid_selected)
            / len(ignore_valid_selected)
        )
        if ignore_valid_selected
        else None
    )
    return summary
