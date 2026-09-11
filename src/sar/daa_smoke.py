from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

from sar.config import ExperimentConfig, load_experiment_config
from sar.daa_pipeline import DAAStageError, run_daa_pair, summarize_daa_rows
from sar.data.relevance_pairs import load_jsonl_records, pair_by_id
from sar.data.schema import RelevancePairRecord, validate_pair_records
from sar.eval import write_results


def dry_run_daa(config_path: str | Path) -> dict:
    cfg = load_experiment_config(config_path)
    if cfg.method.name != "daa" or cfg.method.daa is None:
        raise ValueError("DAA launcher requires method.name=daa and method.daa config")
    records = load_jsonl_records(cfg.data.manifest)
    validate_pair_records(records)
    return {
        "model": cfg.model.model_id,
        "method": cfg.method.name,
        "block_strategy": cfg.method.daa.block_strategy,
        "focus_source": cfg.method.daa.focus_source,
        "apply_kv_mask": cfg.method.daa.apply_kv_mask,
        "records": len(records),
        "pairs": len({record.pair_id for record in records}),
        "layers": cfg.method.layers,
        "manifest": cfg.data.manifest,
        "output_dir": cfg.output_dir,
    }


def _duration_s(path: str) -> float:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("soundfile is required for DAA evaluation") from exc
    return float(sf.info(path).duration)


def _build_wrapper(cfg: ExperimentConfig):
    if cfg.model.attention_backend != "eager":
        raise ValueError("DAA requires model.attention_backend=eager")
    from sar.models.qwen_omni import QwenOmniConfig
    from sar.models.qwen_omni_daa import QwenOmniDAAWrapper

    wrapper = QwenOmniDAAWrapper(
        QwenOmniConfig(
            model_id=cfg.model.model_id,
            quantization=cfg.model.quantization,
            thinker_only=cfg.model.thinker_only,
            device_map=cfg.model.device_map,
            attention_backend=cfg.model.attention_backend,
            routing_layer=cfg.method.layers[0] if cfg.method.layers else 0,
        )
    )
    wrapper.load()
    if cfg.method.daa is not None:
        wrapper.stop_on_complete_declaration = cfg.method.daa.stop_on_complete_declaration
    return wrapper


def _failure_rows(
    records: list[RelevancePairRecord],
    error: DAAStageError,
) -> list[dict]:
    return [
        {
            "pair_id": record.pair_id,
            "role": record.role,
            "waveform_sha256": record.waveform_sha256,
            "answer": record.answer,
            "prediction": None,
            "correct": False,
            "selected_blocks": [],
            "event_selected": None,
            "target_coverage": None,
            "ignore_selection_valid": None,
            "daa_error_stage": error.stage,
            "daa_error": str(error),
        }
        for record in records
    ]


def evaluate_daa_with_wrapper(
    cfg: ExperimentConfig,
    wrapper,
    records: list[RelevancePairRecord],
    *,
    duration_resolver: Callable[[str], float] = _duration_s,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], dict]:
    if cfg.method.name != "daa" or cfg.method.daa is None:
        raise ValueError("evaluate_daa_with_wrapper requires method.name=daa")

    pairs = pair_by_id(records)
    rows: list[dict] = []
    failed_pairs = 0
    stage_counts: dict[str, int] = {}
    daa = cfg.method.daa

    for index, pair in enumerate(pairs.values(), 1):
        pair_records = [pair["ignore"], pair["use"]]
        try:
            pair_rows = run_daa_pair(
                wrapper,
                pair_records,
                duration_s=duration_resolver(pair_records[0].waveform_path),
                block_strategy=daa.block_strategy,
                block_seconds=daa.block_seconds,
                max_blocks=daa.max_blocks,
                max_focus_blocks=daa.max_focus_blocks,
                scan_max_new_tokens=daa.scan_max_new_tokens,
                select_max_new_tokens=daa.select_max_new_tokens,
                layers=cfg.method.layers,
                focus_source=daa.focus_source,
                apply_kv_mask=daa.apply_kv_mask,
                min_target_coverage=daa.min_target_coverage,
            )
        except DAAStageError as exc:
            failed_pairs += 1
            stage_counts[exc.stage] = stage_counts.get(exc.stage, 0) + 1
            pair_rows = _failure_rows(pair_records, exc)
        rows.extend(pair_rows)
        if progress is not None:
            progress(index, len(pairs))

    summary = summarize_daa_rows(rows)
    summary["protocol_failure_pairs"] = float(failed_pairs)
    summary["protocol_completion_rate"] = float(
        (len(pairs) - failed_pairs) / max(len(pairs), 1)
    )
    summary["protocol_failure_stages"] = stage_counts
    summary["focus_source"] = daa.focus_source
    summary["apply_kv_mask"] = daa.apply_kv_mask
    summary["block_strategy"] = daa.block_strategy
    return rows, summary


def record_manifest_sha256(records) -> str:
    payload = json.dumps(
        [record.model_dump(mode="json") for record in records],
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def evaluate_daa(cfg: ExperimentConfig) -> tuple[list[dict], dict]:
    records = load_jsonl_records(cfg.data.manifest)
    validate_pair_records(records)
    metadata = {"config": cfg.model_dump(mode="json"),
                "records_sha256": record_manifest_sha256(records)}
    wrapper = _build_wrapper(cfg)
    rows, summary = evaluate_daa_with_wrapper(
        cfg, wrapper, records,
        progress=lambda i, n: print(f"DAA pairs {i}/{n}", flush=True)
        if i % 10 == 0 or i == n else None,
    )
    write_results(rows, summary, cfg.output_dir)
    metadata["results_sha256"] = hashlib.sha256(
        (Path(cfg.output_dir) / "results.jsonl").read_bytes()
    ).hexdigest()
    (Path(cfg.output_dir) / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Declarative Acoustic Attention evaluator"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run_daa(args.config), indent=2))
        return
    cfg = load_experiment_config(args.config)
    _, summary = evaluate_daa(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
