from __future__ import annotations

import argparse
import json
from pathlib import Path

from sar.config import ExperimentConfig, load_experiment_config
from sar.data.relevance_pairs import load_jsonl_records
from sar.data.schema import validate_pair_records
from sar.models.qwen_omni import QwenOmniConfig, QwenOmniWrapper
from sar.validity import (
    evaluate_capability_pairs,
    filter_eligible_records,
    write_capability_outputs,
)


def _build_wrapper(cfg: ExperimentConfig) -> QwenOmniWrapper:
    wrapper = QwenOmniWrapper(
        QwenOmniConfig(
            model_id=cfg.model.model_id,
            quantization=cfg.model.quantization,
            thinker_only=cfg.model.thinker_only,
            device_map=cfg.model.device_map,
            attention_backend=cfg.model.attention_backend,
            torch_dtype=cfg.model.torch_dtype,
            routing_layer=cfg.method.layers[0] if cfg.method.layers else 0,
        )
    )
    wrapper.load()
    return wrapper


def run_capability_gate(
    cfg: ExperimentConfig,
    *,
    source_manifest: str | Path,
    eligible_manifest: str | Path,
) -> dict:
    records = load_jsonl_records(cfg.data.manifest)
    validate_pair_records(records)
    wrapper = _build_wrapper(cfg)
    rows = evaluate_capability_pairs(wrapper, records, source_manifest)
    eligible = filter_eligible_records(records, rows)
    _, eligible_path, summary = write_capability_outputs(
        rows,
        eligible,
        output_dir=cfg.output_dir,
        eligible_manifest=eligible_manifest,
    )
    summary = dict(summary)
    summary["eligible_manifest"] = str(eligible_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated capability gate for Selective Acoustic Relevance"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/mvp_capability_gate.yaml"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/mvp/source.jsonl"),
    )
    parser.add_argument(
        "--eligible-manifest",
        type=Path,
        default=Path("data/mvp/pairs_eligible.jsonl"),
    )
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    summary = run_capability_gate(
        cfg,
        source_manifest=args.source_manifest,
        eligible_manifest=args.eligible_manifest,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
