from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from sar.config import load_experiment_config
from sar.data.blocks import AcousticBlock
from sar.data.relevance_pairs import load_jsonl_records
from sar.hook_sanity import compare_score_vectors
from sar.models.qwen_omni import QwenOmniConfig
from sar.models.qwen_omni_daa import QwenOmniDAAWrapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that DAA with all audio allowed matches the identical prompt without the hook"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/mvp_daa_fixed.yaml"),
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    records = load_jsonl_records(cfg.data.manifest)
    if not records:
        raise SystemExit("eligible manifest is empty; run the capability gate first")
    record = records[0]
    if not record.options:
        raise SystemExit("identity check requires canonical options")

    wrapper = QwenOmniDAAWrapper(
        QwenOmniConfig(
            model_id=cfg.model.model_id,
            quantization=cfg.model.quantization,
            thinker_only=cfg.model.thinker_only,
            device_map=cfg.model.device_map,
            attention_backend=cfg.model.attention_backend,
        )
    )
    wrapper.load()
    duration = float(sf.info(record.waveform_path).duration)
    blocks = [AcousticBlock("B1", 0.0, duration, "full audio")]
    selected = ["B1"]

    prompt_only = wrapper.score_single_token_options_daa(
        record.waveform_path,
        record.query,
        record.options,
        blocks,
        selected,
        layers=[],
        apply_kv_mask=False,
    )
    with_hook = wrapper.score_single_token_options_daa(
        record.waveform_path,
        record.query,
        record.options,
        blocks,
        selected,
        layers=[],
        apply_kv_mask=True,
    )
    report = compare_score_vectors(
        prompt_only,
        with_hook,
        tolerance=args.tolerance,
    )
    report.update(
        {
            "pair_id": record.pair_id,
            "role": record.role,
            "reference_prediction": prompt_only.predicted_option,
            "hook_prediction": with_hook.predicted_option,
        }
    )
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
