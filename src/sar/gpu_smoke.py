from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from sar.config import ExperimentConfig, load_experiment_config
from sar.data.relevance_pairs import load_jsonl_records
from sar.data.schema import RelevancePairRecord, validate_pair_records
from sar.eval import write_results
from sar.methods.qacr import QACRRouter
from sar.metrics import compute_pair_metrics
from sar.models.base import OptionScores
from sar.models.qwen_omni import QwenOmniConfig, QwenOmniWrapper
from sar.train import freeze_backbone, qacr_record_loss


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dry_run(config_path: str | Path) -> dict:
    cfg = load_experiment_config(config_path)
    records = load_jsonl_records(cfg.data.manifest)
    validate_pair_records(records)
    pair_count = len({r.pair_id for r in records})
    return {
        "model": cfg.model.model_id,
        "method": cfg.method.name,
        "records": len(records),
        "pairs": pair_count,
        "manifest": str(cfg.data.manifest),
        "output_dir": cfg.output_dir,
    }


def _build_wrapper(cfg: ExperimentConfig) -> QwenOmniWrapper:
    if cfg.model.attention_backend != "eager" and cfg.method.name == "qacr":
        raise ValueError("QACR requires model.attention_backend=eager")
    wrapper = QwenOmniWrapper(
        QwenOmniConfig(
            model_id=cfg.model.model_id,
            quantization=cfg.model.quantization,
            thinker_only=cfg.model.thinker_only,
            device_map=cfg.model.device_map,
            attention_backend=cfg.model.attention_backend,
            torch_dtype=cfg.model.torch_dtype,
            routing_layer=cfg.method.layers[0],
        )
    )
    wrapper.load()
    return wrapper


def _build_router(wrapper: QwenOmniWrapper, cfg: ExperimentConfig) -> QACRRouter:
    layer = cfg.method.layers[0]
    attn = wrapper.get_text_attention_layer(layer)
    hidden_size = int(attn.q_proj.in_features)
    router = QACRRouter(hidden_size, hidden_size, cfg.method.router_dim)
    router.to(device=attn.q_proj.weight.device, dtype=torch.float32)
    return router


def _checkpoint_path(cfg: ExperimentConfig) -> Path:
    if not cfg.method.checkpoint:
        raise ValueError("QACR requires method.checkpoint in the experiment config")
    return Path(cfg.method.checkpoint)


def save_router(router: QACRRouter, cfg: ExperimentConfig) -> Path:
    path = _checkpoint_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": router.state_dict(),
        "router_dim": cfg.method.router_dim,
        "layers": cfg.method.layers,
        "model_id": cfg.model.model_id,
    }
    torch.save(payload, path)
    return path


def load_router(router: QACRRouter, cfg: ExperimentConfig) -> None:
    path = _checkpoint_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"QACR checkpoint not found: {path}. Run scripts/train_qacr.sh first.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    router.load_state_dict(payload["state_dict"], strict=True)


def _gold_index(record: RelevancePairRecord) -> int:
    if not record.options:
        raise ValueError(f"{record.pair_id}/{record.role}: options are required")
    try:
        return record.options.index(record.answer)
    except ValueError as exc:
        raise ValueError(f"{record.pair_id}/{record.role}: answer {record.answer!r} is not in options") from exc


def _event_mask(wrapper: QwenOmniWrapper, record: RelevancePairRecord, audio_mask: torch.Tensor) -> torch.Tensor | None:
    if not record.source_mask_valid:
        return None
    if record.source_start_s is None or record.source_end_s is None:
        raise ValueError(f"{record.pair_id}: source_mask_valid requires source_start_s/source_end_s")
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("soundfile is required to map temporal event spans") from exc
    duration_s = float(sf.info(record.waveform_path).duration)
    return wrapper.event_token_mask(
        audio_mask,
        duration_s=duration_s,
        start_s=float(record.source_start_s),
        end_s=min(float(record.source_end_s), duration_s),
    )


def _option_scores_from_tensor(options: list[str], scores: torch.Tensor) -> OptionScores:
    return OptionScores(options=list(options), logprobs=[float(v) for v in scores.detach().cpu()])


def train_qacr(cfg: ExperimentConfig) -> dict:
    if cfg.method.name != "qacr":
        raise ValueError("--train-router currently supports method.name=qacr only")
    _set_seed(cfg.seed)
    records = load_jsonl_records(cfg.data.manifest)
    validate_pair_records(records)
    wrapper = _build_wrapper(cfg)
    freeze_backbone(wrapper.model)
    wrapper.model.eval()
    router = _build_router(wrapper, cfg)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    history: list[dict] = []
    generator = random.Random(cfg.seed)
    for epoch in range(cfg.training.epochs):
        order = list(records)
        generator.shuffle(order)
        sums = {"loss": 0.0, "qa": 0.0, "switch": 0.0, "identity": 0.0}
        router.train()
        for record in order:
            if not record.options:
                raise ValueError(f"{record.pair_id}/{record.role}: QACR training requires canonical options")
            optimizer.zero_grad(set_to_none=True)
            scores, gate, audio_mask = wrapper.score_single_token_options_qacr_tensors(
                record.waveform_path,
                record.query,
                record.options,
                router,
                layer=cfg.method.layers[0],
            )
            event_mask = _event_mask(wrapper, record, audio_mask)
            loss, parts = qacr_record_loss(
                scores=scores,
                gold_index=_gold_index(record),
                gate=gate,
                audio_mask=audio_mask,
                event_mask=event_mask,
                role=record.role,
                lambda_switch=cfg.method.lambda_switch,
                lambda_identity=cfg.method.lambda_identity,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()
            sums["loss"] += float(loss.detach().cpu())
            for key in ("qa", "switch", "identity"):
                sums[key] += parts[key]
        n = max(len(order), 1)
        epoch_row = {"epoch": epoch + 1, **{k: v / n for k, v in sums.items()}}
        history.append(epoch_row)
        print(json.dumps(epoch_row))

    checkpoint = save_router(router, cfg)
    report = {
        "checkpoint": str(checkpoint),
        "trainable_parameters": router.trainable_parameter_count(),
        "records": len(records),
        "history": history,
    }
    report_path = Path(cfg.output_dir) / "training_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_base(cfg: ExperimentConfig, wrapper: QwenOmniWrapper, records: list[RelevancePairRecord]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for record in records:
        if not record.options:
            raise ValueError(f"{record.pair_id}/{record.role}: options are required")
        try:
            scores = wrapper.score_single_token_options(record.waveform_path, record.query, record.options)
        except ValueError:
            scores = wrapper.score_options(record.waveform_path, record.query, record.options)
        rows.append({
            "pair_id": record.pair_id,
            "role": record.role,
            "waveform_sha256": record.waveform_sha256,
            "answer": record.answer,
            "prediction": scores.predicted_option,
            "correct": scores.predicted_option == record.answer,
            "logprobs": scores.logprobs,
        })
    return rows, compute_pair_metrics(rows)


def evaluate_qacr(cfg: ExperimentConfig, wrapper: QwenOmniWrapper, records: list[RelevancePairRecord]) -> tuple[list[dict], dict]:
    router = _build_router(wrapper, cfg)
    load_router(router, cfg)
    router.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for record in records:
            if not record.options:
                raise ValueError(f"{record.pair_id}/{record.role}: options are required")
            scores_t, gate, audio_mask = wrapper.score_single_token_options_qacr_tensors(
                record.waveform_path, record.query, record.options, router, layer=cfg.method.layers[0]
            )
            scores = _option_scores_from_tensor(record.options, scores_t)
            event_mask = _event_mask(wrapper, record, audio_mask)
            audio_values = gate[audio_mask]
            row = {
                "pair_id": record.pair_id,
                "role": record.role,
                "waveform_sha256": record.waveform_sha256,
                "answer": record.answer,
                "prediction": scores.predicted_option,
                "correct": scores.predicted_option == record.answer,
                "logprobs": scores.logprobs,
                "audio_gate_mean": float(audio_values.mean().cpu()) if audio_values.numel() else None,
                "event_gate_mean": None,
            }
            if event_mask is not None and event_mask.any():
                row["event_gate_mean"] = float(gate[event_mask].mean().cpu())
            rows.append(row)
    summary = compute_pair_metrics(rows)
    event_by_role = {
        role: [r["event_gate_mean"] for r in rows if r["role"] == role and r["event_gate_mean"] is not None]
        for role in ("use", "ignore")
    }
    for role, vals in event_by_role.items():
        summary[f"event_gate_{role}_mean"] = float(np.mean(vals)) if vals else None
    if event_by_role["use"] and event_by_role["ignore"]:
        summary["event_gate_switch_gap"] = summary["event_gate_use_mean"] - summary["event_gate_ignore_mean"]
    else:
        summary["event_gate_switch_gap"] = None
    summary["trainable_parameters"] = router.trainable_parameter_count()
    return rows, summary


def evaluate_experiment(cfg: ExperimentConfig) -> tuple[list[dict], dict]:
    _set_seed(cfg.seed)
    records = load_jsonl_records(cfg.data.manifest)
    validate_pair_records(records)
    wrapper = _build_wrapper(cfg)
    if cfg.method.name == "base":
        rows, summary = evaluate_base(cfg, wrapper, records)
    elif cfg.method.name == "qacr":
        rows, summary = evaluate_qacr(cfg, wrapper, records)
    else:
        raise NotImplementedError(
            f"GPU launcher for {cfg.method.name!r} is not wired yet; the baseline module is available for controlled tensor experiments."
        )
    write_results(rows, summary, cfg.output_dir)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech_noise QACR experiment launcher")
    parser.add_argument("--config", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--train-router", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    if args.dry_run:
        print(json.dumps(dry_run(args.config), indent=2))
        return
    if args.train_router:
        print(json.dumps(train_qacr(cfg), indent=2))
        return
    _, summary = evaluate_experiment(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
