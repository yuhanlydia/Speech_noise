"""Replay recorded declarations while recomputing the KV-masked answer pass.

Keep the source config, manifest and results immutable together. This tool
compares the same model and prompts; it does not transfer selections across
models or invent replacements for failed declarations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from sar.config import load_experiment_config
from sar.daa_pipeline import summarize_daa_rows
from sar.daa_smoke import _build_wrapper, _duration_s, record_manifest_sha256
from sar.data.blocks import AcousticBlock, fixed_temporal_blocks, validate_blocks
from sar.data.relevance_pairs import load_jsonl_records
from sar.data.schema import validate_pair_records
from sar.eval import write_results
from sar.methods.daa import parse_focus_declaration


def validate_replay_inputs(cfg, source_cfg, records, source_rows, *, duration_resolver=_duration_s):
    if cfg.method.daa is None or source_cfg.method.daa is None:
        raise ValueError("replay requires DAA configs")
    if cfg.method.name != "daa" or source_cfg.method.name != "daa":
        raise ValueError("replay requires method.name=daa")
    if any(d.focus_source != "model" or d.block_strategy != "fixed"
           for d in (cfg.method.daa, source_cfg.method.daa)):
        raise ValueError("replay supports only model-selected fixed blocks")
    if source_cfg.method.daa.apply_kv_mask or not cfg.method.daa.apply_kv_mask:
        raise ValueError("replay requires prompt-only source and KV-mask target")
    if cfg.model != source_cfg.model or cfg.data != source_cfg.data or cfg.seed != source_cfg.seed:
        raise ValueError("replay requires the same model, manifest and seed")
    source_method = source_cfg.method.model_dump()
    source_method["daa"]["apply_kv_mask"] = True
    if source_method != cfg.method.model_dump():
        raise ValueError("replay method configs may differ only in apply_kv_mask")
    validate_pair_records(records)
    by_key = {(r["pair_id"], r["role"]): r for r in source_rows}
    expected = {(r.pair_id, r.role) for r in records}
    if len(by_key) != len(source_rows) or set(by_key) != expected:
        raise ValueError("cached rows must match the full manifest exactly, without duplicates")
    failures = {}
    for record in records:
        row = by_key[(record.pair_id, record.role)]
        if row["waveform_sha256"] != record.waveform_sha256 or row["answer"] != record.answer:
            raise ValueError("cached waveform hash or answer differs from manifest")
        stage = row.get("daa_error_stage")
        if stage:
            if stage not in {"block_declaration", "focus_ignore", "focus_use"}:
                raise ValueError("cannot replay a source reasoning/runtime failure as a declaration")
            other = by_key[(record.pair_id, "use" if record.role == "ignore" else "ignore")]
            if other.get("daa_error_stage") != stage:
                raise ValueError("cached declaration failures must cover the whole pair consistently")
            failures[record.pair_id] = stage
            continue
        if (row["focus_source"] != source_cfg.method.daa.focus_source
                or row["block_strategy"] != source_cfg.method.daa.block_strategy
                or row["apply_kv_mask"] is not False):
            raise ValueError("cached row condition differs from source config")
        blocks = [AcousticBlock(**b) for b in row["blocks"]]
        duration = duration_resolver(record.waveform_path)
        validate_blocks(blocks, duration_s=duration, max_blocks=cfg.method.daa.max_blocks)
        if cfg.method.daa.block_strategy == "fixed" and cfg.method.daa.focus_source == "model":
            expected_blocks = fixed_temporal_blocks(
                duration, block_seconds=cfg.method.daa.block_seconds,
                max_blocks=cfg.method.daa.max_blocks,
            )
            if blocks != expected_blocks:
                raise ValueError("cached fixed block table differs from the configured waveform partition")
        if cfg.method.daa.focus_source == "model":
            parsed = parse_focus_declaration(
                row["raw_focus_declaration"], valid_block_ids={b.block_id for b in blocks},
                max_selected=cfg.method.daa.max_focus_blocks,
            )
            if parsed != row["selected_blocks"]:
                raise ValueError("cached selection differs from its raw declaration")
    return by_key, failures


def evaluate_replay_with_wrapper(cfg, source_cfg, wrapper, records, source_rows,
                                 *, duration_resolver=_duration_s, progress=None):
    cached, failures = validate_replay_inputs(
        cfg, source_cfg, records, source_rows, duration_resolver=duration_resolver,
    )
    rows = []
    for index, record in enumerate(records, 1):
        row = deepcopy(cached[(record.pair_id, record.role)])
        row["selection_replayed"] = True
        row["apply_kv_mask"] = True
        if record.pair_id in failures:
            # Preserve end-to-end declaration failures; never replace them with
            # oracle/base selections to make the KV condition look better.
            row.update(prediction=None, correct=False)
        else:
            scores = wrapper.score_single_token_options_daa(
                record.waveform_path, record.query, record.options,
                [AcousticBlock(**b) for b in row["blocks"]], row["selected_blocks"],
                layers=cfg.method.layers, apply_kv_mask=True,
            )
            if any(not math.isfinite(value) for value in scores.logprobs):
                raise ValueError("non-finite replay answer scores")
            row.update(prediction=scores.predicted_option,
                       correct=scores.predicted_option == record.answer,
                       logprobs=scores.logprobs)
        rows.append(row)
        if progress is not None:
            progress(index, len(records))
    summary = summarize_daa_rows(rows)
    pair_count = len(records) // 2
    summary.update(
        protocol_failure_pairs=float(len(failures)),
        protocol_completion_rate=(pair_count - len(failures)) / max(pair_count, 1),
        protocol_failure_stages=dict(Counter(failures.values())),
        focus_source=cfg.method.daa.focus_source,
        apply_kv_mask=True,
        block_strategy=cfg.method.daa.block_strategy,
        selection_replayed=True,
    )
    return rows, summary


def load_replay_source(cfg, source_cfg, records, source_results):
    source_results = Path(source_results)
    source_root = Path(source_cfg.output_dir).resolve()
    target_root = Path(cfg.output_dir).resolve()
    if source_results.resolve() != (source_root / "results.jsonl").resolve():
        raise ValueError("source results must belong to the source config output directory")
    if source_root == target_root:
        raise ValueError("replay output must not overwrite the source directory")
    protected = {(source_root / name).resolve()
                 for name in ("results.jsonl", "summary.json", "run_metadata.json")}
    if any((target_root / name).resolve() in protected
           for name in ("results.jsonl", "summary.json", "run_metadata.json")):
        raise ValueError("replay output aliases a source artifact")
    metadata_path = source_root / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("source results require producer run metadata")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("config") != source_cfg.model_dump(mode="json"):
        raise ValueError("source config differs from producer metadata")
    if metadata.get("records_sha256") != record_manifest_sha256(records):
        raise ValueError("manifest differs from producer metadata")
    payload = source_results.read_bytes()
    if metadata.get("results_sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError("source results differ from producer metadata")
    return [json.loads(line) for line in payload.decode().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    source_cfg = load_experiment_config(args.source_config)
    records = load_jsonl_records(cfg.data.manifest)
    source_rows = load_replay_source(cfg, source_cfg, records, args.source_results)
    validate_replay_inputs(cfg, source_cfg, records, source_rows)
    wrapper = _build_wrapper(cfg)
    rows, summary = evaluate_replay_with_wrapper(
        cfg, source_cfg, wrapper, records, source_rows,
        progress=lambda i, n: print(f"replayed answers {i}/{n}", flush=True) if i % 20 == 0 or i == n else None,
    )
    summary["selection_source_results"] = str(args.source_results)
    summary["selection_source_sha256"] = hashlib.sha256(args.source_results.read_bytes()).hexdigest()
    summary["manifest_sha256"] = hashlib.sha256(Path(cfg.data.manifest).read_bytes()).hexdigest()
    write_results(rows, summary, cfg.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
