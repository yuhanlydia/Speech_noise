from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sar.data.relevance_pairs import load_jsonl_records
from sar.data.schema import RelevancePairRecord, validate_pair_records
from sar.metrics import compute_pair_metrics
from sar.models.base import AudioLMWrapper


def evaluate_records(wrapper: AudioLMWrapper, records: Sequence[RelevancePairRecord]):
    validate_pair_records(records)
    rows: list[dict] = []
    for record in records:
        if not record.options:
            raise ValueError(f'{record.pair_id}/{record.role}: canonical evaluator requires options')
        scores = wrapper.score_options(record.waveform_path, record.query, record.options)
        pred = scores.predicted_option
        rows.append({
            'pair_id': record.pair_id,
            'role': record.role,
            'waveform_sha256': record.waveform_sha256,
            'answer': record.answer,
            'prediction': pred,
            'correct': pred == record.answer,
            'logprobs': scores.logprobs,
        })
    return rows, compute_pair_metrics(rows)


def write_results(rows: Sequence[dict], summary: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / 'results.jsonl'
    with rows_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return rows_path, summary_path


def main() -> None:
    p = argparse.ArgumentParser(description='Evaluate same-waveform relevance-switch records.')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    raise SystemExit('Instantiate a configured AudioLMWrapper in your experiment launcher; see README GPU smoke.')


if __name__ == '__main__':
    main()
