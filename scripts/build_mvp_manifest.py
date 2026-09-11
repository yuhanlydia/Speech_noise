from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from sar.data.mixing import mix_sources, sha256_waveform
from sar.data.relevance_pairs import write_jsonl_records
from sar.data.schema import RelevancePairRecord, validate_pair_records


def build_manifest(source_manifest: str | Path, output_dir: str | Path) -> Path:
    source_manifest = Path(source_manifest)
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[RelevancePairRecord] = []

    with source_manifest.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = str(row["pair_id"])
            target, sr_t = sf.read(row["target_wav"], dtype="float32")
            event, sr_e = sf.read(row["event_wav"], dtype="float32")
            if target.ndim != 1 or event.ndim != 1:
                raise ValueError(f"{pair_id}: MVP builder expects mono audio")
            if sr_t != sr_e:
                raise ValueError(f"{pair_id}: target/event sample rates differ")
            mixed = mix_sources(
                target,
                event,
                sample_rate=sr_t,
                snr_db=row.get("snr_db"),
                tir_db=row.get("tir_db"),
                offset_samples=int(row.get("offset_samples", 0)),
            )
            wav_path = audio_dir / f"{pair_id}.wav"
            sf.write(wav_path, mixed, sr_t, subtype="FLOAT")
            digest = sha256_waveform(mixed, sr_t)
            common = dict(
                pair_id=pair_id,
                waveform_path=str(wav_path.resolve()),
                waveform_sha256=digest,
                event_type=str(row["event_type"]),
                source_id=row.get("source_id"),
                seed=int(row.get("seed", 0)),
                snr_db=row.get("snr_db"),
                tir_db=row.get("tir_db"),
                sample_rate=int(sr_t),
                offset_samples=int(row.get("offset_samples", 0)),
                source_start_s=float(int(row.get("offset_samples", 0)) / sr_t),
                source_end_s=float((int(row.get("offset_samples", 0)) + len(event)) / sr_t),
                source_mask_valid=bool(row.get("source_mask_valid", False)),
            )
            records.extend([
                RelevancePairRecord(role="ignore", query=row["q_ignore"], answer=row["a_ignore"], options=row.get("options_ignore"), **common),
                RelevancePairRecord(role="use", query=row["q_use"], answer=row["a_use"], options=row.get("options_use"), **common),
            ])
    validate_pair_records(records)
    return write_jsonl_records(output_dir / "pairs.jsonl", records)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source_manifest", type=Path)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    print(build_manifest(args.source_manifest, args.output_dir))


if __name__ == "__main__":
    main()
