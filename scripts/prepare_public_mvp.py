from __future__ import annotations

import argparse
from pathlib import Path

from scripts.build_mvp_manifest import build_manifest
from sar.data.public_mvp import prepare_source_manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Prepare a fully public same-waveform relevance-switch MVP from "
            "MMLU + ESC-50 or LibriSpeech, then build data/mvp/pairs.jsonl."
        )
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/mvp"))
    p.add_argument("--num-pairs", type=int, default=128)
    p.add_argument("--protocol", choices=["dev", "confirm"], default="dev")
    p.add_argument("--event-source", choices=["esc50", "librispeech"], default="esc50")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--gap-s", type=float, default=0.35)
    p.add_argument("--ratio-db", type=float, default=0.0)
    p.add_argument("--max-spoken-words", type=int, default=60)
    p.add_argument("--tts-voice", default="en-us")
    p.add_argument("--tts-speed", type=int, default=190)
    args = p.parse_args()

    source = prepare_source_manifest(
        args.output_dir,
        num_pairs=args.num_pairs,
        protocol=args.protocol,
        event_source=args.event_source,
        seed=args.seed,
        sample_rate=args.sample_rate,
        gap_s=args.gap_s,
        ratio_db=args.ratio_db,
        max_spoken_words=args.max_spoken_words,
        tts_voice=args.tts_voice,
        tts_speed=args.tts_speed,
    )
    pairs = build_manifest(source, args.output_dir)
    print(f"source_manifest={source}")
    print(f"pairs_manifest={pairs}")
    print(f"pairs={args.num_pairs}")
    print(f"protocol={args.protocol}")
    print(f"event_source={args.event_source}")


if __name__ == "__main__":
    main()
