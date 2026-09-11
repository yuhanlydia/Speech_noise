from __future__ import annotations

import argparse
import json
from pathlib import Path

from sar.data.audit import audit_public_mvp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit source/mixed WAV integrity before Speech_noise experiments"
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/mvp/source.jsonl"),
    )
    parser.add_argument(
        "--pairs-manifest",
        type=Path,
        default=Path("data/mvp/pairs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/data_audit/summary.json"),
    )
    args = parser.parse_args()
    report = audit_public_mvp(args.source_manifest, args.pairs_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
