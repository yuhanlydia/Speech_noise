from __future__ import annotations

import argparse
import json
from pathlib import Path

from sar.report import export_v2_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V2 summaries to a git-trackable report")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/run_v2"))
    args = parser.parse_args()
    report = export_v2_report(results_root=args.results_root, output_dir=args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
