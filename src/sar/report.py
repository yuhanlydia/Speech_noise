from __future__ import annotations

import json
from pathlib import Path


STAGES = {
    "capability": "mvp_capability_gate",
    "hook_identity": "daa_identity",
    "base": "mvp_diagnostic",
    "oracle_prompt_only": "mvp_daa_oracle_prompt_only",
    "oracle_kv": "mvp_daa_oracle",
    "prompt_only": "mvp_daa_prompt_only",
    "fixed_daa": "mvp_daa_fixed",
    "declared_daa": "mvp_daa_declared",
}


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def export_v2_report(
    *, results_root: str | Path = "results", output_dir: str | Path = "reports/run_v2"
) -> dict:
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        key: _read_json(results_root / directory / "summary.json")
        for key, directory in STAGES.items()
    }
    report = {"protocol": "validity-gated-daa-v2", "stages": stages}
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = ["# Validity-Gated DAA V2 Run", ""]
    for key in STAGES:
        md.append(f"## {key}")
        payload = stages[key]
        md.append(
            "not run"
            if payload is None
            else "```json\n" + json.dumps(payload, indent=2) + "\n```"
        )
        md.append("")
    (output_dir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    return report
