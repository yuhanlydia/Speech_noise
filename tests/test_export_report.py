import json

from sar.report import export_v2_report


def test_export_v2_report_collects_available_stage_summaries(tmp_path):
    results = tmp_path / "results"
    for name, payload in {
        "mvp_capability_gate": {"eligible_pairs": 64},
        "mvp_diagnostic": {"pair_switch_acc": 0.4},
        "mvp_daa_oracle": {"pair_switch_acc": 0.7},
    }.items():
        d = results / name
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "reports" / "run_v2"
    report = export_v2_report(results_root=results, output_dir=out)
    assert report["stages"]["capability"]["eligible_pairs"] == 64
    assert report["stages"]["base"]["pair_switch_acc"] == 0.4
    assert report["stages"]["oracle_kv"]["pair_switch_acc"] == 0.7
    assert report["stages"]["prompt_only"] is None
    assert (out / "summary.json").exists()
