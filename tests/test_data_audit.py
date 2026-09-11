import json

import numpy as np
import soundfile as sf

from scripts.build_mvp_manifest import build_manifest
from sar.data.audit import audit_public_mvp


def _write_wav(path, x, sr=16000):
    sf.write(path, np.asarray(x, dtype=np.float32), sr, subtype="FLOAT")


def _source_row(target, event):
    return {
        "pair_id": "p1",
        "target_wav": str(target),
        "event_wav": str(event),
        "event_type": "dog",
        "source_id": "dog-1",
        "source_mask_valid": True,
        "offset_samples": 16000,
        "snr_db": 0.0,
        "q_ignore": "Which option answers the spoken question?",
        "a_ignore": "A",
        "options_ignore": ["A", "B", "C", "D"],
        "q_use": "Which sound is audible?",
        "a_use": "B",
        "options_use": ["A", "B", "C", "D"],
        "seed": 0,
    }


def test_audit_accepts_consistent_source_pair_and_waveform(tmp_path):
    target = tmp_path / "target.wav"
    event = tmp_path / "event.wav"
    _write_wav(target, np.ones(16000) * 0.05)
    _write_wav(event, np.ones(8000) * 0.02)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_source_row(target, event)) + "\n", encoding="utf-8")
    pairs = build_manifest(source, tmp_path / "mvp")

    report = audit_public_mvp(source, pairs)
    assert report["ok"] is True
    assert report["num_pairs"] == 1
    assert report["num_errors"] == 0
    pair = report["pairs"][0]
    assert pair["mixed_duration_s"] == 1.5
    assert pair["event_duration_s"] == 0.5
    assert pair["source_start_s"] == 1.0


def test_audit_detects_tampered_mixed_waveform_hash(tmp_path):
    target = tmp_path / "target.wav"
    event = tmp_path / "event.wav"
    _write_wav(target, np.ones(16000) * 0.05)
    _write_wav(event, np.ones(8000) * 0.02)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_source_row(target, event)) + "\n", encoding="utf-8")
    pairs = build_manifest(source, tmp_path / "mvp")
    first = json.loads(pairs.read_text(encoding="utf-8").splitlines()[0])
    mixed = first["waveform_path"]
    _write_wav(mixed, np.zeros(24000))

    report = audit_public_mvp(source, pairs)
    assert report["ok"] is False
    assert any("waveform_sha256" in err for err in report["errors"])


def test_audit_detects_source_span_metadata_mismatch(tmp_path):
    target = tmp_path / "target.wav"
    event = tmp_path / "event.wav"
    _write_wav(target, np.ones(16000) * 0.05)
    _write_wav(event, np.ones(8000) * 0.02)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_source_row(target, event)) + "\n", encoding="utf-8")
    pairs = build_manifest(source, tmp_path / "mvp")
    rows = [json.loads(line) for line in pairs.read_text(encoding="utf-8").splitlines()]
    rows[0]["source_start_s"] = 0.25
    rows[1]["source_start_s"] = 0.25
    pairs.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    report = audit_public_mvp(source, pairs)
    assert report["ok"] is False
    assert any("offset/source_start_s" in err for err in report["errors"])
