import json
from pathlib import Path

import numpy as np
import soundfile as sf

from sar.data.schema import validate_pair_records
from sar.data.relevance_pairs import load_jsonl_records
from scripts.build_mvp_manifest import build_manifest


def test_builder_creates_same_waveform_pair(tmp_path: Path):
    sr = 16000
    target = tmp_path / "target.wav"
    event = tmp_path / "event.wav"
    sf.write(target, np.array([0.0, 0.2, -0.2, 0.1], dtype=np.float32), sr)
    sf.write(event, np.array([0.1, -0.1], dtype=np.float32), sr)
    src = tmp_path / "source.jsonl"
    src.write_text(json.dumps({
        "pair_id": "p1",
        "target_wav": str(target),
        "event_wav": str(event),
        "q_ignore": "What number?",
        "a_ignore": "A",
        "q_use": "What sound?",
        "a_use": "dog",
        "event_type": "dog_bark",
        "snr_db": 0.0,
        "offset_samples": 1,
        "seed": 9
    }) + "\n")
    out_dir = tmp_path / "out"
    manifest = build_manifest(src, out_dir)
    records = load_jsonl_records(manifest)
    validate_pair_records(records)
    assert len(records) == 2
    assert records[0].waveform_sha256 == records[1].waveform_sha256
    assert Path(records[0].waveform_path).read_bytes() == Path(records[1].waveform_path).read_bytes()


def test_builder_records_event_time_span(tmp_path: Path):
    sr = 100
    target = np.zeros(400, dtype=np.float32)
    event = np.ones(100, dtype=np.float32) * 0.1
    target_wav = tmp_path / 'target.wav'; event_wav = tmp_path / 'event.wav'
    sf.write(target_wav, target, sr, subtype='FLOAT'); sf.write(event_wav, event, sr, subtype='FLOAT')
    src = tmp_path / 'src.jsonl'
    src.write_text(json.dumps({
        'pair_id':'p-span','target_wav':str(target_wav),'event_wav':str(event_wav),
        'event_type':'dog','offset_samples':100,'q_ignore':'number?','a_ignore':'A',
        'q_use':'sound?','a_use':'B','options_ignore':['A','B'],'options_use':['A','B']
    })+'\n')
    records = load_jsonl_records(build_manifest(src, tmp_path/'out-span'))
    assert {r.source_start_s for r in records} == {1.0}
    assert {r.source_end_s for r in records} == {2.0}


def test_builder_preserves_explicit_temporal_mask_validity(tmp_path: Path):
    sr=100
    t=np.zeros(200,dtype=np.float32); e=np.ones(50,dtype=np.float32)*0.1
    tw=tmp_path/'t-mask.wav'; ew=tmp_path/'e-mask.wav'; sf.write(tw,t,sr,subtype='FLOAT'); sf.write(ew,e,sr,subtype='FLOAT')
    src=tmp_path/'src-mask.jsonl'
    src.write_text(json.dumps({'pair_id':'pm','target_wav':str(tw),'event_wav':str(ew),'event_type':'siren','source_mask_valid':True,'q_ignore':'q0','a_ignore':'A','q_use':'q1','a_use':'B','options_ignore':['A','B'],'options_use':['A','B']})+'\n')
    recs=load_jsonl_records(build_manifest(src,tmp_path/'out-mask'))
    assert all(r.source_mask_valid for r in recs)
