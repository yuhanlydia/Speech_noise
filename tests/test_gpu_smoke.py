import json
from pathlib import Path

from sar.gpu_smoke import dry_run


def test_dry_run_validates_config_and_manifest_without_loading_gpu(tmp_path: Path):
    manifest = tmp_path / 'pairs.jsonl'
    rows = [
        dict(pair_id='p', role='ignore', waveform_path='x.wav', waveform_sha256='h', query='q0', answer='A', options=['A','B'], event_type='dog'),
        dict(pair_id='p', role='use', waveform_path='x.wav', waveform_sha256='h', query='q1', answer='B', options=['A','B'], event_type='dog'),
    ]
    manifest.write_text('\n'.join(json.dumps(x) for x in rows) + '\n')
    cfg = tmp_path / 'cfg.yaml'
    cfg.write_text(f'''model:\n  model_id: Qwen/Qwen2.5-Omni-3B\n  quantization: nf4\n  thinker_only: true\n  device_map: auto\n  attention_backend: eager\ndata:\n  manifest: {manifest}\n  seed: 0\nmethod:\n  name: qacr\n  layers: [0]\n  router_dim: 16\n  lambda_switch: 1.0\n  lambda_identity: 0.1\noutput_dir: {tmp_path / "out"}\nseed: 0\n''')
    info = dry_run(cfg)
    assert info['pairs'] == 1
    assert info['records'] == 2
    assert info['method'] == 'qacr'
    assert info['model'] == 'Qwen/Qwen2.5-Omni-3B'
