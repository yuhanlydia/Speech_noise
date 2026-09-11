from pathlib import Path
import pytest

from sar.config import load_experiment_config


def test_mvp_config_loads_strictly():
    cfg = load_experiment_config(Path('configs/experiment/mvp_diagnostic.yaml'))
    assert cfg.model.model_id == 'Qwen/Qwen2.5-Omni-3B'
    assert cfg.model.quantization == 'nf4'
    assert cfg.method.name == 'base'


def test_unknown_config_key_is_rejected(tmp_path: Path):
    path = tmp_path / 'bad.yaml'
    path.write_text('model:\n  model_id: x\n  unknown: 1\ndata:\n  manifest: a\nmethod:\n  name: base\noutput_dir: out\n')
    with pytest.raises(Exception):
        load_experiment_config(path)


def test_qacr_training_config_exposes_checkpoint_and_optimization_fields():
    cfg = load_experiment_config(Path('configs/experiment/mvp_qacr.yaml'))
    assert cfg.method.name == 'qacr'
    assert cfg.method.checkpoint == 'artifacts/qacr/router.pt'
    assert cfg.training.epochs == 3
    assert cfg.training.learning_rate == pytest.approx(1e-3)
