from sar.config import ExperimentConfig


def test_daa_config_accepts_declared_blocks_and_empty_layers_for_all_layers():
    cfg = ExperimentConfig.model_validate(
        {
            'model': {},
            'data': {'manifest': 'pairs.jsonl'},
            'method': {
                'name': 'daa',
                'layers': [],
                'daa': {'block_strategy': 'declared', 'max_blocks': 6},
            },
            'output_dir': 'out',
        }
    )
    assert cfg.method.name == 'daa'
    assert cfg.method.layers == []
    assert cfg.method.daa is not None
    assert cfg.method.daa.max_blocks == 6
