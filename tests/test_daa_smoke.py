from sar.config import ExperimentConfig
from sar.daa_smoke import evaluate_daa_with_wrapper
from sar.data.blocks import AcousticBlock
from sar.data.schema import RelevancePairRecord
from sar.models.base import OptionScores


class FakeWrapper:
    def declare_audio_blocks(self, *args, **kwargs):
        return [
            AcousticBlock('B1', 0, 1, 'speech'),
            AcousticBlock('B2', 1, 2, 'dog'),
        ], 'ok'

    def select_audio_blocks(self, audio_path, query, blocks, **kwargs):
        return (['B2'] if 'animal' in query else ['B1']), 'focus'

    def score_single_token_options_daa(
        self,
        audio_path,
        query,
        options,
        blocks,
        selected_ids,
        **kwargs,
    ):
        gold = 'B' if 'animal' in query else 'C'
        return OptionScores(options, [0 if option == gold else -1 for option in options])


def _records():
    common = dict(
        pair_id='p',
        waveform_path='/x.wav',
        waveform_sha256='h',
        options=['A', 'B', 'C', 'D'],
        event_type='dog',
        source_start_s=1.0,
        source_end_s=2.0,
        source_mask_valid=True,
    )
    return [
        RelevancePairRecord(role='ignore', query='number?', answer='C', **common),
        RelevancePairRecord(role='use', query='animal?', answer='B', **common),
    ]


def test_daa_evaluator_reports_protocol_and_selection_metrics():
    cfg = ExperimentConfig.model_validate(
        {
            'model': {},
            'data': {'manifest': 'unused'},
            'method': {
                'name': 'daa',
                'layers': [],
                'daa': {'block_strategy': 'declared', 'max_focus_blocks': 1},
            },
            'output_dir': 'out',
        }
    )
    rows, summary = evaluate_daa_with_wrapper(
        cfg,
        FakeWrapper(),
        _records(),
        duration_resolver=lambda _: 2.0,
    )
    assert len(rows) == 2
    assert summary['protocol_completion_rate'] == 1.0
    assert summary['selection_switch_acc'] == 1.0
    assert summary['pair_switch_acc'] == 1.0
