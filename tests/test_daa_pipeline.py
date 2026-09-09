from dataclasses import dataclass

from sar.daa_pipeline import run_daa_pair, summarize_daa_rows
from sar.data.blocks import AcousticBlock
from sar.models.base import OptionScores


@dataclass
class Record:
    pair_id: str
    role: str
    waveform_path: str
    waveform_sha256: str
    query: str
    answer: str
    options: list[str]
    source_start_s: float = 1.0
    source_end_s: float = 2.0
    source_mask_valid: bool = True


class FakeWrapper:
    def __init__(self):
        self.declare_calls = 0

    def declare_audio_blocks(self, audio_path, *, duration_s, max_blocks, max_new_tokens):
        self.declare_calls += 1
        return [
            AcousticBlock('B1', 0, 1, 'speaker'),
            AcousticBlock('B2', 1, 2, 'dog bark'),
        ], '<audio_blocks>...'

    def select_audio_blocks(
        self,
        audio_path,
        query,
        blocks,
        *,
        max_selected,
        max_new_tokens,
    ):
        ids = ['B2'] if 'animal' in query else ['B1']
        return ids, f'<focus_audio blocks="{ids[0]}">'

    def score_single_token_options_daa(
        self,
        audio_path,
        query,
        options,
        blocks,
        selected_ids,
        *,
        layers,
    ):
        gold = 'B' if 'animal' in query else 'C'
        return OptionScores(options, [0.0 if option == gold else -2.0 for option in options])


def _records():
    common = dict(
        pair_id='p1',
        waveform_path='/x.wav',
        waveform_sha256='abc',
        options=['A', 'B', 'C', 'D'],
    )
    return [
        Record(role='ignore', query='What number was spoken?', answer='C', **common),
        Record(role='use', query='What animal is audible?', answer='B', **common),
    ]


def test_pipeline_declares_blocks_once_and_switches_focus_by_query():
    wrapper = FakeWrapper()
    rows = run_daa_pair(
        wrapper,
        _records(),
        duration_s=2.0,
        block_strategy='declared',
        block_seconds=1.0,
        max_blocks=4,
        max_focus_blocks=1,
        scan_max_new_tokens=64,
        select_max_new_tokens=16,
        layers=[0],
    )
    assert wrapper.declare_calls == 1
    by_role = {row['role']: row for row in rows}
    assert by_role['ignore']['selected_blocks'] == ['B1']
    assert by_role['use']['selected_blocks'] == ['B2']
    assert by_role['ignore']['event_selected'] is False
    assert by_role['use']['event_selected'] is True
    summary = summarize_daa_rows(rows)
    assert summary['selection_switch_acc'] == 1.0
    assert summary['pair_switch_acc'] == 1.0
    assert summary['reasoning_acc_given_use_selection'] == 1.0
