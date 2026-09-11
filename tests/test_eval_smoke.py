from sar.data.schema import RelevancePairRecord
from sar.eval import evaluate_records
from sar.models.base import OptionScores


class FakeWrapper:
    def score_options(self, audio_path, query, options):
        if 'use' in query:
            return OptionScores(options=list(options), logprobs=[-1.0, -0.1])
        return OptionScores(options=list(options), logprobs=[-0.1, -1.0])


def rec(pair, role, query, answer):
    return RelevancePairRecord(
        pair_id=pair, role=role, waveform_path='same.wav', waveform_sha256=pair,
        query=query, answer=answer, options=['A','B'], event_type='dog'
    )


def test_evaluator_groups_pairs_and_returns_metrics():
    rows = [rec('p1','ignore','ignore question','A'), rec('p1','use','use question','B')]
    result_rows, summary = evaluate_records(FakeWrapper(), rows)
    assert len(result_rows) == 2
    assert summary['pair_switch_acc'] == 1.0
    assert summary['sar'] == 1.0
