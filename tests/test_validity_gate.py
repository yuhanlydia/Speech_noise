import json
from dataclasses import dataclass

import pytest

from sar.models.base import OptionScores
from sar.validity import (
    evaluate_capability_pairs,
    filter_eligible_records,
    summarize_capability_rows,
)


@dataclass
class Record:
    pair_id: str
    role: str
    waveform_path: str
    waveform_sha256: str
    query: str
    answer: str
    options: list[str]


class FakeWrapper:
    def __init__(self, predictions):
        self.predictions = predictions

    def score_single_token_options(self, audio_path, query, options):
        pred = self.predictions[audio_path]
        return OptionScores(
            list(options),
            [0.0 if option == pred else -3.0 for option in options],
        )


class BrokenCanonicalWrapper:
    def score_single_token_options(self, audio_path, query, options):
        raise ValueError("canonical scorer broken")

    def score_options(self, audio_path, query, options):
        # This fallback must never hide a canonical-scoring integration error.
        return OptionScores(list(options), [0.0, -1.0, -2.0, -3.0])


def _records():
    common1 = dict(pair_id="p1", waveform_path="mix1.wav", waveform_sha256="h1", options=["A", "B", "C", "D"])
    common2 = dict(pair_id="p2", waveform_path="mix2.wav", waveform_sha256="h2", options=["A", "B", "C", "D"])
    return [
        Record(role="ignore", query="mmlu?", answer="C", **common1),
        Record(role="use", query="sound?", answer="B", **common1),
        Record(role="ignore", query="mmlu?", answer="A", **common2),
        Record(role="use", query="sound?", answer="D", **common2),
    ]


def test_capability_gate_requires_both_isolated_tasks_correct(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"pair_id": "p1", "target_wav": "target1.wav", "event_wav": "event1.wav"},
        {"pair_id": "p2", "target_wav": "target2.wav", "event_wav": "event2.wav"},
    ]
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    wrapper = FakeWrapper(
        {
            "target1.wav": "C",
            "event1.wav": "B",
            "target2.wav": "A",
            "event2.wav": "B",  # p2 use target is D, so p2 must be rejected
        }
    )
    capability = evaluate_capability_pairs(wrapper, _records(), source)
    by_pair = {row["pair_id"]: row for row in capability}
    assert by_pair["p1"]["eligible"] is True
    assert by_pair["p2"]["eligible"] is False
    assert by_pair["p2"]["target_only_correct"] is True
    assert by_pair["p2"]["event_only_correct"] is False

    eligible_records = filter_eligible_records(_records(), capability)
    assert {r.pair_id for r in eligible_records} == {"p1"}

    summary = summarize_capability_rows(capability)
    assert summary["num_pairs"] == 2
    assert summary["eligible_pairs"] == 1
    assert summary["eligibility_rate"] == 0.5


def test_capability_gate_does_not_silently_fallback_when_canonical_scorer_breaks(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"pair_id": "p1", "target_wav": "target1.wav", "event_wav": "event1.wav"}) + "\n",
        encoding="utf-8",
    )
    pair = _records()[:2]
    with pytest.raises(ValueError, match="canonical scorer broken"):
        evaluate_capability_pairs(BrokenCanonicalWrapper(), pair, source)
