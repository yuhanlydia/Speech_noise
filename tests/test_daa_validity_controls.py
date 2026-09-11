from dataclasses import dataclass

from sar.daa_pipeline import (
    _oracle_focus,
    _target_coverage,
    summarize_daa_rows,
)
from sar.data.blocks import AcousticBlock


@dataclass
class Record:
    pair_id: str = "p1"
    role: str = "ignore"
    waveform_path: str = "mix.wav"
    waveform_sha256: str = "hash"
    query: str = "q"
    answer: str = "A"
    options: list[str] = None
    source_start_s: float = 2.0
    source_end_s: float = 3.0
    source_mask_valid: bool = True

    def __post_init__(self):
        if self.options is None:
            self.options = ["A", "B", "C", "D"]


def _blocks():
    return [
        AcousticBlock("B1", 0.0, 1.0, "segment 1"),
        AcousticBlock("B2", 1.0, 2.0, "segment 2"),
        AcousticBlock("B3", 2.0, 3.0, "event"),
    ]


def test_oracle_focus_uses_event_for_use_and_target_region_for_ignore():
    use = Record(role="use")
    ignore = Record(role="ignore")
    assert _oracle_focus(use, _blocks()) == ["B3"]
    assert _oracle_focus(ignore, _blocks()) == ["B1", "B2"]


def test_target_coverage_measures_fraction_of_pre_event_context_kept():
    record = Record(role="ignore", source_start_s=2.0)
    assert _target_coverage(record, _blocks(), ["B1", "B2"]) == 1.0
    assert _target_coverage(record, _blocks(), ["B1"]) == 0.5


def test_selection_switch_requires_target_coverage_not_only_event_avoidance():
    rows = [
        {
            "pair_id": "p1",
            "role": "ignore",
            "correct": True,
            "event_selected": False,
            "target_coverage": 0.25,
            "ignore_selection_valid": False,
        },
        {
            "pair_id": "p1",
            "role": "use",
            "correct": True,
            "event_selected": True,
            "target_coverage": None,
            "ignore_selection_valid": None,
        },
    ]
    summary = summarize_daa_rows(rows)
    assert summary["ignore_event_avoid_rate"] == 1.0
    assert summary["selection_switch_acc"] == 0.0
    assert summary["ignore_target_coverage_mean"] == 0.25


def test_prompt_only_rows_can_be_identified_separately_from_kv_mask_rows():
    rows = [
        {
            "pair_id": "p1",
            "role": "ignore",
            "correct": True,
            "event_selected": False,
            "target_coverage": 1.0,
            "ignore_selection_valid": True,
            "apply_kv_mask": False,
        },
        {
            "pair_id": "p1",
            "role": "use",
            "correct": True,
            "event_selected": True,
            "target_coverage": None,
            "ignore_selection_valid": None,
            "apply_kv_mask": False,
        },
    ]
    summary = summarize_daa_rows(rows)
    assert summary["pair_switch_acc"] == 1.0
    assert summary["selection_switch_acc"] == 1.0
