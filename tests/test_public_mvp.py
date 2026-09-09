import random

import numpy as np
import pytest

from sar.data.public_mvp import (
    DISPLAY_EVENT,
    esc50_use_query,
    format_mmlu_spoken_prompt,
    ignore_query,
    librispeech_use_query,
    protocol_spec,
    resample_linear,
)


def test_protocol_keeps_development_and_confirmation_sources_disjoint():
    dev = protocol_spec("dev")
    confirm = protocol_spec("confirm")
    assert dev.mmlu_split == "validation"
    assert confirm.mmlu_split == "test"
    assert set(dev.esc50_folds).isdisjoint(confirm.esc50_folds)
    assert dev.librispeech_split == "validation.clean"
    assert confirm.librispeech_split == "test.clean"


def test_format_mmlu_spoken_prompt_contains_all_four_choices():
    text = format_mmlu_spoken_prompt("Two plus two is?", ["1", "2", "4", "8"])
    assert "Question. Two plus two is?" in text
    assert "A. 1" in text
    assert "C. 4" in text
    assert "D. 8" in text


def test_ignore_query_explicitly_marks_later_event_irrelevant():
    text = ignore_query().lower()
    assert "background event" in text
    assert "irrelevant" in text
    assert "a, b, c, or d" in text


def test_esc50_use_query_is_deterministic_and_encodes_correct_letter():
    q1, a1 = esc50_use_query("dog", rng=random.Random(11))
    q2, a2 = esc50_use_query("dog", rng=random.Random(11))
    assert (q1, a1) == (q2, a2)
    assert a1 in {"A", "B", "C", "D"}
    assert DISPLAY_EVENT["dog"] in q1


def test_librispeech_use_query_contains_correct_phrase_once():
    query, answer = librispeech_use_query(
        "THE QUICK BROWN FOX",
        ["HELLO WORLD", "A SHORT TEST", "GOOD MORNING", "ANOTHER PHRASE"],
        rng=random.Random(3),
    )
    assert query.count("THE QUICK BROWN FOX") == 1
    assert answer in {"A", "B", "C", "D"}


def test_resample_linear_preserves_duration_to_rounding():
    x = np.linspace(-1.0, 1.0, 8000, dtype=np.float32)
    y = resample_linear(x, 8000, 16000)
    assert y.dtype == np.float32
    assert len(y) == 16000
    assert y[0] == pytest.approx(x[0], abs=1e-6)


def test_protocol_rejects_unknown_name():
    with pytest.raises(ValueError, match="protocol"):
        protocol_spec("future")
