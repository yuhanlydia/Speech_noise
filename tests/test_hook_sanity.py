import pytest

from sar.hook_sanity import compare_score_vectors
from sar.models.base import OptionScores


def test_compare_score_vectors_reports_identity_error():
    reference = OptionScores(["A", "B", "C", "D"], [-1.0, -2.0, -3.0, -4.0])
    masked = OptionScores(["A", "B", "C", "D"], [-1.0, -2.000001, -3.0, -4.0])
    report = compare_score_vectors(reference, masked, tolerance=1e-4)
    assert report["ok"] is True
    assert report["predictions_match"] is True
    assert report["max_abs_logprob_diff"] == pytest.approx(1e-6)


def test_compare_score_vectors_rejects_large_hook_drift():
    reference = OptionScores(["A", "B"], [-1.0, -2.0])
    masked = OptionScores(["A", "B"], [-1.1, -2.0])
    report = compare_score_vectors(reference, masked, tolerance=1e-4)
    assert report["ok"] is False
    assert report["max_abs_logprob_diff"] == pytest.approx(0.1)
