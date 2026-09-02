import pytest

from sar.metrics import compute_pair_metrics


def test_compute_pair_metrics_matches_hand_calculation():
    rows = [
        {"pair_id": "p1", "role": "ignore", "correct": True},
        {"pair_id": "p1", "role": "use", "correct": True},
        {"pair_id": "p2", "role": "ignore", "correct": True},
        {"pair_id": "p2", "role": "use", "correct": False},
        {"pair_id": "p3", "role": "ignore", "correct": False},
        {"pair_id": "p3", "role": "use", "correct": True},
        {"pair_id": "p4", "role": "ignore", "correct": False},
        {"pair_id": "p4", "role": "use", "correct": False},
    ]
    m = compute_pair_metrics(rows)
    assert m["ignore_acc"] == pytest.approx(0.5)
    assert m["use_acc"] == pytest.approx(0.5)
    assert m["sar"] == pytest.approx(0.5)
    assert m["pair_switch_acc"] == pytest.approx(0.25)
