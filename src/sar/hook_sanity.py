from __future__ import annotations

from sar.models.base import OptionScores


def compare_score_vectors(
    reference: OptionScores,
    masked: OptionScores,
    *,
    tolerance: float = 1e-4,
) -> dict:
    if reference.options != masked.options:
        raise ValueError("score vectors must use identical option ordering")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    diffs = [
        abs(float(a) - float(b))
        for a, b in zip(reference.logprobs, masked.logprobs)
    ]
    max_diff = max(diffs) if diffs else 0.0
    predictions_match = reference.predicted_option == masked.predicted_option
    return {
        "ok": bool(predictions_match and max_diff <= tolerance),
        "predictions_match": bool(predictions_match),
        "max_abs_logprob_diff": float(max_diff),
        "tolerance": float(tolerance),
    }
