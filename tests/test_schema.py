import pytest
from pydantic import ValidationError

from sar.data.schema import RelevancePairRecord, validate_pair_records


def make(role: str, sha: str = "abc") -> RelevancePairRecord:
    return RelevancePairRecord(
        pair_id="pair-1",
        role=role,
        waveform_path="x.wav",
        waveform_sha256=sha,
        query=f"query-{role}",
        answer="A",
        options=["A", "B"],
        event_type="dog_bark",
        seed=7,
    )


def test_validate_pair_records_requires_exact_use_ignore_pair():
    with pytest.raises(ValueError, match="exactly one use and one ignore"):
        validate_pair_records([make("use")])


def test_validate_pair_records_rejects_mismatched_waveform_hashes():
    with pytest.raises(ValueError, match="waveform hash"):
        validate_pair_records([make("use", "abc"), make("ignore", "xyz")])


def test_role_is_strict_literal():
    with pytest.raises(ValidationError):
        RelevancePairRecord(
            pair_id="pair-1",
            role="other",
            waveform_path="x.wav",
            waveform_sha256="abc",
            query="q",
            answer="A",
            event_type="dog",
        )
