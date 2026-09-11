import numpy as np

from sar.data.mixing import mix_sources, sha256_waveform


def test_mix_sources_is_deterministic_and_peak_safe():
    target = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    distractor = np.array([0.4, -0.4], dtype=np.float32)
    a = mix_sources(target, distractor, sample_rate=16000, snr_db=0.0, offset_samples=1)
    b = mix_sources(target, distractor, sample_rate=16000, snr_db=0.0, offset_samples=1)
    assert np.allclose(a, b)
    assert np.max(np.abs(a)) <= 1.0
    assert sha256_waveform(a, 16000) == sha256_waveform(b, 16000)


def test_mix_sources_places_distractor_at_requested_offset():
    target = np.zeros(5, dtype=np.float32)
    distractor = np.array([1.0, 0.5], dtype=np.float32)
    mixed = mix_sources(target, distractor, sample_rate=16000, offset_samples=2)
    assert np.allclose(mixed[:2], 0.0)
    assert np.any(np.abs(mixed[2:4]) > 0)
