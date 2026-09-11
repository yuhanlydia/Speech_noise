from __future__ import annotations

import hashlib
import struct

import numpy as np


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def mix_sources(
    target: np.ndarray,
    distractor: np.ndarray,
    sample_rate: int,
    snr_db: float | None = None,
    tir_db: float | None = None,
    offset_samples: int = 0,
) -> np.ndarray:
    del sample_rate  # retained in the public API and hash metadata
    if offset_samples < 0:
        raise ValueError("offset_samples must be non-negative")
    target = np.asarray(target, dtype=np.float32)
    distractor = np.asarray(distractor, dtype=np.float32)
    total_len = max(len(target), offset_samples + len(distractor))
    t = np.zeros(total_len, dtype=np.float32)
    d = np.zeros(total_len, dtype=np.float32)
    t[: len(target)] = target
    d[offset_samples : offset_samples + len(distractor)] = distractor

    ratio_db = tir_db if tir_db is not None else snr_db
    if ratio_db is not None and np.any(distractor):
        target_rms = _rms(target) if np.any(target) else 1.0
        dist_rms = _rms(distractor)
        desired_dist_rms = target_rms / (10.0 ** (float(ratio_db) / 20.0))
        d *= np.float32(desired_dist_rms / max(dist_rms, 1e-12))

    mixed = t + d
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / np.float32(peak)
    return mixed.astype(np.float32, copy=False)


def sha256_waveform(waveform: np.ndarray, sample_rate: int) -> str:
    x = np.asarray(waveform, dtype="<f4")
    h = hashlib.sha256()
    h.update(struct.pack("<I", int(sample_rate)))
    h.update(struct.pack("<Q", int(x.size)))
    h.update(x.tobytes(order="C"))
    return h.hexdigest()
