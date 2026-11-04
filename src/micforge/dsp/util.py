"""Small numeric helpers shared by the DSP stages."""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-10


def db2lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin2db(x: float) -> float:
    return 20.0 * math.log10(max(x, EPS))


def db2lin_a(db: np.ndarray | float) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)


def lin2db_a(x: np.ndarray | float) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), EPS))


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    return 10.0 * math.log10(max(float(np.mean(np.square(x, dtype=np.float64))), EPS * EPS))


def peak_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    return lin2db(float(np.max(np.abs(x))))


def time_const_coeff(ms: float, rate_hz: float) -> float:
    """One-pole smoothing coefficient for a time constant of `ms` at `rate_hz`."""
    if ms <= 0:
        return 0.0
    return math.exp(-1.0 / (ms * 1e-3 * rate_hz))


def n_blocks(n: int, size: int) -> int:
    return -(-n // size)


def block_peak(x: np.ndarray, size: int) -> np.ndarray:
    n = x.shape[0]
    m = n_blocks(n, size)
    pad = m * size - n
    a = np.abs(x)
    if pad:
        a = np.concatenate((a, np.zeros(pad, dtype=a.dtype)))
    return a.reshape(m, size).max(axis=1)


def block_rms(x: np.ndarray, size: int) -> np.ndarray:
    n = x.shape[0]
    m = n_blocks(n, size)
    pad = m * size - n
    sq = np.square(x, dtype=np.float64)
    if pad:
        sq = np.concatenate((sq, np.zeros(pad)))
    sums = sq.reshape(m, size).sum(axis=1)
    counts = np.full(m, float(size))
    if pad:
        counts[-1] = size - pad
    return np.sqrt(sums / counts)


def expand_gain(prev: float, targets: np.ndarray, n: int, size: int) -> np.ndarray:
    """Linearly ramp a per-sub-block gain curve out to one gain value per sample."""
    m = targets.shape[0]
    edges = np.minimum(np.arange(1, m + 1) * size, n)
    xp = np.concatenate(([0], edges))
    fp = np.concatenate(([prev], targets))
    return np.interp(np.arange(n), xp, fp).astype(np.float32)
