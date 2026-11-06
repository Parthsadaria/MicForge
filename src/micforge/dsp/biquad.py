"""RBJ-cookbook biquads, run as streaming second-order sections."""
from __future__ import annotations

import math
from typing import Iterable, Protocol

import numpy as np

from .base import Processor


def sosfilt(sos: np.ndarray, x: np.ndarray, zi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy Direct Form II Transposed Second-Order Sections (biquad) filter.

    Computes identical numerical output to scipy.signal.sosfilt without requiring
    the 65 MB memory footprint of loading compiled SciPy BLAS/LAPACK runtime DLLs.
    """
    sos = np.atleast_2d(np.asarray(sos, dtype=np.float64))
    n_sections = sos.shape[0]
    if zi is None:
        zi_out = np.zeros((n_sections, 2), dtype=np.float64)
    else:
        zi_out = np.array(zi, dtype=np.float64, copy=True)

    y = np.array(x, dtype=np.float64, copy=True)
    for s in range(n_sections):
        b0, b1, b2, _, a1, a2 = sos[s]
        z1, z2 = zi_out[s, 0], zi_out[s, 1]
        for i in range(len(y)):
            xi = y[i]
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y[i] = yi
        zi_out[s, 0] = z1
        zi_out[s, 1] = z2
    return y.astype(np.float32, copy=False), zi_out


class BandLike(Protocol):
    kind: str
    freq: float
    gain_db: float
    q: float


def design_biquad(kind: str, fs: float, f0: float, q: float = 0.7071, gain_db: float = 0.0) -> np.ndarray:
    """Return one normalised SOS row [b0 b1 b2 1 a1 a2]."""
    f0 = min(max(f0, 10.0), 0.49 * fs)
    w0 = 2.0 * math.pi * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    A = 10.0 ** (gain_db / 40.0)

    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind in ("lowshelf", "highshelf"):
        beta = 2.0 * math.sqrt(A) * alpha
        if kind == "lowshelf":
            b = [
                A * ((A + 1) - (A - 1) * cw + beta),
                2 * A * ((A - 1) - (A + 1) * cw),
                A * ((A + 1) - (A - 1) * cw - beta),
            ]
            a = [(A + 1) + (A - 1) * cw + beta, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - beta]
        else:
            b = [
                A * ((A + 1) + (A - 1) * cw + beta),
                -2 * A * ((A - 1) + (A + 1) * cw),
                A * ((A + 1) + (A - 1) * cw - beta),
            ]
            a = [(A + 1) - (A - 1) * cw + beta, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - beta]
    else:
        raise ValueError(f"unknown filter kind: {kind!r}")

    a0 = a[0]
    return np.array([b[0] / a0, b[1] / a0, b[2] / a0, 1.0, a[1] / a0, a[2] / a0])


class BiquadFilter(Processor):
    """A cascade of biquads with persistent state (safe for block-wise streaming)."""

    def __init__(self, sample_rate: int, sos: np.ndarray, name: str = "biquad", enabled: bool = True) -> None:
        super().__init__(sample_rate, enabled)
        self.name = name
        self.sos = np.atleast_2d(np.asarray(sos, dtype=np.float64))
        self._zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)
        self._scratch = np.zeros(0, dtype=np.float64)

    @classmethod
    def from_bands(
        cls, sample_rate: int, bands: Iterable[BandLike], name: str = "eq", enabled: bool = True
    ) -> "BiquadFilter":
        rows = [design_biquad(b.kind, sample_rate, b.freq, b.q, b.gain_db) for b in bands]
        if not rows:  # identity section keeps shapes valid
            rows = [np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])]
        return cls(sample_rate, np.vstack(rows), name=name, enabled=enabled)

    @classmethod
    def highpass(cls, sample_rate: int, freq: float, order: int = 4, name: str = "highpass", enabled: bool = True):
        """Butterworth-ish high-pass built from cascaded biquads (order 2 per section)."""
        sections = max(1, order // 2)
        # Butterworth Q values for an even order
        n = 2 * sections
        qs = [1.0 / (2.0 * math.cos(math.pi * (2 * k + 1) / (2 * n))) for k in range(sections)]
        rows = [design_biquad("highpass", sample_rate, max(freq, 10.0), q) for q in qs]
        return cls(sample_rate, np.vstack(rows), name=name, enabled=enabled)

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if self._scratch.shape[0] != n:
            self._scratch = np.empty(n, dtype=np.float64)
        self._scratch[:] = x

        y = self._scratch
        zi = self._zi
        sos = self.sos
        for s in range(sos.shape[0]):
            b0, b1, b2, _, a1, a2 = sos[s]
            z1 = zi[s, 0]
            z2 = zi[s, 1]
            for i in range(n):
                xi = y[i]
                yi = b0 * xi + z1
                z1 = b1 * xi - a1 * yi + z2
                z2 = b2 * xi - a2 * yi
                y[i] = yi
            zi[s, 0] = z1
            zi[s, 1] = z2

        return y.astype(np.float32, copy=False)

    def reset(self) -> None:
        self._zi[:] = 0.0
