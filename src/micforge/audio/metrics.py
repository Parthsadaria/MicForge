"""Level metering and spectrum analysis for the live UI."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..dsp.util import EPS


@dataclass(frozen=True)
class LevelSnapshot:
    peak_db: float = -120.0  # ballistic peak (falls smoothly)
    rms_db: float = -120.0
    hold_db: float = -120.0
    clip_count: int = 0
    clipping: bool = False  # clipped within the last second


class LevelMeter:
    def __init__(self, sample_rate: int, fall_db_per_s: float = 26.0, hold_s: float = 1.5, rms_s: float = 0.3):
        self.sr = sample_rate
        self.fall = fall_db_per_s
        self.hold_s = hold_s
        self.rms_s = rms_s
        self.reset()

    def reset(self) -> None:
        self._peak = -120.0
        self._hold = -120.0
        self._hold_age = 0.0
        self._ms = 0.0
        self._clips = 0
        self._last_clip = -1e9
        self._snap = LevelSnapshot()

    @property
    def snapshot(self) -> LevelSnapshot:
        return self._snap

    def update(self, x: np.ndarray) -> LevelSnapshot:
        n = x.shape[0]
        if n == 0:
            return self._snap
        dt = n / self.sr
        a = np.abs(x)
        pk = float(a.max())
        pk_db = 20.0 * math.log10(max(pk, EPS))
        ms = float(np.mean(np.square(x, dtype=np.float64)))
        k = math.exp(-dt / self.rms_s)
        self._ms = k * self._ms + (1.0 - k) * ms
        self._peak = max(pk_db, self._peak - self.fall * dt)
        if pk_db >= self._hold:
            self._hold, self._hold_age = pk_db, 0.0
        else:
            self._hold_age += dt
            if self._hold_age > self.hold_s:
                self._hold = max(pk_db, self._hold - self.fall * dt)
        clipped = int(np.count_nonzero(a >= 0.999))
        now = time.monotonic()
        if clipped:
            self._clips += clipped
            self._last_clip = now
        self._snap = LevelSnapshot(
            peak_db=max(self._peak, -120.0),
            rms_db=10.0 * math.log10(max(self._ms, EPS * EPS)),
            hold_db=max(self._hold, -120.0),
            clip_count=self._clips,
            clipping=(now - self._last_clip) < 1.0,
        )
        return self._snap


class SpectrumAnalyzer:
    """Log-spaced band levels (dBFS per band) with fast attack / slow release."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 2048,
        n_bands: int = 32,
        f_lo: float = 60.0,
        f_hi: float = 16000.0,
        interval_s: float = 0.05,
    ) -> None:
        self.sr = sample_rate
        self.n_fft = n_fft
        self.n_bands = n_bands
        self.f_lo = f_lo
        self.f_hi = min(f_hi, 0.45 * sample_rate)
        self.interval = interval_s
        self._win = np.hanning(n_fft)
        self._norm = float(np.sum(self._win) / 2.0)
        edges = np.geomspace(self.f_lo, self.f_hi, n_bands + 1)
        idx = np.round(edges / (sample_rate / n_fft)).astype(int)
        for i in range(1, idx.size):
            if idx[i] <= idx[i - 1]:
                idx[i] = idx[i - 1] + 1
        self._idx = np.minimum(idx, n_fft // 2)
        self._counts = np.maximum(np.diff(self._idx), 1)
        self._buf = np.zeros(n_fft, dtype=np.float32)
        self._bands = np.full(n_bands, -120.0)
        self._last = 0.0
        self._dirty = False

    def _compute(self) -> None:
        self._dirty = False
        spec = np.fft.rfft(self._buf * self._win)
        power = spec.real**2 + spec.imag**2
        sums = np.add.reduceat(power[: self._idx[-1]], self._idx[:-1])
        band_power = sums / self._counts / (self._norm**2)
        db = 10.0 * np.log10(np.maximum(band_power, 1e-14))
        self._bands = np.maximum(db, self._bands - 3.0)

    @property
    def bands(self) -> np.ndarray:
        if self._dirty:
            self._compute()
        return self._bands

    def feed(self, x: np.ndarray) -> None:
        n = x.shape[0]
        if n >= self.n_fft:
            self._buf[:] = x[-self.n_fft :]
        elif n > 0:
            self._buf[:-n] = self._buf[n:]
            self._buf[-n:] = x
        self._dirty = True

    def band_centers(self) -> np.ndarray:
        e = np.geomspace(self.f_lo, self.f_hi, self.n_bands + 1)
        return np.sqrt(e[:-1] * e[1:])
