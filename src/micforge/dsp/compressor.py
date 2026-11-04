"""Feed-forward compressor with soft knee, and a safety limiter."""
from __future__ import annotations

import numpy as np

from .base import Processor
from .util import block_peak, block_rms, db2lin_a, expand_gain, lin2db_a, time_const_coeff


class Compressor(Processor):
    name = "compressor"

    def __init__(
        self,
        sample_rate: int,
        threshold_db: float = -22.0,
        ratio: float = 3.0,
        attack_ms: float = 6.0,
        release_ms: float = 90.0,
        knee_db: float = 6.0,
        makeup_db: float = 3.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(sample_rate, enabled)
        self.threshold_db = threshold_db
        self.ratio = max(1.0, ratio)
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.knee_db = knee_db
        self.makeup_db = makeup_db
        self._size = max(8, int(sample_rate * 0.001))
        self.reset()

    def reset(self) -> None:
        self._gr = 0.0  # smoothed gain change in dB (<= 0)
        self._prev_gain = 1.0
        self._last_min = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n == 0:
            return x
        size = self._size
        over = lin2db_a(block_rms(x, size)) - self.threshold_db
        slope = 1.0 / self.ratio - 1.0
        k = self.knee_db
        if k > 0:
            soft = slope * np.square(over + k / 2.0) / (2.0 * k)
            target = np.where(over <= -k / 2.0, 0.0, np.where(over >= k / 2.0, slope * over, soft))
        else:
            target = np.where(over > 0.0, slope * over, 0.0)

        rate = self.sample_rate / size
        a_att = time_const_coeff(self.attack_ms, rate)
        a_rel = time_const_coeff(self.release_ms, rate)
        g = self._gr
        smoothed = np.empty_like(target)
        for i, t in enumerate(target):
            coef = a_att if t < g else a_rel
            g = t + coef * (g - t)
            smoothed[i] = g
        self._gr = float(g)
        self._last_min = float(smoothed.min())

        gains = db2lin_a(smoothed + self.makeup_db)
        gain = expand_gain(self._prev_gain, gains, n, size)
        self._prev_gain = float(gains[-1])
        return x * gain

    def telemetry(self) -> dict[str, float]:
        return {"gain_reduction_db": -self._last_min}


class Limiter(Processor):
    """Brick-wall-ish peak limiter: instant attack, smoothed release, final clip."""

    name = "limiter"

    def __init__(self, sample_rate: int, ceiling_db: float = -1.5, release_ms: float = 60.0, enabled: bool = True):
        super().__init__(sample_rate, enabled)
        self.ceiling = float(10.0 ** (ceiling_db / 20.0))
        self.release_ms = release_ms
        self._size = max(4, int(sample_rate * 0.0005))
        self.reset()

    def reset(self) -> None:
        self._g = 1.0
        self._last_min = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n == 0:
            return x
        size = self._size
        peaks = block_peak(x, size)
        wanted = np.minimum(1.0, self.ceiling / np.maximum(peaks, 1e-9))
        a_rel = time_const_coeff(self.release_ms, self.sample_rate / size)
        g = self._g
        gains = np.empty_like(wanted)
        for i, w in enumerate(wanted):
            g = w if w < g else w + a_rel * (g - w)
            gains[i] = g
        self._g = float(g)
        self._last_min = float(gains.min())
        y = x * np.repeat(gains, size)[:n].astype(np.float32)
        return np.clip(y, -self.ceiling, self.ceiling)

    def telemetry(self) -> dict[str, float]:
        return {"gain_reduction_db": -20.0 * np.log10(max(self._last_min, 1e-6))}
