"""Split-band de-esser: tames harsh sibilance, especially after a presence boost."""
from __future__ import annotations

import numpy as np

from .base import Processor
from .biquad import BiquadFilter, design_biquad
from .util import block_rms, db2lin_a, expand_gain, lin2db_a, time_const_coeff


class DeEsser(Processor):
    name = "deesser"

    def __init__(
        self,
        sample_rate: int,
        freq_hz: float = 7000.0,
        threshold_db: float = -32.0,
        ratio: float = 3.0,
        max_reduction_db: float = 12.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(sample_rate, enabled)
        f = min(freq_hz, 0.42 * sample_rate)
        self._band = BiquadFilter(sample_rate, design_biquad("bandpass", sample_rate, f, q=1.2)[None, :])
        self.threshold_db = threshold_db
        self.ratio = max(1.0, ratio)
        self.max_reduction_db = max_reduction_db
        self._size = max(8, int(sample_rate * 0.001))
        self.reset()

    def reset(self) -> None:
        self._band.reset()
        self._gr = 0.0
        self._prev = 1.0
        self._last_min = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n == 0:
            return x
        size = self._size
        hi = self._band.process(x)
        over = np.maximum(lin2db_a(block_rms(hi, size)) - self.threshold_db, 0.0)
        target = np.maximum(-over * (1.0 - 1.0 / self.ratio), -self.max_reduction_db)

        rate = self.sample_rate / size
        a_att, a_rel = time_const_coeff(1.0, rate), time_const_coeff(30.0, rate)
        g = self._gr
        sm = np.empty_like(target)
        for i, t in enumerate(target):
            coef = a_att if t < g else a_rel
            g = t + coef * (g - t)
            sm[i] = g
        self._gr = float(g)
        self._last_min = float(sm.min())

        gains = db2lin_a(sm)
        gain = expand_gain(self._prev, gains, n, size)
        self._prev = float(gains[-1])
        return x - hi * (1.0 - gain)

    def telemetry(self) -> dict[str, float]:
        return {"gain_reduction_db": -self._last_min}
