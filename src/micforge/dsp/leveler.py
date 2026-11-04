"""Slow automatic level control (so quiet and loud talking end up similar)."""
from __future__ import annotations

import math

import numpy as np

from .base import Processor
from .util import block_rms, db2lin_a, expand_gain, time_const_coeff


class Leveler(Processor):
    name = "leveler"

    def __init__(
        self,
        sample_rate: int,
        target_db: float = -20.0,
        max_gain_db: float = 9.0,
        max_cut_db: float = 6.0,
        gate_db: float = -48.0,
        attack_ms: float = 250.0,
        release_ms: float = 1500.0,
        window_ms: float = 400.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(sample_rate, enabled)
        self.target_db = target_db
        self.max_gain_db = max_gain_db
        self.max_cut_db = max_cut_db
        self.gate_db = gate_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.window_ms = window_ms
        self._size = max(16, int(sample_rate * 0.01))
        self.reset()

    def reset(self) -> None:
        self._g_db = 0.0
        self._power: float | None = None
        self._prev = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n == 0:
            return x
        size = self._size
        rate = self.sample_rate / size
        a_win = time_const_coeff(self.window_ms, rate)
        a_dn = time_const_coeff(self.attack_ms, rate)
        a_up = time_const_coeff(self.release_ms, rate)

        rms = block_rms(x, size)
        g = self._g_db
        out = np.empty(rms.shape[0])
        for i, r in enumerate(rms):
            p = float(r * r)
            level_db = 10.0 * math.log10(max(p, 1e-20))
            if level_db > self.gate_db:  # only learn from actual speech
                self._power = p if self._power is None else a_win * self._power + (1 - a_win) * p
            if self._power is not None and level_db > self.gate_db:
                lv = 10.0 * math.log10(max(self._power, 1e-20))
                desired = min(self.max_gain_db, max(-self.max_cut_db, self.target_db - lv))
                coef = a_dn if desired < g else a_up
                g = desired + coef * (g - desired)
            out[i] = g
        self._g_db = float(g)
        gains = db2lin_a(out)
        gain = expand_gain(self._prev, gains, n, size)
        self._prev = float(gains[-1])
        return x * gain

    def telemetry(self) -> dict[str, float]:
        return {"gain_db": self._g_db}
