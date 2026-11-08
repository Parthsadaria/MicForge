"""Soft noise gate (downward expander) with lookahead and dual-trigger VAD speech onset protection."""
from __future__ import annotations

import numpy as np

from .base import Processor
from .util import block_rms, db2lin, expand_gain, lin2db, lin2db_a, time_const_coeff


class NoiseGate(Processor):
    name = "gate"

    def __init__(
        self,
        sample_rate: int,
        threshold_db: float = -52.0,
        range_db: float = 20.0,
        attack_ms: float = 2.0,
        hold_ms: float = 120.0,
        release_ms: float = 150.0,
        hysteresis_db: float = 4.0,
        lookahead_ms: float = 10.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(sample_rate, enabled)
        self.threshold_db = threshold_db
        self.range_db = range_db
        self.attack_ms = attack_ms
        self.hold_ms = hold_ms
        self.release_ms = release_ms
        self.hysteresis_db = hysteresis_db
        self.lookahead_ms = max(0.0, float(lookahead_ms))
        self.speech_prob: float | None = None
        self._size = max(8, int(sample_rate * 0.002))

        # Circular zero-allocation lookahead buffer to preserve speech onsets & consonants
        self.lookahead_samples = int(round(self.lookahead_ms * 1e-3 * sample_rate))
        self._look_buf = np.zeros(max(1, self.lookahead_samples), dtype=np.float32)
        self._delayed_scratch = np.zeros(max(4096, self.lookahead_samples * 2 + 1024), dtype=np.float32)
        self.reset()

    @property
    def latency_samples(self) -> int:
        return self.lookahead_samples if self.enabled else 0

    @property
    def is_open(self) -> bool:
        return self._open

    def reset(self) -> None:
        self._open = False
        self._hold_left = 0
        self._gain = db2lin(-self.range_db)
        self._prev = self._gain
        self._look_buf.fill(0.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n == 0:
            return x
        size = self._size
        rate = self.sample_rate / size
        a_att = time_const_coeff(self.attack_ms, rate)
        a_rel = time_const_coeff(self.release_ms, rate)
        hold_blocks = int(round(self.hold_ms * 1e-3 * rate))
        floor = db2lin(-self.range_db)
        close_thr = self.threshold_db - self.hysteresis_db

        levels = lin2db_a(block_rms(x, size))
        out = np.empty(levels.shape[0])
        g, is_open, hold = self._gain, self._open, self._hold_left
        for i, lv in enumerate(levels):
            # Dual-trigger speech onset:
            # Detects initial consonants (p, t, k, s, ch) using energy rise coupled with
            # mask probability, avoiding the slow ramp latency that cuts off leading words.
            if self.speech_prob is not None:
                is_speech_frame = (self.speech_prob >= 0.28) or (self.speech_prob >= 0.16 and lv >= self.threshold_db)
                is_hold_frame = (self.speech_prob >= 0.14) or (lv >= close_thr and self.speech_prob >= 0.10)
            else:
                is_speech_frame = lv >= self.threshold_db
                is_hold_frame = lv >= close_thr
            if is_open:
                if is_hold_frame:
                    hold = hold_blocks
                elif hold > 0:
                    hold -= 1
                else:
                    is_open = False
            elif is_speech_frame:
                is_open, hold = True, hold_blocks
            target = 1.0 if is_open else floor
            coef = a_att if target > g else a_rel
            g = target + coef * (g - target)
            out[i] = g
        self._gain, self._open, self._hold_left = g, is_open, hold

        gain = expand_gain(self._prev, out, n, size)
        self._prev = float(out[-1])

        # Apply gain to lookahead-delayed audio to guarantee leading consonants are preserved
        L = self.lookahead_samples
        if L > 0:
            if self._delayed_scratch.shape[0] < n:
                self._delayed_scratch = np.zeros(n + 1024, dtype=np.float32)
            if n >= L:
                self._delayed_scratch[:L] = self._look_buf[:L]
                self._delayed_scratch[L:n] = x[: n - L]
                self._look_buf[:L] = x[n - L : n]
            else:
                self._delayed_scratch[:n] = self._look_buf[:n]
                self._look_buf[: L - n] = self._look_buf[n:L]
                self._look_buf[L - n : L] = x[:n]

            return self._delayed_scratch[:n] * gain

        return x * gain

        return x * gain

    def telemetry(self) -> dict[str, float]:
        return {
            "open": 1.0 if self._open else 0.0,
            "gain_db": lin2db(self._gain),
            "attenuation_db": float(-lin2db(self._gain)),
        }
