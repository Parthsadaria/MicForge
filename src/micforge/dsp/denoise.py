"""Streaming spectral noise suppressor.

* 75 %-overlap STFT with sqrt-Hann analysis/synthesis windows
* Noise PSD tracked with windowed minimum statistics (no VAD needed), optionally
  seeded/anchored by a calibrated room-noise profile
* Decision-directed Wiener gain, which avoids the "musical noise" of plain
  spectral subtraction
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import Processor
from .util import db2lin, time_const_coeff

_EPS = 1e-12


class SpectralDenoiser(Processor):
    name = "denoise"

    def __init__(
        self,
        sample_rate: int,
        strength: float = 0.6,
        max_reduction_db: float = 18.0,
        bias: float = 2.2,
        enabled: bool = True,
    ) -> None:
        super().__init__(sample_rate, enabled)
        self.n_fft = 1 << max(6, round(math.log2(sample_rate * 0.0107)))
        self.hop = self.n_fft // 4
        self.n_bins = self.n_fft // 2 + 1
        self.bias = bias
        self.max_reduction_db = max_reduction_db
        self.set_strength(strength)

        n = self.n_fft
        self.win = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n))  # periodic sqrt-Hann
        self._scale = self.hop / float(np.sum(self.win**2))
        self._win_energy = float(np.sum(self.win**2))

        fr = sample_rate / self.hop  # frame rate
        self._a_psd = time_const_coeff(25.0, fr)
        self._a_dd = 0.97
        self._sub_len = max(4, int(0.25 * fr))
        self._sub_count = 6  # ~1.5 s search window
        self._decay = time_const_coeff(15.0, fr)
        self._kernel = np.array([0.25, 0.5, 0.25])
        self._profile: np.ndarray | None = None
        self.reset()

    # ------------------------------------------------------------------ API
    @property
    def latency_samples(self) -> int:
        return self.n_fft - 1

    def set_strength(self, strength: float) -> None:
        self.strength = float(min(1.0, max(0.0, strength)))
        self._over = 1.0 + 2.0 * self.strength
        self._gmin = db2lin(-self.max_reduction_db * (0.4 + 0.6 * self.strength))

    def reset(self) -> None:
        n, nb = self.n_fft, self.n_bins
        self._in = np.zeros(0)
        self._ola = np.zeros(n)
        self._out = np.zeros(n - 1)  # priming = fixed latency of n_fft-1 samples
        self._p_smooth: np.ndarray | None = None
        self._sub_min: np.ndarray | None = None
        self._hist: list[np.ndarray] = []
        self._hist_min: np.ndarray | None = None  # cached elementwise min over self._hist
        self._sub_i = 0
        self._prev_clean = np.zeros(nb)
        self._prev_g = np.ones(nb)
        self._noise = np.full(nb, _EPS)
        self._reduction_db = 0.0
        if self._profile is not None:
            self._seed(self._profile)

    def _seed(self, psd: np.ndarray) -> None:
        self._hist = [psd / self.bias]
        self._hist_min = psd / self.bias
        self._sub_min = (psd / self.bias).copy()
        self._noise = psd.copy()

    def learn(self, noise: np.ndarray) -> np.ndarray:
        """Measure a room-noise profile from a recording of silence."""
        x = np.asarray(noise, dtype=np.float64)
        if x.size < self.n_fft * 4:
            raise ValueError("calibration clip too short")
        frames = sliding_window_view(x, self.n_fft)[:: self.hop]
        spec = np.fft.rfft(frames * self.win, axis=1)
        psd = np.mean(spec.real**2 + spec.imag**2, axis=0)
        self.set_profile(psd)
        return psd

    def set_profile(self, psd: np.ndarray | None) -> bool:
        if psd is None or psd.shape != (self.n_bins,):
            return False
        self._profile = np.maximum(np.asarray(psd, dtype=np.float64), _EPS)
        self._seed(self._profile)
        return True

    @property
    def profile(self) -> np.ndarray | None:
        return self._profile

    def noise_floor_db(self) -> float:
        """Broadband noise level (dBFS RMS) implied by the current noise estimate."""
        psd = self._noise
        full = 2.0 * float(np.sum(psd)) - float(psd[0]) - float(psd[-1])
        var = full / self.n_fft / self._win_energy
        return 10.0 * math.log10(max(var, 1e-20))

    # ------------------------------------------------------------ internals
    def _track_noise(self, p: np.ndarray) -> np.ndarray:
        if self._p_smooth is None:
            self._p_smooth = p.copy()
            self._sub_min = p.copy()
        else:
            self._p_smooth = self._a_psd * self._p_smooth + (1.0 - self._a_psd) * p
            np.minimum(self._sub_min, self._p_smooth, out=self._sub_min)
        self._sub_i += 1
        if self._sub_i >= self._sub_len:
            self._hist.append(self._sub_min.copy())
            self._hist = self._hist[-self._sub_count :]
            self._hist_min = np.min(np.vstack(self._hist), axis=0)
            self._sub_min = self._p_smooth.copy()
            self._sub_i = 0
        cur = self._sub_min if self._hist_min is None else np.minimum(self._hist_min, self._sub_min)
        floor = cur * self.bias
        if self._profile is not None:
            floor = np.maximum(floor, 0.3 * self._profile)
        self._noise = np.maximum(floor, _EPS)
        return self._noise

    def _frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame * self.win)
        p = spec.real**2 + spec.imag**2
        noise = self._track_noise(p) * self._over

        gamma = p / noise
        xi = self._a_dd * (self._prev_clean / noise) + (1.0 - self._a_dd) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, 1e-3)
        g = xi / (1.0 + xi)
        padded = np.pad(g, 1, mode="edge")
        g = np.convolve(padded, self._kernel, mode="valid")
        g = np.clip(g, self._gmin, 1.0)
        g = np.maximum(g, self._prev_g * self._decay)  # no abrupt gain collapse
        self._prev_g = g

        out = spec * g
        self._prev_clean = out.real**2 + out.imag**2
        self._reduction_db = float(-20.0 * np.log10(max(float(np.mean(g)), 1e-6)))
        return np.fft.irfft(out, self.n_fft) * self.win * self._scale

    # ------------------------------------------------------------------ run
    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n == 0:
            return x
        self._in = np.concatenate((self._in, x.astype(np.float64, copy=False)))
        chunks = []
        while self._in.shape[0] >= self.n_fft:
            self._ola += self._frame(self._in[: self.n_fft])
            chunks.append(self._ola[: self.hop].copy())
            self._ola = np.concatenate((self._ola[self.hop :], np.zeros(self.hop)))
            self._in = self._in[self.hop :]
        if chunks:
            self._out = np.concatenate((self._out, *chunks))
        emit = self._out[:n]
        self._out = self._out[n:]
        if emit.shape[0] < n:  # cannot happen with the priming above; stay safe anyway
            emit = np.concatenate((np.zeros(n - emit.shape[0]), emit))
        return emit.astype(np.float32)

    def telemetry(self) -> dict[str, float]:
        return {"reduction_db": self._reduction_db, "noise_floor_db": self.noise_floor_db()}
