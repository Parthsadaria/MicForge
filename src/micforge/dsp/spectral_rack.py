"""Unified Spectral Rack: Shared STFT pass for AEC, Denoiser, and De-Verb.

Eliminates cascading framing latencies, redundant FFT/iFFT transforms, and phase
comb-filtering artifacts by running echo cancellation, neural/spectral denoising,
naturalness dry blend, and room dereverberation in a single frequency-domain pass.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .aec import AcousticEchoCanceller
from .base import Processor
from .deverb import RoomDereverberation

log = logging.getLogger("micforge.spectral_rack")


class SpectralRack(Processor):
    name = "spectral_rack"

    def __init__(
        self,
        sample_rate: int,
        aec: AcousticEchoCanceller,
        denoise: Processor,
        deverb: RoomDereverberation,
        enabled: bool = True,
    ) -> None:
        super().__init__(sample_rate, enabled)
        self.aec = aec
        self.denoise = denoise
        self.deverb = deverb
        self.sub_stages: dict[str, Processor] = {
            "aec": self.aec,
            "denoise": self.denoise,
            "deverb": self.deverb,
        }

        # Determine processing grid sample rate
        # CoFi-Lite neural model natively operates on 16 kHz grid (512 FFT = 257 bins)
        if getattr(self.denoise, "model_sr", None) == 16000:
            self.grid_sr = 16000
        else:
            self.grid_sr = sample_rate

        self.n_fft = 512
        self.hop = 256
        self.n_bins = self.n_fft // 2 + 1  # 257

        if hasattr(self.deverb, "set_hop"):
            self.deverb.set_hop(self.hop)

        # Sqrt-Hann analysis and synthesis window
        self.win = np.hanning(self.n_fft).astype(np.float32) ** 0.5
        self._scale = float(self.hop / np.sum(self.win**2))

        # Timing instrumentation for internal sub-stages
        self._sub_timings_ns: dict[str, int] = {"aec": 0, "denoise": 0, "deverb": 0}

        # Design stateful polyphase FIR anti-aliasing filters for 48k <-> 16k resampling
        if self.sample_rate == 48000 and self.grid_sr == 16000:
            self._resample_ratio = 3
            n = np.arange(31) - 15.0
            h = (1.0 / 3.0) * np.sinc((1.0 / 3.0) * n)
            w = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(31) / 30.0)
            hw = h * w
            self._fir_h = (hw / np.sum(hw)).astype(np.float32)
            self._fir_h_up = (3.0 * self._fir_h).astype(np.float32)
            self._ext_down = np.zeros(0, dtype=np.float32)
            self._ext_up = np.zeros(0, dtype=np.float32)
            self._up_zero = np.zeros(480, dtype=np.float32)
        else:
            self._resample_ratio = 1
            self._fir_h = None
            self._fir_h_up = None
            self._ext_down = np.zeros(0, dtype=np.float32)
            self._ext_up = np.zeros(0, dtype=np.float32)
            self._up_zero = np.zeros(0, dtype=np.float32)

        # Pre-allocated circular buffers
        self._in_buf = np.zeros(4096, dtype=np.float32)
        self._in_len = 0
        self._out_buf = np.zeros(4096, dtype=np.float32)
        self._out_len = 0
        self._ola = np.zeros(self.n_fft, dtype=np.float32)
        self._shift_scratch = np.zeros(4096, dtype=np.float32)

        if self._fir_h is not None:
            self._zi_down = np.zeros(len(self._fir_h) - 1, dtype=np.float32)
            self._zi_up = np.zeros(len(self._fir_h) - 1, dtype=np.float32)
        else:
            self._zi_down = None
            self._zi_up = None

    @property
    def latency_samples(self) -> int:
        if not (self.enabled and (self.aec.enabled or self.denoise.enabled or self.deverb.enabled)):
            return 0
        if self.sample_rate == 48000 and self.grid_sr == 16000:
            return 1470  # CoFi-Lite STFT group delay (490 @ 16k * 3)
        return self.n_fft

    @property
    def speech_prob(self) -> float | None:
        if not (self.enabled and getattr(self.denoise, "enabled", False)):
            return None
        return float(getattr(self.denoise, "speech_prob", 0.0))

    @property
    def is_speech(self) -> bool:
        return bool(getattr(self.denoise, "is_speech", False))

    def reset(self) -> None:
        self.aec.reset()
        self.denoise.reset()
        self.deverb.reset()
        self._in_buf.fill(0.0)
        self._in_len = 0
        self._out_buf.fill(0.0)
        self._out_len = 0
        self._ola.fill(0.0)
        if self._zi_down is not None:
            self._zi_down.fill(0.0)
        if self._zi_up is not None:
            self._zi_up.fill(0.0)

    def get_sub_timing_ns(self, name: str) -> int:
        return self._sub_timings_ns.get(name, 0)

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x

        # If all internal stages are disabled, pass through directly
        any_enabled = self.aec.enabled or self.denoise.enabled or self.deverb.enabled
        if not any_enabled:
            return x

        self._sub_timings_ns["aec"] = 0
        self._sub_timings_ns["denoise"] = 0
        self._sub_timings_ns["deverb"] = 0

        n_in = x.shape[0]

        # 1. Resample to grid sample rate (48 kHz -> 16 kHz if applicable)
        if self._resample_ratio == 3:
            ext_len = len(self._zi_down) + len(x)
            if self._ext_down.shape[0] != ext_len:
                self._ext_down = np.empty(ext_len, dtype=np.float32)
            self._ext_down[: len(self._zi_down)] = self._zi_down
            self._ext_down[len(self._zi_down) :] = x
            down = np.convolve(self._ext_down, self._fir_h, mode="valid")[: len(x)]
            self._zi_down[:] = self._ext_down[-len(self._zi_down) :]
            x_grid = down[::3].astype(np.float32, copy=False)
        else:
            x_grid = x.astype(np.float32, copy=False)

        n_grid = x_grid.shape[0]

        # Append to input streaming buffer
        if self._in_len + n_grid > self._in_buf.shape[0]:
            new_in = np.zeros(max(self._in_buf.shape[0] * 2, self._in_len + n_grid + 1024), dtype=np.float32)
            new_in[: self._in_len] = self._in_buf[: self._in_len]
            self._in_buf = new_in

        self._in_buf[self._in_len : self._in_len + n_grid] = x_grid
        self._in_len += n_grid

        # 2. Process STFT frames in a single unified spectral pass
        while self._in_len >= self.n_fft:
            frame = self._in_buf[: self.n_fft].copy()

            # Safe shift
            remain = self._in_len - self.hop
            if self._shift_scratch.shape[0] < remain:
                self._shift_scratch = np.zeros(remain + 1024, dtype=np.float32)
            self._shift_scratch[:remain] = self._in_buf[self.hop : self._in_len]
            self._in_buf[:remain] = self._shift_scratch[:remain]
            self._in_len = remain

            # Forward STFT (only once!)
            spec = np.fft.rfft(frame * self.win, n=self.n_fft)

            # Frequency-domain stage 1: Acoustic Echo Cancellation & Residual Suppression
            if self.aec.enabled:
                t0 = time.perf_counter_ns()
                spec = self.aec.process_spectrum(spec)
                self._sub_timings_ns["aec"] += (time.perf_counter_ns() - t0)

            # Frequency-domain stage 2: Neural / Spectral Denoising + Mask VAD + Naturalness
            if self.denoise.enabled:
                t0 = time.perf_counter_ns()
                spec = self.denoise.process_spectrum(spec)
                self._sub_timings_ns["denoise"] += (time.perf_counter_ns() - t0)

            # Frequency-domain stage 3: Room Dereverberation
            if self.deverb.enabled:
                t0 = time.perf_counter_ns()
                spec = self.deverb.process_spectrum(spec)
                self._sub_timings_ns["deverb"] += (time.perf_counter_ns() - t0)

            # Inverse STFT (only once!)
            out_frame = np.fft.irfft(spec, n=self.n_fft).real * self.win * self._scale
            self._ola += out_frame

            # Append to output buffer
            if self._out_len + self.hop > self._out_buf.shape[0]:
                new_out = np.zeros(max(self._out_buf.shape[0] * 2, self._out_len + self.hop + 1024), dtype=np.float32)
                new_out[: self._out_len] = self._out_buf[: self._out_len]
                self._out_buf = new_out

            self._out_buf[self._out_len : self._out_len + self.hop] = self._ola[: self.hop]
            self._out_len += self.hop

            # Shift OLA
            self._ola[: self.n_fft - self.hop] = self._ola[self.hop :].copy()
            self._ola[self.n_fft - self.hop :] = 0.0

        # 3. Pull required samples at grid rate
        if self._out_len >= n_grid:
            chunk_grid = self._out_buf[:n_grid].copy()
            out_remain = self._out_len - n_grid
            if out_remain > 0:
                if self._shift_scratch.shape[0] < out_remain:
                    self._shift_scratch = np.zeros(out_remain + 1024, dtype=np.float32)
                self._shift_scratch[:out_remain] = self._out_buf[n_grid : self._out_len]
                self._out_buf[:out_remain] = self._shift_scratch[:out_remain]
            self._out_len = out_remain
        else:
            chunk_grid = np.zeros(n_grid, dtype=np.float32)

        # 4. Upsample back to target sample rate (16 kHz -> 48 kHz if applicable)
        if self._resample_ratio == 3:
            if self._up_zero.shape[0] != n_grid * 3:
                self._up_zero = np.zeros(n_grid * 3, dtype=np.float32)
            else:
                self._up_zero.fill(0)
            self._up_zero[::3] = chunk_grid
            ext_up_len = len(self._zi_up) + len(self._up_zero)
            if self._ext_up.shape[0] != ext_up_len:
                self._ext_up = np.empty(ext_up_len, dtype=np.float32)
            self._ext_up[: len(self._zi_up)] = self._zi_up
            self._ext_up[len(self._zi_up) :] = self._up_zero
            out_48k = np.convolve(self._ext_up, self._fir_h_up, mode="valid")[: len(self._up_zero)]
            self._zi_up[:] = self._ext_up[-len(self._zi_up) :]
            return out_48k[:n_in].astype(np.float32, copy=False)

        return chunk_grid[:n_in]

    def telemetry(self) -> dict[str, Any]:
        return {
            "speech_prob": self.speech_prob,
            "is_speech": 1.0 if self.is_speech else 0.0,
        }
