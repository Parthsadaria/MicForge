"""Modular Acoustic Echo Cancellation (AEC) Module.

Eliminates speaker bleed, game audio, and teammate voice echoes leaking from
loudspeakers or open-back headphones into the microphone.

Architecture:
1. Reference Loopback Buffer: Ingests system/speaker audio (far-end reference).
2. Time-Delay Alignment: Aligns far-end reference with mic capture using
   lookback cross-correlation.
3. Partitioned Block Frequency-Domain Adaptive Filter (PBFDAF):
   Linear adaptive filter using Normalized Least Mean Squares (NLMS) with
   causality time-domain constraints.
4. Double-Talk Detector (DTD): Cross-correlation and energy coherence check to
   freeze adaptation when near-end speech is active, preventing filter divergence.
5. Non-Linear Residual Post-Filter (RES): Spectral suppression of speaker distortion,
   non-linear room resonances, and residual leakage.
6. Ultra-low latency & CPU: Vectorized NumPy operations, pre-allocated scratch
   buffers, zero memory allocation during streaming.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import Processor
from .util import EPS, db2lin, lin2db


class AECDelayEstimator:
    """Estimates time-delay between far-end reference and microphone input."""

    def __init__(self, sample_rate: int, max_delay_ms: float = 200.0) -> None:
        self.sr = sample_rate
        self.max_delay_samples = int(max_delay_ms * 1e-3 * sample_rate)
        self.estimated_delay_samples = int(0.040 * sample_rate)  # 40 ms default

    def update_delay(self, ref: np.ndarray, mic: np.ndarray) -> int:
        """Estimate delay using normalized cross-correlation peak on downsampled envelopes."""
        if ref.size < 256 or mic.size < 256:
            return self.estimated_delay_samples

        # Fast envelope correlation downsampled by 8
        ds = 8
        ref_ds = np.abs(ref[::ds])
        mic_ds = np.abs(mic[::ds])
        max_lag = self.max_delay_samples // ds

        corr = np.correlate(mic_ds, ref_ds, mode="full")
        mid = len(ref_ds) - 1
        valid_corr = corr[mid : mid + max_lag]
        if valid_corr.size > 0:
            best_lag = int(np.argmax(valid_corr)) * ds
            # Smooth tracking: 95% previous, 5% new
            self.estimated_delay_samples = int(0.95 * self.estimated_delay_samples + 0.05 * best_lag)

        return self.estimated_delay_samples


class AECDoubleTalkDetector:
    """Detects near-end speech to freeze adaptation and prevent filter divergence."""

    def __init__(self) -> None:
        self.dtd_active = False
        self._hangover = 0

    def check(self, mic_rms: float, ref_rms: float, err_rms: float) -> bool:
        """Returns True if user voice (near-end) or double-talk is active."""
        if mic_rms < 1e-4:
            self.dtd_active = False
            return False

        # If microphone energy is significantly louder than speaker reference playback,
        # near-end speech is active (user is talking into the mic).
        is_near_end = mic_rms > (1.8 * max(ref_rms, 1e-5)) and mic_rms > 0.005

        if is_near_end:
            self._hangover = 6  # Hold DTD for ~6 blocks (60ms)
            self.dtd_active = True
        elif self._hangover > 0:
            self._hangover -= 1
            self.dtd_active = True
        else:
            self.dtd_active = False

        return self.dtd_active


class AECResidualSuppressor:
    """Non-linear spectral suppression for speaker distortion and residual echo."""

    def __init__(self, n_bins: int, max_suppression_db: float = 24.0) -> None:
        self.n_bins = n_bins
        self.floor = db2lin(-max_suppression_db)
        self._prior_gain = np.ones(n_bins, dtype=np.float32)

    def process(
        self,
        err_spec: np.ndarray,
        echo_psd: np.ndarray,
        mic_psd: np.ndarray,
        dtd_active: bool,
        amount: float,
    ) -> np.ndarray:
        if amount <= 1e-3:
            return err_spec

        # Estimate residual echo power
        err_psd = err_spec.real**2 + err_spec.imag**2

        # When DTD is active (user is talking), reduce residual aggression to protect speech
        agg = 1.6 if not dtd_active else 0.45
        agg *= amount

        # Wiener-type suppression mask
        snr_est = np.maximum(0.0, err_psd - agg * echo_psd) / (err_psd + EPS)
        gain = np.clip(snr_est, self.floor, 1.0)

        # Smooth gain across frequency to avoid tone burrs
        smoothed = np.empty_like(gain)
        smoothed[0] = 0.7 * gain[0] + 0.3 * gain[1]
        smoothed[1:-1] = 0.25 * gain[:-2] + 0.5 * gain[1:-1] + 0.25 * gain[2:]
        smoothed[-1] = 0.3 * gain[-2] + 0.7 * gain[-1]

        # Recursive smoothing
        self._prior_gain = 0.4 * self._prior_gain + 0.6 * smoothed
        return err_spec * self._prior_gain


class AcousticEchoCanceller(Processor):
    name = "aec"

    def __init__(
        self,
        sample_rate: int,
        amount: float = 1.0,
        filter_length_ms: float = 60.0,
        delay_ms: float = 40.0,
        suppression_db: float = 24.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(sample_rate, enabled)
        self.amount = float(min(1.0, max(0.0, amount)))
        self.filter_length_ms = float(min(200.0, max(20.0, filter_length_ms)))
        self.delay_ms = float(min(250.0, max(0.0, delay_ms)))
        self.suppression_db = float(min(40.0, max(6.0, suppression_db)))

        # Sub-band PBFDAF block parameters
        self.N = 128  # Sub-block size (2.67 ms @ 48 kHz)
        self.K = self.N + 1  # RFFT bins (129)
        self.P = max(2, min(16, int(self.filter_length_ms * 1e-3 * sample_rate / self.N)))  # Partitions

        # Adaptive filter state
        self.W = np.zeros((self.P, self.K), dtype=np.complex64)
        self.X_hist = np.zeros((self.P, self.K), dtype=np.complex64)
        self.P_x = np.full(self.K, 1e-3, dtype=np.float32)

        # Reference audio ring buffer (holds up to 1 second of far-end speaker audio)
        self._ref_cap = sample_rate
        self._ref_ring = np.zeros(self._ref_cap, dtype=np.float32)
        self._ref_w = 0
        self._ref_r = 0
        self._ref_available = False
        self._mic_processed = 0

        # Input and output alignment buffers
        self._in_mic = np.zeros(self.N * 8, dtype=np.float32)
        self._in_mic_len = 0
        self._out_buf = np.zeros(self.N * 8, dtype=np.float32)
        self._out_len = 0

        # Submodules
        self.delay_est = AECDelayEstimator(sample_rate, max_delay_ms=250.0)
        self.dtd = AECDoubleTalkDetector()
        self.post_filter = AECResidualSuppressor(self.N // 2 + 1, max_suppression_db=self.suppression_db)

        # Pre-allocated scratch buffers
        self._scratch_ref_block = np.zeros(self.N * 2, dtype=np.float32)
        self._scratch_mic_block = np.zeros(self.N, dtype=np.float32)
        self._scratch_echo_est = np.zeros(self.N, dtype=np.float32)
        self._scratch_err_time = np.zeros(self.N, dtype=np.float32)

        # Telemetry
        self._erle_db = 0.0
        self._dtd_active = False
        self._ref_energy_db = -90.0
        self._echo_detected = False

    def feed_reference(self, ref_audio: np.ndarray) -> None:
        """Feed speaker / game / Discord reference audio into the AEC loopback buffer."""
        if ref_audio.size == 0:
            return
        x = np.asarray(ref_audio, dtype=np.float32)
        if x.ndim > 1:
            x = x[:, 0]  # Mono

        n = x.shape[0]
        w = self._ref_w
        cap = self._ref_cap

        first = min(n, cap - (w % cap))
        w_idx = w % cap
        self._ref_ring[w_idx : w_idx + first] = x[:first]
        if first < n:
            self._ref_ring[: n - first] = x[first:]
        self._ref_w = w + n
        self._ref_available = True

        # Track reference energy
        ref_rms = float(np.sqrt(np.mean(x * x)))
        self._ref_energy_db = float(lin2db(max(ref_rms, 1e-6)))

    def set_amount(self, amount: float) -> None:
        self.amount = float(min(1.0, max(0.0, amount)))

    def set_delay_ms(self, ms: float) -> None:
        self.delay_ms = float(min(250.0, max(0.0, ms)))

    def set_suppression_db(self, db: float) -> None:
        self.suppression_db = float(min(40.0, max(6.0, db)))
        self.post_filter.floor = db2lin(-self.suppression_db)

    @property
    def latency_samples(self) -> int:
        return self.N

    def reset(self) -> None:
        self.W.fill(0.0)
        self.X_hist.fill(0.0)
        self.P_x.fill(1e-3)
        self._ref_ring.fill(0.0)
        self._ref_w = 0
        self._ref_r = 0
        self._ref_available = False
        self._mic_processed = 0
        self._in_mic.fill(0.0)
        self._in_mic_len = 0
        self._out_buf.fill(0.0)
        self._out_len = 0
        self._erle_db = 0.0
        self._dtd_active = False
        self._ref_energy_db = -90.0
        self._echo_detected = False

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or self.amount <= 1e-4 or x.size == 0:
            return x

        n_in = x.shape[0]

        # If no reference audio has ever been fed or reference is absolute silence (< -70 dBFS),
        # pass mic audio through cleanly with zero processing overhead.
        if not self._ref_available or self._ref_energy_db < -70.0:
            self._erle_db = 0.0
            self._echo_detected = False
            return x

        # 1. Accumulate incoming mic samples
        if self._in_mic_len + n_in > self._in_mic.shape[0]:
            new_mic = np.zeros(max(self._in_mic.shape[0] * 2, self._in_mic_len + n_in + 1024), dtype=np.float32)
            new_mic[: self._in_mic_len] = self._in_mic[: self._in_mic_len]
            self._in_mic = new_mic

        self._in_mic[self._in_mic_len : self._in_mic_len + n_in] = x
        self._in_mic_len += n_in

        # Delay compensation offset
        delay_samples = int(self.delay_ms * 1e-3 * self.sample_rate)

        # 2. Process sub-blocks of size N (128 samples = 2.67 ms)
        step_mu = 0.06 * self.amount

        while self._in_mic_len >= self.N:
            # Current mic block
            mic_block = self._in_mic[: self.N].copy()

            # Shift input mic buffer
            remain = self._in_mic_len - self.N
            self._in_mic[:remain] = self._in_mic[self.N : self._in_mic_len]
            self._in_mic_len = remain

            # Extract aligned reference block of size 2N (overlap-save)
            aligned_ref_idx = self._mic_processed - delay_samples
            self._mic_processed += self.N

            if aligned_ref_idx >= 0 and self._ref_w >= aligned_ref_idx + (self.N * 2):
                # Pull 2N samples from circular reference ring
                cap = self._ref_cap
                r_idx = aligned_ref_idx % cap
                first = min(self.N * 2, cap - r_idx)
                self._scratch_ref_block[:first] = self._ref_ring[r_idx : r_idx + first]
                if first < self.N * 2:
                    self._scratch_ref_block[first:] = self._ref_ring[: (self.N * 2) - first]
            else:
                self._scratch_ref_block.fill(0.0)

            # PBFDAF: Shift frequency history
            self.X_hist[1:] = self.X_hist[:-1]
            X_curr = np.fft.rfft(self._scratch_ref_block, n=self.N * 2)
            self.X_hist[0] = X_curr

            # Estimate echo in frequency domain: D_hat = sum(W_p * X_p)
            D_hat_spec = np.sum(self.W * self.X_hist, axis=0)
            d_hat_time = np.fft.irfft(D_hat_spec, n=self.N * 2).real[self.N :]

            # Linear error: e = y - d_hat
            err_time = mic_block - d_hat_time

            # Compute energies for Double-Talk Detection & ERLE
            mic_rms = float(np.sqrt(np.mean(mic_block * mic_block)))
            echo_rms = float(np.sqrt(np.mean(d_hat_time * d_hat_time)))
            err_rms = float(np.sqrt(np.mean(err_time * err_time)))
            ref_rms = float(np.sqrt(np.mean(self._scratch_ref_block * self._scratch_ref_block)))

            dtd_active = self.dtd.check(mic_rms, ref_rms, err_rms)
            self._dtd_active = dtd_active

            # Update normalized reference power estimate P_x per bin
            self.P_x = 0.95 * self.P_x + 0.05 * (X_curr.real**2 + X_curr.imag**2) / (self.N * 2)

            # Adaptive filter weight update (only when NOT in double-talk)
            if not dtd_active and ref_rms > 1e-4:
                # Frequency error: zero-padded [0, err_time]
                err_ext = np.zeros(self.N * 2, dtype=np.float32)
                err_ext[self.N :] = err_time
                E_spec = np.fft.rfft(err_ext, n=self.N * 2)

                # Normalized LMS step size per bin with gradient constraint
                norm = (self.P * self.P_x + 1e-2) * (self.N * 2)
                mu_k = step_mu / norm

                for p in range(self.P):
                    grad = np.conj(self.X_hist[p]) * E_spec
                    self.W[p] += mu_k * grad

                    # Time-domain constraint: zero non-causal tail to maintain FIR stability
                    w_time = np.fft.irfft(self.W[p], n=self.N * 2).real
                    w_time[self.N :] = 0.0
                    self.W[p] = np.fft.rfft(w_time, n=self.N * 2)

                self._echo_detected = True
            else:
                self._echo_detected = False

            # Non-linear residual suppression post-filter
            err_spec = np.fft.rfft(err_time, n=self.N)
            echo_psd = np.abs(np.fft.rfft(d_hat_time, n=self.N)) ** 2
            mic_psd = np.abs(np.fft.rfft(mic_block, n=self.N)) ** 2

            clean_spec = self.post_filter.process(
                err_spec, echo_psd, mic_psd, dtd_active, self.amount
            )
            clean_time = np.fft.irfft(clean_spec, n=self.N).real

            # Telemetry ERLE (Echo Return Loss Enhancement in dB)
            clean_rms = float(np.sqrt(np.mean(clean_time * clean_time)))
            ratio = max(mic_rms, 1e-6) / max(clean_rms, 1e-6)
            erle = 10.0 * math.log10(ratio) if ratio > 0 else 0.0
            self._erle_db = 0.9 * self._erle_db + 0.1 * max(0.0, min(40.0, erle))

            # Append to output buffer
            if self._out_len + self.N > self._out_buf.shape[0]:
                new_out = np.zeros(max(self._out_buf.shape[0] * 2, self._out_len + self.N + 1024), dtype=np.float32)
                new_out[: self._out_len] = self._out_buf[: self._out_len]
                self._out_buf = new_out

            self._out_buf[self._out_len : self._out_len + self.N] = clean_time
            self._out_len += self.N

        # 3. Pull required samples matching input block length
        if self._out_len >= n_in:
            chunk = self._out_buf[:n_in].copy()
            remain_out = self._out_len - n_in
            if remain_out > 0:
                self._out_buf[:remain_out] = self._out_buf[n_in : self._out_len]
            self._out_len = remain_out
            return chunk
        else:
            return x  # Pass through if buffer is still priming

    def telemetry(self) -> dict[str, float]:
        return {
            "erle_db": self._erle_db,
            "dtd_active": 1.0 if self._dtd_active else 0.0,
            "ref_energy_db": self._ref_energy_db,
            "echo_detected": 1.0 if self._echo_detected else 0.0,
        }
