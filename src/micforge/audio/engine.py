"""Real-time engine: mic -> DSP chain -> one or more output sinks (virtual cable, headphones).

Threading model
---------------
* The input callback (PortAudio thread) does all DSP, so there is exactly one hop of buffering.
* Each sink owns an output stream whose callback pulls from a ring buffer, with 1-sample
  drift correction because input/output devices run on independent clocks.
* The UI thread only reads `snapshot()` (immutable) and flips simple flags.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..dsp.chain import Chain
from ..dsp.denoise import SpectralDenoiser
from ..dsp.util import db2lin, rms_db
from ..errors import DeviceError
from .backend import get_sd
from .cues import CuePlayer
from .devices import DeviceInfo, extra_settings
from .metrics import LevelMeter, LevelSnapshot, SpectrumAnalyzer
from .ring import RingBuffer

log = logging.getLogger("micforge.engine")


@dataclass
class SinkSpec:
    name: str  # "cable" | "monitor"
    device: DeviceInfo
    enabled: bool = True
    gain_db: float = 0.0


@dataclass(frozen=True)
class CalibrationResult:
    psd: np.ndarray | None
    noise_db: float
    gate_threshold_db: float
    n_fft: int
    sample_rate: int


@dataclass(frozen=True)
class EngineSnapshot:
    running: bool
    in_level: LevelSnapshot
    out_level: LevelSnapshot
    in_bands: np.ndarray
    out_bands: np.ndarray
    telemetry: dict[str, dict[str, float]]
    cpu_load: float
    cpu_peak: float
    input_overflows: int
    output_underruns: int
    drift_corrections: int
    errors: int
    latency_ms: float
    bypass: bool
    muted: bool
    monitor_on: bool
    monitor_available: bool
    calibrating: float | None
    input_gain_db: float
    sinks: dict[str, bool] = field(default_factory=dict)


class OutputSink:
    def __init__(self, spec: SinkSpec, sample_rate: int, block_size: int, prefill_blocks: float | int) -> None:
        self.spec = spec
        self.sr = sample_rate
        self.block = block_size
        self.base_prefill = int(prefill_blocks * block_size)
        self.prefill = self.base_prefill
        self.max_prefill = int(max(3.5, float(prefill_blocks) * 2.0) * block_size)
        self.ring = RingBuffer(sample_rate)  # 1 s of headroom
        self.enabled = spec.enabled
        self.gain = db2lin(spec.gain_db)
        self._gain_is_unity = abs(self.gain - 1.0) < 1e-6
        self.underruns = 0
        self.corrections = 0
        self.stream: Any = None
        self.channels = max(1, min(2, spec.device.max_output))
        self._avg_fill = float(self.prefill)
        self._drift_cooldown = 0
        self._active = False
        self._consecutive_underruns = 0
        self._last_underrun_time = 0.0
        self._needs_fade_in = False
        self._fade_samples = min(32, block_size // 4)
        self._stable_blocks = 0
        # Pre-allocated read buffer to avoid allocation in the output callback
        self._read_buf = np.zeros(block_size + 2, dtype=np.float32)
        # Pre-allocated silence cushion to avoid allocation in underrun path
        self._silence = np.zeros(self.max_prefill, dtype=np.float32)
        # Track the last sample output for continuity checking
        self._last_sample = np.float32(0.0)

    def open(self) -> None:
        sd = get_sd()
        try:
            self.stream = sd.OutputStream(
                device=self.spec.device.index,
                samplerate=self.sr,
                blocksize=self.block,
                channels=self.channels,
                dtype="float32",
                latency="low",
                callback=self._callback,
                extra_settings=extra_settings(self.spec.device, "output"),
            )
        except Exception as exc:  # sd.PortAudioError and friends
            raise DeviceError(
                f"Cannot open output '{self.spec.device.name}' at {self.sr} Hz: {exc}",
                hint="Try another device, or set sample_rate to 44100 in the config.",
            ) from exc
        self._prime()

    def _prime(self) -> None:
        self.ring.clear()
        self.prefill = self.base_prefill
        self._avg_fill = float(self.prefill + 0.5 * self.block)
        self._drift_cooldown = 0
        self._consecutive_underruns = 0
        self._last_underrun_time = 0.0
        self._needs_fade_in = False
        self._stable_blocks = 0
        self._active = False
        self._last_sample = np.float32(0.0)

    def start(self) -> None:
        self.stream.start()

    def close(self) -> None:
        self._active = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001 - closing must never raise
                log.debug("sink close failed", exc_info=True)
            self.stream = None

    def set_enabled(self, on: bool) -> None:
        if on and not self.enabled:
            self._prime()
        self.enabled = on

    def notify_pipeline_change(self) -> None:
        """Reset drift detector cooldown and alignment on preset or model swap.

        Prevents the drift detector from misinterpreting a brief DSP swap
        pause as a physical hardware clock drift, stopping false clock fixes.
        """
        self._avg_fill = float(self.prefill + 0.5 * self.block)
        self._drift_cooldown = 0
        self._consecutive_underruns = 0
        self._stable_blocks = 0

    def push(self, x: np.ndarray) -> None:
        if self.enabled:
            if not self._active:
                # Arm the output sink with jitter cushion on the first incoming audio block
                self.ring.clear()
                self.ring.write(self._silence[:self.prefill])
                self._avg_fill = float(self.prefill + 0.5 * self.block)
                self._drift_cooldown = 0
                self.underruns = 0
                self._active = True
            # Avoid allocation when gain is unity (0 dB) — very common case
            if self._gain_is_unity:
                self.ring.write(x)
            else:
                self.ring.write(x * np.float32(self.gain))

    @property
    def latency_s(self) -> float:
        try:
            return float(self.stream.latency) if self.stream is not None else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _callback(self, outdata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        if not self.enabled or not self._active:
            outdata.fill(0)
            return

        fill = self.ring.fill
        self._avg_fill = 0.99 * self._avg_fill + 0.01 * fill
        self._drift_cooldown += 1

        want = frames
        target_fill = self.prefill + 0.5 * frames
        deadband = int(0.4 * frames)

        # Clock drift: continuous low-pass drift tracking with 200 ms cooldown (20 blocks @ 10 ms).
        # Nudges read length by 1 sample only on sustained real drift (>0.4 blocks deviation).
        # Allows smoothly correcting up to 5 samples/sec (104 ppm drift), easily matching any PC hardware.
        if self._drift_cooldown >= 20:
            if self._avg_fill > target_fill + deadband:
                want = frames + 1
                self._drift_cooldown = 0
            elif self._avg_fill < target_fill - deadband and fill >= frames:
                want = frames - 1
                self._drift_cooldown = 0

        # Ensure pre-allocated read buffer is large enough
        if self._read_buf.shape[0] < want:
            self._read_buf = np.zeros(want + 2, dtype=np.float32)
        else:
            self._read_buf[:want] = 0.0

        got = self.ring.read_into(self._read_buf[:want])

        # CRITICAL FIX: An underrun ONLY occurs if we got fewer samples than we requested (got < want).
        # Previously, checking `got < frames` when `want == frames - 1` misidentified intentional
        # 1-sample drift corrections as underruns, triggering a cascading starvation loop!
        if got < want:
            self.underruns += 1
            now = time.monotonic()

            # 1. Pop concealment: crossfade from last known sample to silence
            if got > 0:
                flen = min(got, self._fade_samples)
                fade = np.arange(flen, 0, -1, dtype=np.float32) / flen
                self._read_buf[got - flen : got] *= fade

            # 2. Adaptive cushion expansion on repeated underruns (auto-adapts for low-end PCs)
            if now - self._last_underrun_time < 4.0:
                self._consecutive_underruns += 1
                if self._consecutive_underruns >= 2 and self.prefill < self.max_prefill:
                    step = int(0.5 * frames)
                    self.prefill = min(self.max_prefill, self.prefill + step)
                    log.debug("Adaptive buffer expanded to %d samples for '%s'", self.prefill, self.spec.name)
            else:
                self._consecutive_underruns = 1
            self._last_underrun_time = now
            self._stable_blocks = 0

            # 3. Re-arm cushion if buffer starved to prevent runaway starvation
            if fill == 0 or got == 0:
                self.ring.clear()
                self.ring.write(self._silence[:self.prefill])
                self._avg_fill = float(target_fill)
            else:
                self._avg_fill = float(self.ring.fill)
            self._drift_cooldown = 0
            self._needs_fade_in = True
        else:
            self._stable_blocks += 1
            # Gentle relaxation towards base prefill after sustained stability (3000 blocks ~ 30 s)
            if self._stable_blocks > 3000 and self.prefill > self.base_prefill:
                self.prefill = max(self.base_prefill, self.prefill - 1)
                self._stable_blocks = 2000

        # Smooth resume fade-in when recovering from an underrun (prevents leading edge click)
        if self._needs_fade_in and got >= want:
            flen = min(got, self._fade_samples)
            fade = np.arange(1, flen + 1, dtype=np.float32) / flen
            self._read_buf[:flen] *= fade
            self._needs_fade_in = False

        # Write to output: fractional linear interpolation resampling for clock drift
        if want == frames + 1 and got >= frames + 1:
            # Squeeze frames+1 samples into frames output samples via linear interpolation
            src = self._read_buf[:frames + 1]
            indices = np.linspace(0, frames, frames, endpoint=False, dtype=np.float32)
            idx_int = indices.astype(np.int32)
            frac = indices - idx_int
            outdata[:, 0] = src[idx_int] * (1.0 - frac) + src[np.minimum(idx_int + 1, frames)] * frac
            self.corrections += 1
        elif want == frames - 1 and got >= frames - 1:
            # Stretch frames-1 samples into frames output samples via linear interpolation
            src = self._read_buf[:frames - 1]
            indices = np.linspace(0, frames - 2, frames, endpoint=True, dtype=np.float32)
            idx_int = np.minimum(indices.astype(np.int32), frames - 2)
            frac = indices - idx_int
            outdata[:, 0] = src[idx_int] * (1.0 - frac) + src[np.minimum(idx_int + 1, frames - 2)] * frac
            self.corrections += 1
        else:
            outdata[:frames, 0] = self._read_buf[:frames]

        # Track last output sample for continuity
        self._last_sample = outdata[frames - 1, 0]

        if self.channels > 1:
            outdata[:, 1:] = outdata[:, :1]

        # Track last output sample for continuity
        self._last_sample = outdata[frames - 1, 0]

        if self.channels > 1:
            outdata[:, 1:] = outdata[:, :1]


class AudioEngine:
    def __init__(
        self,
        *,
        chain: Chain,
        input_device: DeviceInfo,
        sinks: list[SinkSpec],
        input_channel: int = 0,
        sample_rate: int = 48000,
        block_size: int = 480,
        jitter_blocks: float | int = 1.5,
        play_cues: bool = True,
        cue_volume: float = 0.5,
    ) -> None:
        self.chain = chain
        self.input_device = input_device
        self.input_channel = input_channel
        self.sr = sample_rate
        self.block = block_size
        self.sinks = {s.name: OutputSink(s, sample_rate, block_size, jitter_blocks) for s in sinks}
        self.jitter_blocks = jitter_blocks
        self.cues = CuePlayer(sample_rate=sample_rate, volume=cue_volume, enabled=play_cues)

        self.bypass = False
        self.muted = False
        self._in_meter = LevelMeter(sample_rate)
        self._out_meter = LevelMeter(sample_rate)
        self._in_spec = SpectrumAnalyzer(sample_rate)
        self._out_spec = SpectrumAnalyzer(sample_rate)
        self._in_stream: Any = None
        self._running = False

        self._cpu = 0.0
        self._cpu_peak = 0.0
        self._overflows = 0
        self._errors = 0
        self._telemetry: dict[str, dict[str, float]] = {}
        self._tel_count = 0

        self._cal_need = 0
        self._cal_buf: list[np.ndarray] = []
        self._cal_have = 0
        self._cal_progress: float | None = None
        self._cal_result: CalibrationResult | None = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        sd = get_sd()
        dev = self.input_device
        channels = max(1, min(dev.max_input, self.input_channel + 1))
        if self.input_channel >= dev.max_input:
            raise DeviceError(
                f"Input channel {self.input_channel} does not exist on '{dev.name}' ({dev.max_input} channels)."
            )
        self._channels = channels
        try:
            self._in_stream = sd.InputStream(
                device=dev.index,
                samplerate=self.sr,
                blocksize=self.block,
                channels=channels,
                dtype="float32",
                latency="low",
                callback=self._on_input,
                extra_settings=extra_settings(dev, "input"),
            )
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(
                f"Cannot open microphone '{dev.name}' at {self.sr} Hz: {exc}",
                hint="Close apps using the mic in exclusive mode, or try sample_rate 44100.",
            ) from exc
        opened: list[OutputSink] = []
        try:
            for s in self.sinks.values():
                s.open()
                opened.append(s)
            self._running = True
            self._in_stream.start()
            for s in opened:
                s.start()
        except Exception:
            self._running = False
            for s in opened:
                s.close()
            if self._in_stream is not None:
                self._in_stream.close()
            raise

    def stop(self) -> None:
        self._running = False
        if self._in_stream is not None:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:  # noqa: BLE001
                log.debug("input close failed", exc_info=True)
            self._in_stream = None
        for s in self.sinks.values():
            s.close()

    def __enter__(self) -> "AudioEngine":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------ controls
    def set_chain(self, chain: Chain) -> None:
        self.chain = chain  # atomic swap; old chain is simply dropped
        for s in self.sinks.values():
            s.notify_pipeline_change()

    def toggle_bypass(self) -> bool:
        return self.set_bypass(not self.bypass)

    def set_bypass(self, bypass: bool) -> bool:
        if self.bypass != bypass:
            self.bypass = bypass
            self.cues.play("toggle")
            mon = self.sinks.get("monitor")
            if mon is None or not mon.enabled:
                self.cues.play_standalone("toggle")
        return self.bypass

    def toggle_mute(self) -> bool:
        return self.set_muted(not self.muted)

    def set_muted(self, muted: bool) -> bool:
        if self.muted != muted:
            self.muted = muted
            cue = "mute" if muted else "unmute"
            self.cues.play(cue)
            mon = self.sinks.get("monitor")
            if mon is None or not mon.enabled:
                self.cues.play_standalone(cue)
        return self.muted

    @property
    def monitor_available(self) -> bool:
        return "monitor" in self.sinks

    def toggle_monitor(self) -> bool:
        s = self.sinks.get("monitor")
        if s is None:
            return False
        s.set_enabled(not s.enabled)
        return s.enabled

    def begin_calibration(self, seconds: float = 3.0) -> None:
        with self._lock:
            self._cal_need = int(seconds * self.sr)
            self._cal_buf, self._cal_have = [], 0
            self._cal_progress = 0.0
            self._cal_result = None

    def pop_calibration(self) -> CalibrationResult | None:
        with self._lock:
            res, self._cal_result = self._cal_result, None
        return res

    # --------------------------------------------------------------- audio
    def _on_input(self, indata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        t0 = time.perf_counter()
        if status and getattr(status, "input_overflow", False):
            self._overflows += 1
        x = np.ascontiguousarray(indata[:, self.input_channel], dtype=np.float32)
        try:
            self._in_meter.update(x)
            self._in_spec.feed(x)
            self._collect_calibration(x)
            processed = self.chain.process(x)
            out = x if self.bypass else processed
            if self.muted:
                out = np.zeros_like(out)
            self._out_meter.update(out)
            self._out_spec.feed(out)
            for name, s in self.sinks.items():
                if name == "monitor":
                    s.push(self.cues.mix_into(out))
                else:
                    s.push(out)
            self._tel_count = (self._tel_count + 1) % 4
            if self._tel_count == 0:
                self._telemetry = self.chain.telemetry()
        except Exception:  # noqa: BLE001 - never kill the audio thread; fall back to pass-through
            self._errors += 1
            log.debug("DSP failure; passing audio through", exc_info=True)
            # Mute on error instead of switching to raw audio, which causes a
            # spectral/gain discontinuity heard as a loud click.  Missing 10 ms
            # of silence is far less objectionable than a pop.
            silence = np.zeros_like(x)
            for name, s in self.sinks.items():
                if name == "monitor":
                    s.push(self.cues.mix_into(silence))
                else:
                    s.push(silence)
        load = (time.perf_counter() - t0) / (frames / self.sr)
        self._cpu = 0.9 * self._cpu + 0.1 * load
        self._cpu_peak = max(load, self._cpu_peak * 0.995)

    def _collect_calibration(self, x: np.ndarray) -> None:
        if self._cal_progress is None or self._cal_need == 0:
            return
        self._cal_buf.append(x.copy())
        self._cal_have += x.shape[0]
        self._cal_progress = min(1.0, self._cal_have / self._cal_need)
        if self._cal_have >= self._cal_need:
            noise = np.concatenate(self._cal_buf)
            self._cal_need = 0
            self._cal_buf = []
            threading.Thread(target=self._finish_calibration, args=(noise,), daemon=True).start()

    def _finish_calibration(self, noise: np.ndarray) -> None:
        try:
            chain = self.chain
            dn = chain.stage("denoise")
            psd = dn.learn(noise) if hasattr(dn, "learn") else None
            noise_db = rms_db(noise)
            threshold = noise_db + chain.input_gain_db + 9.0
            gate = chain.stage("gate")
            if gate is not None:
                gate.threshold_db = threshold  # type: ignore[attr-defined]
            n_fft = getattr(dn, "n_fft", 0)
            result = CalibrationResult(psd, noise_db, threshold, n_fft, self.sr)
            with self._lock:
                self._cal_result = result
        except Exception:  # noqa: BLE001
            self._errors += 1
            log.exception("calibration failed")
        finally:
            self._cal_progress = None

    # ------------------------------------------------------------ snapshot
    def latency_ms(self) -> float:
        # Realistic hardware I/O latency: one hardware block period each for ADC capture
        # and DAC output, bounded by the driver-reported buffer.
        hw_in = self.block / self.sr
        if self._in_stream is not None:
            try:
                # Operational delay is ~half of PortAudio's double-buffer capacity
                hw_in = min(float(self._in_stream.latency) * 0.5, hw_in * 1.5)
            except Exception:  # noqa: BLE001
                pass
        hw_out = self.block / self.sr
        if self.sinks:
            try:
                max_out = max(s.latency_s for s in self.sinks.values())
                hw_out = min(max_out * 0.5, hw_out * 1.5)
            except Exception:  # noqa: BLE001
                pass
        buf = float(self.jitter_blocks) * self.block / self.sr
        dsp = self.chain.latency_samples / self.sr
        return (hw_in + hw_out + buf + dsp) * 1000.0

    def snapshot(self) -> EngineSnapshot:
        mon = self.sinks.get("monitor")
        return EngineSnapshot(
            running=self._running,
            in_level=self._in_meter.snapshot,
            out_level=self._out_meter.snapshot,
            in_bands=self._in_spec.bands.copy(),
            out_bands=self._out_spec.bands.copy(),
            telemetry=dict(self._telemetry),
            cpu_load=self._cpu,
            cpu_peak=self._cpu_peak,
            input_overflows=self._overflows,
            output_underruns=sum(s.underruns for s in self.sinks.values() if s.enabled),
            drift_corrections=sum(s.corrections for s in self.sinks.values()),
            errors=self._errors,
            latency_ms=self.latency_ms(),
            bypass=self.bypass,
            muted=self.muted,
            monitor_on=bool(mon and mon.enabled),
            monitor_available=mon is not None,
            calibrating=self._cal_progress,
            input_gain_db=self.chain.input_gain_db,
            sinks={n: s.enabled for n, s in self.sinks.items()},
        )

    @property
    def spectrum_centers(self) -> np.ndarray:
        return self._in_spec.band_centers()

    @property
    def spectrum_range(self) -> tuple[float, float]:
        return self._in_spec.f_lo, self._in_spec.f_hi
