"""Process-isolated audio engine: moves real-time DSP into an independent process.

Guarantees zero GIL contention from Rich terminal rendering, Tkinter GUI, or hotkey
hooks. Uses shared memory for audio waveforms, atomic shared array for telemetry and VU
meters, and a sub-millisecond IPC queue for instant controls with an auto-restarting watchdog.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import threading
import time
from typing import Any

import numpy as np

from .engine import AudioEngine, CalibrationResult, EngineSnapshot, LevelSnapshot, SinkSpec
from ..config import AppConfig
from ..errors import AudioBackendError
from ..presets import get_preset
from .cues import CuePlayer

log = logging.getLogger("micforge.isolated_engine")

# Indices in shared_metrics array
RUNNING = 0
IN_RMS = 1
IN_PEAK = 2
IN_CLIP = 3
OUT_RMS = 4
OUT_PEAK = 5
OUT_CLIP = 6
CPU_LOAD = 7
CPU_PEAK = 8
INPUT_OVERFLOWS = 9
OUTPUT_UNDERRUNS = 10
DRIFT_CORRECTIONS = 11
ERRORS = 12
LATENCY_MS = 13
BYPASS = 14
MUTED = 15
MONITOR_ON = 16
MONITOR_AVAILABLE = 17
CALIBRATING = 18
INPUT_GAIN_DB = 19
REF_RMS = 20
REF_PEAK = 21
RECOVERING = 22
SPEECH_PROB = 23
IS_SPEECH = 24
NAPPING = 25
ERLE_DB = 26
DTD_ACTIVE = 27
REVERB_RED_DB = 28
DENOISE_RED_DB = 29
NOISE_FLOOR_DB = 30
TOT_P50_US = 31
TOT_P99_US = 32
IN_BANDS_START = 33   # 32 bands (33..64)
OUT_BANDS_START = 65  # 32 bands (65..96)

# Stage reductions & flags (97..115)
GATE_ATTEN_DB = 97
COMP_RED_DB = 98
LIM_RED_DB = 99
DEESS_RED_DB = 100
DN_P99_US = 101
AEC_ENABLED = 102
AI_ENABLED = 103
VERB_ENABLED = 104
GATE_ENABLED = 105
PROX_ENABLED = 106
DEESS_ENABLED = 107
COMP_ENABLED = 108
LIM_ENABLED = 109
LVL_ENABLED = 110
PROX_BOOST_DB = 111
GATE_OPEN = 112
IS_NEURAL_MODEL = 113
METRICS_SIZE = 128


class IsolatedOutputSink:
    """Proxy for an output sink to satisfy interface expectations in the UI process."""

    def __init__(self, spec: SinkSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.device = spec.device
        self.enabled = spec.enabled
        self.gain_db = spec.gain_db

    def notify_pipeline_change(self) -> None:
        pass


class StageProxy:
    """Wraps a DSP stage so that in-place mutations immediately sync to the worker process."""

    def __init__(self, stage: Any, stage_name: str, cmd_queue: mp.Queue) -> None:
        object.__setattr__(self, "_stage", stage)
        object.__setattr__(self, "_stage_name", stage_name)
        object.__setattr__(self, "_cmd_queue", cmd_queue)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stage, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_stage", "_stage_name", "_cmd_queue"):
            object.__setattr__(self, name, value)
            return
        setattr(self._stage, name, value)
        try:
            self._cmd_queue.put(("set_stage_attr", self._stage_name, name, value))
        except Exception:
            pass

    def set_strength(self, val: float) -> None:
        if hasattr(self._stage, "set_strength"):
            self._stage.set_strength(val)
        try:
            self._cmd_queue.put(("set_stage_attr", self._stage_name, "strength", float(val)))
        except Exception:
            pass


class ChainProxy:
    """Wraps the DSP chain so stage lookups and pipeline alterations sync to the worker."""

    def __init__(self, chain: Any, cmd_queue: mp.Queue, shared_metrics: Any | None = None) -> None:
        self._chain = chain
        self._cmd_queue = cmd_queue
        self._shared_metrics = shared_metrics

    def stage(self, name: str) -> Any:
        st = self._chain.stage(name)
        if st is None:
            return None
        return StageProxy(st, name, self._cmd_queue)

    def set_input_gain_db(self, db: float) -> None:
        self._chain.set_input_gain_db(db)
        try:
            self._cmd_queue.put(("set_input_gain", float(db)))
        except Exception:
            pass

    def replace_stage(self, name: str, new_stage: Any) -> None:
        self._chain.replace_stage(name, new_stage)
        if name == "denoise":
            actual = getattr(new_stage, "_stage", new_stage)
            from ..dsp.cofilite import CoFiLiteDenoiser
            model_type = "cofilite" if isinstance(actual, CoFiLiteDenoiser) or getattr(actual, "model_type", "") == "cofilite" else "spectral"
            if self._shared_metrics is not None:
                self._shared_metrics[IS_NEURAL_MODEL] = 1.0 if model_type == "cofilite" else 0.0
            try:
                self._cmd_queue.put(("switch_model", model_type, float(getattr(new_stage, "strength", 1.0))))
            except Exception:
                pass

    @property
    def input_gain_db(self) -> float:
        return self._chain.input_gain_db

    @property
    def stages(self) -> list[Any]:
        return self._chain.stages

    @stages.setter
    def stages(self, val: list[Any]) -> None:
        self._chain.stages = val
        for s in val:
            dn = s if getattr(s, "name", "") == "denoise" else getattr(s, "sub_stages", {}).get("denoise")
            if dn is not None:
                from ..dsp.cofilite import CoFiLiteDenoiser
                actual = getattr(dn, "_stage", dn)
                model_type = "cofilite" if isinstance(actual, CoFiLiteDenoiser) or getattr(actual, "model_type", "") == "cofilite" else "spectral"
                try:
                    self._cmd_queue.put(("switch_model", model_type, float(getattr(dn, "strength", 1.0))))
                except Exception:
                    pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chain, name)


class SharedAudioBuffer:
    """Zero-serialization circular audio buffer in POSIX / Windows shared memory."""

    def __init__(self, name: str | None = None, capacity_samples: int = 48000, create: bool = True) -> None:
        self.capacity = capacity_samples
        self.bytes_size = capacity_samples * 4  # float32 = 4 bytes
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=self.bytes_size)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self.name = self.shm.name
        self.buf = np.ndarray((self.capacity,), dtype=np.float32, buffer=self.shm.buf)

    def write(self, data: np.ndarray) -> None:
        if data.size == 0:
            return
        n = min(len(data), self.capacity)
        # Shift and write at end
        self.buf[:-n] = self.buf[n:]
        self.buf[-n:] = data[-n:]

    def read_latest(self, n: int) -> np.ndarray:
        n = min(n, self.capacity)
        return self.buf[-n:].copy()

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        try:
            self.shm.unlink()
        except Exception:
            pass


def _engine_worker_entry(
    cmd_queue: Any,
    reply_queue: Any,
    shared_metrics: Any,
    shm_name: str,
    app_dict: dict[str, Any],
    preset_name: str,
    dev_mic_index: int | None,
    dev_mic_name: str,
    dev_mic_max_input: int,
    sink_specs: list[tuple[str, int, str, int, float, bool]],
    model_override: str | None,
) -> None:
    """Worker process entrypoint: runs the audio streams and DSP outside the main GIL."""
    # Elevate audio process priority on Windows for glitch-free real-time streaming
    if os.name == "nt":
        try:
            import psutil
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        except Exception:
            pass

    from .devices import DeviceInfo
    from ..commands.common import make_chain

    app = AppConfig(**app_dict)
    chain, _ = make_chain(app, preset_name, model_override=model_override)

    mic_dev = DeviceInfo(
        index=dev_mic_index if dev_mic_index is not None else 0,
        name=dev_mic_name,
        hostapi="default",
        max_input=dev_mic_max_input,
        max_output=0,
        default_rate=float(app.sample_rate),
        is_default_input=True,
        is_default_output=False,
    )
    sinks = [
        SinkSpec(
            name=s[0],
            device=DeviceInfo(
                index=s[1],
                name=s[2],
                hostapi="default",
                max_input=0,
                max_output=s[3],
                default_rate=float(app.sample_rate),
                is_default_input=False,
                is_default_output=True,
            ),
            gain_db=s[4],
            enabled=s[5],
        )
        for s in sink_specs
    ]

    engine = AudioEngine(
        chain=chain,
        input_device=mic_dev,
        sinks=sinks,
        input_channel=app.input_channel,
        sample_rate=app.sample_rate,
        block_size=app.block_size,
        jitter_blocks=app.jitter_blocks,
        play_cues=app.hotkeys.play_cues,
        cue_volume=app.hotkeys.cue_volume,
        aec_reference_device=app.aec_reference_device,
    )

    audio_buf: SharedAudioBuffer | None = None
    try:
        audio_buf = SharedAudioBuffer(name=shm_name, capacity_samples=app.sample_rate, create=False)
    except Exception as exc:
        log.warning("Could not map shared audio buffer in worker: %s", exc)

    running = True

    with engine:
        shared_metrics[RUNNING] = 1.0
        while running:
            # 1. Process control commands non-blockingly
            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                except Exception:
                    break

                action = cmd[0]
                if action == "stop":
                    running = False
                    break
                elif action == "set_muted":
                    engine.set_muted(bool(cmd[1]))
                elif action == "set_bypass":
                    engine.set_bypass(bool(cmd[1]))
                elif action == "toggle_monitor":
                    engine.toggle_monitor()
                elif action == "switch_preset":
                    p_name = cmd[1]
                    new_chain, _ = make_chain(app, p_name, model_override=model_override)
                    engine.set_chain(new_chain)
                elif action == "set_input_gain":
                    engine.chain.set_input_gain_db(float(cmd[1]))
                elif action == "set_stage_attr":
                    _, s_name, attr, val = cmd
                    st = engine.chain.stage(s_name)
                    if st is not None:
                        if attr == "strength" and hasattr(st, "set_strength"):
                            st.set_strength(float(val))
                        else:
                            setattr(st, attr, val)
                elif action == "switch_model":
                    _, m_type, strn = cmd
                    dn = engine.chain.stage("denoise")
                    from ..dsp.cofilite import CoFiLiteDenoiser
                    from ..dsp.denoise import SpectralDenoiser
                    if m_type == "spectral":
                        new_dn = SpectralDenoiser(
                            app.sample_rate,
                            float(strn),
                            getattr(dn, "max_reduction_db", 24.0),
                            enabled=getattr(dn, "enabled", True),
                        )
                    else:
                        new_dn = CoFiLiteDenoiser(
                            app.sample_rate,
                            float(strn),
                            getattr(dn, "max_reduction_db", 24.0),
                            model_path=model_override,
                        )
                    engine.chain.replace_stage("denoise", new_dn)
                    for s in engine.sinks.values():
                        s.notify_pipeline_change()
                elif action == "begin_calibration":
                    engine.begin_calibration(float(cmd[1]))
                elif action == "pop_calibration":
                    cal = engine.pop_calibration()
                    reply_queue.put(cal)

            # 2. Update shared metrics for UI polling (~50 Hz)
            snap = engine.snapshot()
            shared_metrics[IN_RMS] = snap.in_level.rms_db
            shared_metrics[IN_PEAK] = snap.in_level.peak_db
            shared_metrics[IN_CLIP] = 1.0 if snap.in_level.clipping else 0.0

            shared_metrics[OUT_RMS] = snap.out_level.rms_db
            shared_metrics[OUT_PEAK] = snap.out_level.peak_db
            shared_metrics[OUT_CLIP] = 1.0 if snap.out_level.clipping else 0.0

            shared_metrics[CPU_LOAD] = snap.cpu_load
            shared_metrics[CPU_PEAK] = snap.cpu_peak
            shared_metrics[INPUT_OVERFLOWS] = float(snap.input_overflows)
            shared_metrics[OUTPUT_UNDERRUNS] = float(snap.output_underruns)
            shared_metrics[DRIFT_CORRECTIONS] = float(snap.drift_corrections)
            shared_metrics[ERRORS] = float(snap.errors)
            shared_metrics[LATENCY_MS] = snap.latency_ms

            shared_metrics[BYPASS] = 1.0 if snap.bypass else 0.0
            shared_metrics[MUTED] = 1.0 if snap.muted else 0.0
            shared_metrics[MONITOR_ON] = 1.0 if snap.monitor_on else 0.0
            shared_metrics[MONITOR_AVAILABLE] = 1.0 if snap.monitor_available else 0.0
            shared_metrics[CALIBRATING] = snap.calibrating if snap.calibrating is not None else -1.0
            shared_metrics[INPUT_GAIN_DB] = snap.input_gain_db

            if snap.ref_level:
                shared_metrics[REF_RMS] = snap.ref_level.rms_db
                shared_metrics[REF_PEAK] = snap.ref_level.peak_db

            # Stage telemetry
            dn = snap.telemetry.get("denoise", {})
            aec_tel = snap.telemetry.get("aec", {})
            dv_tel = snap.telemetry.get("deverb", {})
            gate_tel = snap.telemetry.get("gate", {})
            comp_tel = snap.telemetry.get("compressor", {})
            lim_tel = snap.telemetry.get("limiter", {})
            deess_tel = snap.telemetry.get("deesser", {})
            prox_tel = snap.telemetry.get("proximity", {})

            shared_metrics[SPEECH_PROB] = dn.get("speech_prob", 0.0)
            shared_metrics[IS_SPEECH] = dn.get("is_speech", 0.0)
            shared_metrics[NAPPING] = dn.get("napping", 0.0)
            shared_metrics[DENOISE_RED_DB] = dn.get("reduction_db", 0.0)
            shared_metrics[NOISE_FLOOR_DB] = dn.get("noise_floor_db", -65.0)

            dn_stage = engine.chain.stage("denoise")
            from ..dsp.cofilite import CoFiLiteDenoiser
            is_neural = 1.0 if isinstance(dn_stage, CoFiLiteDenoiser) or getattr(dn_stage, "model_type", "") == "cofilite" else 0.0
            shared_metrics[IS_NEURAL_MODEL] = is_neural

            shared_metrics[ERLE_DB] = aec_tel.get("erle_db", 0.0)
            shared_metrics[DTD_ACTIVE] = aec_tel.get("dtd_active", 0.0)
            shared_metrics[REVERB_RED_DB] = dv_tel.get("reverb_reduction_db", 0.0)

            timings = snap.telemetry.get("stage_timings", {})
            tot = timings.get("total", {})
            shared_metrics[TOT_P50_US] = tot.get("p50_us", 0.0)
            shared_metrics[TOT_P99_US] = tot.get("p99_us", 0.0)
            shared_metrics[DN_P99_US] = timings.get("denoise", {}).get("p99_us", 0.0)

            shared_metrics[GATE_ATTEN_DB] = gate_tel.get("attenuation_db", 0.0)
            shared_metrics[GATE_OPEN] = gate_tel.get("open", 1.0)
            shared_metrics[COMP_RED_DB] = comp_tel.get("gain_reduction_db", 0.0)
            shared_metrics[LIM_RED_DB] = lim_tel.get("gain_reduction_db", 0.0)
            shared_metrics[DEESS_RED_DB] = deess_tel.get("gain_reduction_db", 0.0)
            shared_metrics[PROX_BOOST_DB] = prox_tel.get("boost_db", 0.0)

            shared_metrics[AEC_ENABLED] = aec_tel.get("enabled", 1.0)
            shared_metrics[AI_ENABLED] = dn.get("enabled", 1.0)
            shared_metrics[VERB_ENABLED] = dv_tel.get("enabled", 1.0)
            shared_metrics[GATE_ENABLED] = gate_tel.get("enabled", 1.0)
            shared_metrics[PROX_ENABLED] = prox_tel.get("enabled", 1.0)
            shared_metrics[DEESS_ENABLED] = deess_tel.get("enabled", 1.0)
            shared_metrics[COMP_ENABLED] = comp_tel.get("enabled", 1.0)
            shared_metrics[LIM_ENABLED] = lim_tel.get("enabled", 1.0)
            shared_metrics[LVL_ENABLED] = snap.telemetry.get("leveler", {}).get("enabled", 1.0)

            # Spectrum bands (up to 32 bands)
            nb_in = min(len(snap.in_bands), 32)
            if nb_in > 0:
                shared_metrics[IN_BANDS_START : IN_BANDS_START + nb_in] = snap.in_bands[:nb_in]
            nb_out = min(len(snap.out_bands), 32)
            if nb_out > 0:
                shared_metrics[OUT_BANDS_START : OUT_BANDS_START + nb_out] = snap.out_bands[:nb_out]

            time.sleep(0.020)

        shared_metrics[RUNNING] = 0.0

    if audio_buf:
        audio_buf.close()


class IsolatedAudioEngine:
    """Client proxy for the out-of-process AudioEngine with crash-restart watchdog."""

    def __init__(
        self,
        app: AppConfig,
        preset: str,
        input_device: Any,
        sinks: list[SinkSpec],
        model_override: str | None = None,
    ) -> None:
        self.app = app
        self.preset = preset
        self.input_device = input_device
        self.sinks_list = sinks
        self.sinks = {s.name: IsolatedOutputSink(s) for s in sinks}
        self.model_override = model_override
        from ..commands.common import make_chain
        raw_chain, _ = make_chain(self.app, self.preset, model_override=self.model_override)

        self._cmd_queue = mp.Queue()
        self._reply_queue = mp.Queue()
        self._shared_metrics = mp.Array("d", METRICS_SIZE)
        self._shm_buf = SharedAudioBuffer(capacity_samples=app.sample_rate, create=True)

        dn_init = raw_chain.stage("denoise")
        self._shared_metrics[IS_NEURAL_MODEL] = 1.0 if getattr(dn_init, "model_type", "") == "cofilite" else 0.0

        self.chain = ChainProxy(raw_chain, self._cmd_queue, self._shared_metrics)
        from .metrics import SpectrumAnalyzer
        self._analyzer = SpectrumAnalyzer(app.sample_rate)

        self.cues = CuePlayer(sample_rate=app.sample_rate, volume=app.hotkeys.cue_volume, enabled=app.hotkeys.play_cues)

        # Local cached state for sub-millisecond UI responsiveness
        self.muted = False
        self.bypass = False
        self.running = False
        self._stopping = False
        self._recovering = False

        self._proc: mp.Process | None = None
        self._watchdog_thread: threading.Thread | None = None

    @property
    def spectrum_range(self) -> tuple[float, float]:
        return self._analyzer.f_lo, self._analyzer.f_hi

    @property
    def spectrum_centers(self) -> np.ndarray:
        return self._analyzer.band_centers()

    @property
    def monitor_available(self) -> bool:
        return "monitor" in self.sinks

    def start(self) -> None:
        self._stopping = False
        self._spawn_worker()
        self.running = True

        # Launch watchdog thread
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _spawn_worker(self) -> None:
        sink_tuples = [
            (
                s.name,
                s.device.index if s.device else 0,
                s.device.name if s.device else "default",
                getattr(s.device, "max_output", 2),
                float(getattr(s, "gain_db", 0.0)),
                bool(s.enabled),
            )
            for s in self.sinks.values()
        ]

        app_dict = {
            "sample_rate": self.app.sample_rate,
            "block_size": self.app.block_size,
            "jitter_blocks": self.app.jitter_blocks,
            "input_channel": self.app.input_channel,
            "preset": self.preset,
            "aec_reference_device": self.app.aec_reference_device,
            "hotkeys": self.app.hotkeys,
        }

        self._proc = mp.Process(
            target=_engine_worker_entry,
            args=(
                self._cmd_queue,
                self._reply_queue,
                self._shared_metrics,
                self._shm_buf.name,
                app_dict,
                self.preset,
                self.input_device.index if self.input_device else None,
                self.input_device.name if self.input_device else "default",
                getattr(self.input_device, "max_input", 1),
                sink_tuples,
                self.model_override,
            ),
            daemon=True,
        )
        self._proc.start()

    def _watchdog_loop(self) -> None:
        """Watchdog: automatically restarts the engine subprocess on crash."""
        while not self._stopping:
            time.sleep(0.25)
            if self._proc is not None and not self._proc.is_alive() and not self._stopping:
                log.warning("AudioEngine process died unexpectedly! Watchdog restarting engine...")
                self._recovering = True
                try:
                    self._spawn_worker()
                    # Restore state
                    if self.muted:
                        self._cmd_queue.put(("set_muted", True))
                    if self.bypass:
                        self._cmd_queue.put(("set_bypass", True))
                    time.sleep(0.5)
                except Exception as exc:
                    log.error("Failed to restart engine worker: %s", exc)
                finally:
                    self._recovering = False

    def stop(self) -> None:
        self._stopping = True
        self.running = False
        if self._proc and self._proc.is_alive():
            try:
                self._cmd_queue.put(("stop",))
                self._proc.join(timeout=1.5)
                if self._proc.is_alive():
                    self._proc.terminate()
            except Exception:
                pass
        self._shm_buf.close()
        self._shm_buf.unlink()

    def __enter__(self) -> "IsolatedAudioEngine":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------ controls
    def set_muted(self, muted: bool) -> bool:
        self.muted = bool(muted)
        self._cmd_queue.put(("set_muted", self.muted))
        self.cues.play("mute" if self.muted else "unmute")
        return self.muted

    def toggle_mute(self) -> bool:
        return self.set_muted(not self.muted)

    def set_bypass(self, bypass: bool) -> bool:
        self.bypass = bool(bypass)
        self._cmd_queue.put(("set_bypass", self.bypass))
        self.cues.play("toggle")
        return self.bypass

    def toggle_bypass(self) -> bool:
        return self.set_bypass(not self.bypass)

    def toggle_monitor(self) -> bool:
        self._cmd_queue.put(("toggle_monitor",))
        return bool(self._shared_metrics[MONITOR_ON] == 0.0)

    def set_preset(self, name: str) -> None:
        self.preset = name
        from ..commands.common import make_chain
        raw_chain, _ = make_chain(self.app, name, model_override=self.model_override)
        self.chain = ChainProxy(raw_chain, self._cmd_queue, self._shared_metrics)
        self._shared_metrics[IS_NEURAL_MODEL] = 1.0 if getattr(raw_chain.stage("denoise"), "model_type", "") == "cofilite" else 0.0
        self._cmd_queue.put(("switch_preset", name))

    def switch_model(self, model_type: str, strength: float = 1.0) -> None:
        self._shared_metrics[IS_NEURAL_MODEL] = 1.0 if model_type == "cofilite" else 0.0
        try:
            self._cmd_queue.put(("switch_model", model_type, float(strength)))
        except Exception:
            pass

    def set_chain(self, chain: Any) -> None:
        self.chain = ChainProxy(chain, self._cmd_queue, self._shared_metrics)
        self._shared_metrics[IS_NEURAL_MODEL] = 1.0 if getattr(chain.stage("denoise"), "model_type", "") == "cofilite" else 0.0

    def set_input_gain_db(self, db: float) -> None:
        self._cmd_queue.put(("set_input_gain", float(db)))

    def begin_calibration(self, seconds: float = 3.0) -> None:
        self._cmd_queue.put(("begin_calibration", float(seconds)))

    def pop_calibration(self) -> CalibrationResult | None:
        self._cmd_queue.put(("pop_calibration",))
        try:
            return self._reply_queue.get(timeout=0.05)
        except Exception:
            return None

    def snapshot(self) -> EngineSnapshot:
        """Poll latest telemetry and levels from the shared array (zero GIL overhead)."""
        m = self._shared_metrics
        in_level = LevelSnapshot(
            rms_db=float(m[IN_RMS]),
            peak_db=float(m[IN_PEAK]),
            clipping=bool(m[IN_CLIP] > 0.5),
        )
        out_level = LevelSnapshot(
            rms_db=float(m[OUT_RMS]),
            peak_db=float(m[OUT_PEAK]),
            clipping=bool(m[OUT_CLIP] > 0.5),
        )

        in_bands = np.array(m[IN_BANDS_START : IN_BANDS_START + 32], dtype=np.float32)
        out_bands = np.array(m[OUT_BANDS_START : OUT_BANDS_START + 32], dtype=np.float32)

        ref_level = None
        if m[REF_RMS] > -100.0:
            ref_level = LevelSnapshot(rms_db=float(m[REF_RMS]), peak_db=float(m[REF_PEAK]), clipping=False)

        cal = float(m[CALIBRATING])
        cal_val = cal if cal >= 0.0 else None

        telemetry: dict[str, dict[str, float]] = {
            "denoise": {
                "speech_prob": float(m[SPEECH_PROB]),
                "is_speech": float(m[IS_SPEECH]),
                "napping": float(m[NAPPING]),
                "reduction_db": float(m[DENOISE_RED_DB]),
                "noise_floor_db": float(m[NOISE_FLOOR_DB]),
                "enabled": float(m[AI_ENABLED]),
                "is_neural": float(m[IS_NEURAL_MODEL]),
                "model": "cofilite" if m[IS_NEURAL_MODEL] > 0.5 else "spectral",
            },
            "aec": {
                "erle_db": float(m[ERLE_DB]),
                "dtd_active": float(m[DTD_ACTIVE]),
                "enabled": float(m[AEC_ENABLED]),
            },
            "deverb": {
                "reverb_reduction_db": float(m[REVERB_RED_DB]),
                "enabled": float(m[VERB_ENABLED]),
            },
            "gate": {
                "open": float(m[GATE_OPEN]),
                "attenuation_db": float(m[GATE_ATTEN_DB]),
                "enabled": float(m[GATE_ENABLED]),
            },
            "proximity": {
                "boost_db": float(m[PROX_BOOST_DB]),
                "enabled": float(m[PROX_ENABLED]),
            },
            "compressor": {
                "gain_reduction_db": float(m[COMP_RED_DB]),
                "enabled": float(m[COMP_ENABLED]),
            },
            "limiter": {
                "gain_reduction_db": float(m[LIM_RED_DB]),
                "enabled": float(m[LIM_ENABLED]),
            },
            "deesser": {
                "gain_reduction_db": float(m[DEESS_RED_DB]),
                "enabled": float(m[DEESS_ENABLED]),
            },
            "leveler": {
                "enabled": float(m[LVL_ENABLED]),
            },
            "stage_timings": {
                "total": {
                    "p50_us": float(m[TOT_P50_US]),
                    "p99_us": float(m[TOT_P99_US]),
                },
                "denoise": {
                    "p99_us": float(m[DN_P99_US]),
                },
            },
        }

        return EngineSnapshot(
            running=bool(m[RUNNING] > 0.5),
            in_level=in_level,
            out_level=out_level,
            in_bands=in_bands,
            out_bands=out_bands,
            telemetry=telemetry,
            cpu_load=float(m[CPU_LOAD]),
            cpu_peak=float(m[CPU_PEAK]),
            input_overflows=int(m[INPUT_OVERFLOWS]),
            output_underruns=int(m[OUTPUT_UNDERRUNS]),
            drift_corrections=int(m[DRIFT_CORRECTIONS]),
            errors=int(m[ERRORS]),
            latency_ms=float(m[LATENCY_MS]),
            bypass=bool(m[BYPASS] > 0.5),
            muted=bool(m[MUTED] > 0.5),
            monitor_on=bool(m[MONITOR_ON] > 0.5),
            monitor_available=bool(m[MONITOR_AVAILABLE] > 0.5),
            calibrating=cal_val,
            input_gain_db=float(m[INPUT_GAIN_DB]),
            ref_level=ref_level,
            recovering=self._recovering,
        )
