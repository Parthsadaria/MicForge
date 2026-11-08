"""Assemble processors into a chain and build one from a ChainConfig."""
from __future__ import annotations

import time
from typing import Any
import numpy as np

from ..config import ChainConfig
from .aec import AcousticEchoCanceller
from .base import Processor
from .biquad import BiquadFilter
from .compressor import Compressor, Limiter
from .deesser import DeEsser
from .denoise import SpectralDenoiser
from .deverb import RoomDereverberation
from .gate import NoiseGate
from .leveler import Leveler
from .proximity import VoiceProximity
from .spectral_rack import SpectralRack
from .util import db2lin


class StageTimingStats:
    """Pre-allocated circular buffer recording execution time per stage with p50/p99 tracking.

    Ignores initial warmup blocks (ONNX session setup, JIT, page cache priming) to ensure
    statistics reflect true steady-state real-time performance.
    """

    def __init__(self, capacity: int = 512, warmup_blocks: int = 200) -> None:
        self.capacity = capacity
        self.warmup_blocks = warmup_blocks
        self.buf = np.zeros(capacity, dtype=np.int64)
        self.count = 0
        self.total_blocks = 0
        self._cached_p50 = 0.0
        self._cached_p99 = 0.0
        self._last_calc = -1

    def record(self, elapsed_ns: int) -> None:
        self.total_blocks += 1
        if self.total_blocks <= self.warmup_blocks:
            return
        idx = self.count % self.capacity
        self.buf[idx] = elapsed_ns
        self.count += 1

    def percentiles_us(self) -> tuple[float, float]:
        """Return (p50_us, p99_us) in microseconds with count-based caching."""
        if self.count == self._last_calc:
            return self._cached_p50, self._cached_p99
        valid = min(self.count, self.capacity)
        if valid < 5:
            return 0.0, 0.0
        arr = self.buf[:valid]
        p50, p99 = np.percentile(arr, [50, 99])
        self._cached_p50 = float(p50 / 1000.0)
        self._cached_p99 = float(p99 / 1000.0)
        self._last_calc = self.count
        return self._cached_p50, self._cached_p99


class Chain:
    """Ordered list of processors plus input/output trim."""

    def __init__(self, stages: list[Processor], input_gain_db: float = 0.0, output_gain_db: float = 0.0):
        self._stages = stages
        self.input_gain_db = input_gain_db
        self.output_gain_db = output_gain_db
        self._init_stats_and_stages()

    def _init_stats_and_stages(self) -> None:
        self._timing_stats: dict[str, StageTimingStats] = {}
        for s in self._stages:
            self._timing_stats[s.name] = StageTimingStats(capacity=512, warmup_blocks=200)
            if hasattr(s, "sub_stages"):
                for sub_name in s.sub_stages:
                    self._timing_stats[sub_name] = StageTimingStats(capacity=512, warmup_blocks=200)
        self._total_timing = StageTimingStats(capacity=512, warmup_blocks=200)
        self._denoise_stage = self.stage("denoise")
        self._gate_stage = self.stage("gate")

    @property
    def stages(self) -> list[Processor]:
        return self._stages

    @stages.setter
    def stages(self, new_stages: list[Processor]) -> None:
        self._stages = new_stages
        self._init_stats_and_stages()

    def stage(self, name: str) -> Processor | None:
        for s in self._stages:
            if s.name == name:
                return s
            if hasattr(s, "sub_stages") and name in s.sub_stages:
                return s.sub_stages[name]
        return None

    def replace_stage(self, name: str, new_stage: Processor) -> None:
        for i, s in enumerate(self._stages):
            if s.name == name:
                self._stages[i] = new_stage
                self._init_stats_and_stages()
                return
            if hasattr(s, "sub_stages") and name in s.sub_stages:
                if hasattr(s, "replace_sub_stage"):
                    s.replace_sub_stage(name, new_stage)
                else:
                    s.sub_stages[name] = new_stage
                    setattr(s, name, new_stage)
                self._init_stats_and_stages()
                return

    @property
    def latency_samples(self) -> int:
        return sum(s.latency_samples for s in self._stages if s.enabled)

    def set_input_gain_db(self, db: float) -> None:
        self.input_gain_db = float(min(40.0, max(-40.0, db)))

    def process(self, x: np.ndarray) -> np.ndarray:
        t_tot_0 = time.perf_counter_ns()
        y = x * np.float32(db2lin(self.input_gain_db)) if self.input_gain_db else x
        for s in self._stages:
            if s.enabled:
                t0 = time.perf_counter_ns()
                y = s.process(y)
                t1 = time.perf_counter_ns()
                self._timing_stats[s.name].record(t1 - t0)
                if hasattr(s, "sub_stages") and hasattr(s, "get_sub_timing_ns"):
                    for sub_name in s.sub_stages:
                        sub_t = s.get_sub_timing_ns(sub_name)
                        if sub_name in self._timing_stats:
                            self._timing_stats[sub_name].record(sub_t)
                if (s is self._denoise_stage or hasattr(s, "sub_stages")) and self._gate_stage is not None:
                    self._gate_stage.speech_prob = getattr(s, "speech_prob", None)
            else:
                self._timing_stats[s.name].record(0)
        if self.output_gain_db:
            y = y * np.float32(db2lin(self.output_gain_db))
        t_tot_1 = time.perf_counter_ns()
        self._total_timing.record(t_tot_1 - t_tot_0)
        return y

    def reset(self) -> None:
        for s in self._stages:
            s.reset()

    def stage_timings(self) -> dict[str, dict[str, float]]:
        """Return {stage_name: {'p50_us': ..., 'p99_us': ..., 'enabled': bool}}."""
        res: dict[str, dict[str, float]] = {}
        for s in self._stages:
            if hasattr(s, "sub_stages"):
                for sub_name, sub in s.sub_stages.items():
                    stats = self._timing_stats.get(sub_name)
                    if stats is not None:
                        p50, p99 = stats.percentiles_us()
                        res[sub_name] = {"p50_us": p50, "p99_us": p99, "enabled": float(sub.enabled)}
                p50, p99 = self._timing_stats[s.name].percentiles_us()
                res[s.name] = {"p50_us": p50, "p99_us": p99, "enabled": float(s.enabled)}
            else:
                p50, p99 = self._timing_stats[s.name].percentiles_us()
                res[s.name] = {"p50_us": p50, "p99_us": p99, "enabled": float(s.enabled)}
        tot_p50, tot_p99 = self._total_timing.percentiles_us()
        res["total"] = {"p50_us": tot_p50, "p99_us": tot_p99, "enabled": 1.0}
        return res

    def telemetry(self) -> dict[str, Any]:
        tel: dict[str, Any] = {}
        for s in self._stages:
            if hasattr(s, "sub_stages"):
                for sub_name, sub in s.sub_stages.items():
                    if sub.enabled:
                        tel[sub_name] = sub.telemetry()
            elif s.enabled:
                tel[s.name] = s.telemetry()
        tel["stage_timings"] = self.stage_timings()
        return tel


def build_chain(
    cfg: ChainConfig,
    sample_rate: int,
    noise_profile: np.ndarray | None = None,
    model_override: str | None = None,
) -> Chain:
    cfg.validate()
    hp_enabled = cfg.highpass_hz > 0
    model_path = model_override or cfg.denoise.model_path

    denoise_naturalness = getattr(cfg.denoise, "naturalness", 0.08)
    denoise: Processor
    if cfg.denoise.model_type == "cofilite":
        try:
            from .cofilite import CoFiLiteDenoiser, find_model_file
            if find_model_file(model_path) is not None:
                denoise = CoFiLiteDenoiser(
                    sample_rate,
                    cfg.denoise.strength,
                    cfg.denoise.max_reduction_db,
                    model_path=model_path,
                    enabled=cfg.denoise.enabled,
                    naturalness=denoise_naturalness,
                )
            else:
                denoise = SpectralDenoiser(
                    sample_rate,
                    cfg.denoise.strength,
                    cfg.denoise.max_reduction_db,
                    enabled=cfg.denoise.enabled,
                    naturalness=denoise_naturalness,
                )
        except Exception:
            denoise = SpectralDenoiser(
                sample_rate,
                cfg.denoise.strength,
                cfg.denoise.max_reduction_db,
                enabled=cfg.denoise.enabled,
                naturalness=denoise_naturalness,
            )
    else:
        denoise = SpectralDenoiser(
            sample_rate,
            cfg.denoise.strength,
            cfg.denoise.max_reduction_db,
            enabled=cfg.denoise.enabled,
            naturalness=denoise_naturalness,
        )

    if noise_profile is not None and hasattr(denoise, "set_profile"):
        denoise.set_profile(noise_profile)

    aec_cfg = getattr(cfg, "aec", None)
    aec = AcousticEchoCanceller(
        sample_rate,
        aec_cfg.amount if aec_cfg else 1.0,
        aec_cfg.filter_length_ms if aec_cfg else 60.0,
        aec_cfg.delay_ms if aec_cfg else 40.0,
        aec_cfg.suppression_db if aec_cfg else 24.0,
        enabled=aec_cfg.enabled if aec_cfg else False,
    )

    deverb_cfg = getattr(cfg, "deverb", None)
    deverb = RoomDereverberation(
        sample_rate,
        deverb_cfg.amount if deverb_cfg else 0.5,
        deverb_cfg.room_decay if deverb_cfg else 0.4,
        deverb_cfg.early_retention if deverb_cfg else 0.8,
        enabled=deverb_cfg.enabled if deverb_cfg else True,
    )

    prox_cfg = getattr(cfg, "proximity", None)
    proximity = VoiceProximity(
        sample_rate,
        prox_cfg.amount if prox_cfg else 0.5,
        prox_cfg.body_hz if prox_cfg else 120.0,
        prox_cfg.warmth if prox_cfg else 0.35,
        prox_cfg.mud_cleanup if prox_cfg else 0.4,
        enabled=prox_cfg.enabled if prox_cfg else True,
    )

    spectral_rack = SpectralRack(sample_rate, aec, denoise, deverb)

    g, d, lv, c, li = cfg.gate, cfg.deesser, cfg.leveler, cfg.compressor, cfg.limiter
    stages: list[Processor] = [
        BiquadFilter.highpass(sample_rate, cfg.highpass_hz or 80.0, order=4, enabled=hp_enabled),
        spectral_rack,
        NoiseGate(
            sample_rate, g.threshold_db, g.range_db, g.attack_ms, g.hold_ms, g.release_ms, g.hysteresis_db,
            enabled=g.enabled,
        ),
        proximity,
        BiquadFilter.from_bands(sample_rate, cfg.eq, name="eq", enabled=cfg.eq_enabled and bool(cfg.eq)),
        DeEsser(sample_rate, d.freq_hz, d.threshold_db, d.ratio, enabled=d.enabled),
        Leveler(sample_rate, lv.target_db, lv.max_gain_db, lv.max_cut_db, lv.gate_db, enabled=lv.enabled),
        Compressor(
            sample_rate, c.threshold_db, c.ratio, c.attack_ms, c.release_ms, c.knee_db, c.makeup_db,
            enabled=c.enabled,
        ),
        Limiter(sample_rate, li.ceiling_db, li.release_ms, enabled=li.enabled),
    ]
    return Chain(stages, cfg.input_gain_db, cfg.output_gain_db)


def process_offline(chain: Chain, x: np.ndarray, block: int = 480) -> np.ndarray:
    """Run a whole array through the chain block-by-block, compensating latency."""
    x = np.asarray(x, dtype=np.float32)
    lat = chain.latency_samples
    padded = np.concatenate((x, np.zeros(lat, dtype=np.float32)))
    out = np.empty_like(padded)
    for i in range(0, padded.shape[0], block):
        out[i : i + block] = chain.process(padded[i : i + block])
    return out[lat:]
