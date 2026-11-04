"""Assemble processors into a chain and build one from a ChainConfig."""
from __future__ import annotations

import numpy as np

from ..config import ChainConfig
from .base import Processor
from .biquad import BiquadFilter
from .compressor import Compressor, Limiter
from .deesser import DeEsser
from .denoise import SpectralDenoiser
from .gate import NoiseGate
from .leveler import Leveler
from .util import db2lin


class Chain:
    """Ordered list of processors plus input/output trim."""

    def __init__(self, stages: list[Processor], input_gain_db: float = 0.0, output_gain_db: float = 0.0):
        self.stages = stages
        self.input_gain_db = input_gain_db
        self.output_gain_db = output_gain_db

    def stage(self, name: str) -> Processor | None:
        return next((s for s in self.stages if s.name == name), None)

    @property
    def latency_samples(self) -> int:
        return sum(s.latency_samples for s in self.stages if s.enabled)

    def set_input_gain_db(self, db: float) -> None:
        self.input_gain_db = float(min(40.0, max(-40.0, db)))

    def process(self, x: np.ndarray) -> np.ndarray:
        y = x * np.float32(db2lin(self.input_gain_db)) if self.input_gain_db else x
        for s in self.stages:
            if s.enabled:
                y = s.process(y)
        if self.output_gain_db:
            y = y * np.float32(db2lin(self.output_gain_db))
        return y

    def reset(self) -> None:
        for s in self.stages:
            s.reset()

    def telemetry(self) -> dict[str, dict[str, float]]:
        return {s.name: s.telemetry() for s in self.stages if s.enabled}


def build_chain(
    cfg: ChainConfig,
    sample_rate: int,
    noise_profile: np.ndarray | None = None,
    model_override: str | None = None,
) -> Chain:
    cfg.validate()
    hp_enabled = cfg.highpass_hz > 0
    model_path = model_override or cfg.denoise.model_path

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
                )
            else:
                denoise = SpectralDenoiser(
                    sample_rate, cfg.denoise.strength, cfg.denoise.max_reduction_db, enabled=cfg.denoise.enabled
                )
        except Exception:
            denoise = SpectralDenoiser(
                sample_rate, cfg.denoise.strength, cfg.denoise.max_reduction_db, enabled=cfg.denoise.enabled
            )
    else:
        denoise = SpectralDenoiser(
            sample_rate, cfg.denoise.strength, cfg.denoise.max_reduction_db, enabled=cfg.denoise.enabled
        )

    if noise_profile is not None and hasattr(denoise, "set_profile"):
        denoise.set_profile(noise_profile)

    g, d, lv, c, li = cfg.gate, cfg.deesser, cfg.leveler, cfg.compressor, cfg.limiter
    stages: list[Processor] = [
        BiquadFilter.highpass(sample_rate, cfg.highpass_hz or 80.0, order=4, enabled=hp_enabled),
        denoise,
        NoiseGate(
            sample_rate, g.threshold_db, g.range_db, g.attack_ms, g.hold_ms, g.release_ms, g.hysteresis_db,
            enabled=g.enabled,
        ),
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
