"""Simple and Advanced tuning schema and macro-to-DSP mappings."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import ChainConfig, CompressorConfig, DenoiseConfig, EQBand, GateConfig, LimiterConfig
from ..presets import BUILTIN, get_preset


@dataclass
class MacroKnobs:
    """Four intuitive macro knobs for casual/streamer tuning (0-100 scale)."""
    clarity: float = 50.0        # Presence, air, sibilance control, mud cut
    warmth: float = 50.0         # Chest resonance, proximity body, low-end richness
    noise_removal: float = 65.0  # AI suppression depth & gate aggressiveness
    room_cleanup: float = 40.0   # Room flutter dereverberation & rumble isolation

    def __post_init__(self) -> None:
        self.clamp()

    def clamp(self) -> None:
        self.clarity = float(max(0.0, min(100.0, self.clarity)))
        self.warmth = float(max(0.0, min(100.0, self.warmth)))
        self.noise_removal = float(max(0.0, min(100.0, self.noise_removal)))
        self.room_cleanup = float(max(0.0, min(100.0, self.room_cleanup)))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def macro_to_chain_config(macros: MacroKnobs, base_preset_name: str | ChainConfig = "discord") -> ChainConfig:
    """Deterministically map 4 macro knobs (0-100) onto a full 11-stage DSP ChainConfig."""
    macros.clamp()

    # Always clone a fresh instance so built-ins are never mutated
    if isinstance(base_preset_name, ChainConfig):
        cfg = copy.deepcopy(base_preset_name)
    else:
        cfg = copy.deepcopy(get_preset(base_preset_name))

    # 1. CLARITY (0 - 100)
    # Scales: presence peak (+0 to +6 dB at 3200 Hz), air shelf (+0 to +5 dB at 9000 Hz), mud cut (-0 to -4.5 dB at 280 Hz)
    clarity_norm = macros.clarity / 100.0
    presence_gain = 1.0 + (clarity_norm * 4.5)     # 1.0 -> 5.5 dB
    air_gain = 1.0 + (clarity_norm * 4.0)          # 1.0 -> 5.0 dB
    mud_cut = -1.0 - (clarity_norm * 3.5)          # -1.0 -> -4.5 dB

    cfg.eq = [
        EQBand("peak", 280, mud_cut, 1.1),
        EQBand("peak", 3200, presence_gain, 0.9),
        EQBand("highshelf", 9000, air_gain, 0.7),
    ]
    # Sibilance taming scales with clarity to prevent harshness
    if clarity_norm > 0.3:
        cfg.deesser.enabled = True
        cfg.deesser.threshold_db = -28.0 - (clarity_norm * 8.0)  # -30.4 to -36.0 dBFS
        cfg.deesser.ratio = 2.5 + (clarity_norm * 2.0)
    else:
        cfg.deesser.enabled = False

    # 2. WARMTH (0 - 100)
    # Scales: proximity effect amount, saturation warmth, chest resonance
    warmth_norm = macros.warmth / 100.0
    if warmth_norm > 0.05:
        cfg.proximity.enabled = True
        cfg.proximity.amount = float(np_clip(warmth_norm * 1.1, 0.0, 1.0))
        cfg.proximity.warmth = float(np_clip(warmth_norm * 0.75, 0.0, 0.85))
        cfg.proximity.body_hz = 105.0 + (warmth_norm * 35.0)  # 105 -> 140 Hz
        cfg.proximity.mud_cleanup = float(np_clip(0.3 + clarity_norm * 0.4, 0.0, 1.0))
    else:
        cfg.proximity.enabled = False

    # 3. NOISE REMOVAL (0 - 100)
    # Scales: AI suppression strength (0 to 1.0) and gate threshold
    noise_norm = macros.noise_removal / 100.0
    cfg.denoise.enabled = noise_norm > 0.05
    cfg.denoise.strength = float(np_clip(noise_norm, 0.0, 1.0))
    cfg.denoise.max_reduction_db = 14.0 + (noise_norm * 20.0)  # 14 -> 34 dB

    if noise_norm > 0.2:
        cfg.gate.enabled = True
        cfg.gate.threshold_db = -54.0 + (noise_norm * 18.0)    # -50.4 -> -36.0 dBFS
        cfg.gate.range_db = 15.0 + (noise_norm * 20.0)         # 19 -> 35 dB
    else:
        cfg.gate.enabled = False

    # 4. ROOM CLEANUP (0 - 100)
    # Scales: room de-verb amount and highpass rumble filter
    room_norm = macros.room_cleanup / 100.0
    if room_norm > 0.08:
        cfg.deverb.enabled = True
        cfg.deverb.amount = float(np_clip(room_norm * 0.9, 0.0, 0.85))
        cfg.deverb.room_decay = 0.25 + (room_norm * 0.5)       # 0.29 -> 0.75 s
    else:
        cfg.deverb.enabled = False

    cfg.highpass_hz = int(75 + (room_norm * 35))              # 75 -> 110 Hz

    cfg.validate()
    return cfg


def chain_config_to_macro(cfg: ChainConfig) -> MacroKnobs:
    """Project a full DSP ChainConfig back into 4 approximate macro knob values (0-100)."""
    # 1. Clarity estimate from EQ presence and air
    eq_gains = [b.gain_db for b in cfg.eq if b.gain_db > 0]
    clarity = 50.0
    if eq_gains:
        clarity = float(np_clip(np_mean(eq_gains) * 15.0, 0.0, 100.0))

    # 2. Warmth estimate from proximity
    warmth = 0.0
    if cfg.proximity.enabled:
        warmth = float(np_clip(cfg.proximity.amount * 100.0, 0.0, 100.0))

    # 3. Noise removal estimate
    noise = 0.0
    if cfg.denoise.enabled:
        noise = float(np_clip(cfg.denoise.strength * 100.0, 0.0, 100.0))

    # 4. Room cleanup estimate
    room = 0.0
    if cfg.deverb.enabled:
        room = float(np_clip(cfg.deverb.amount * 115.0, 0.0, 100.0))

    knobs = MacroKnobs(clarity=clarity, warmth=warmth, noise_removal=noise, room_cleanup=room)
    knobs.clamp()
    return knobs


def np_clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def np_mean(arr: list[float]) -> float:
    return sum(arr) / len(arr) if arr else 0.0
