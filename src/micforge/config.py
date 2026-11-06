"""Typed configuration: DSP chain settings + application settings.

Everything is a plain dataclass so it round-trips through JSON and stays easy
to test. Unknown keys in files are ignored; missing keys fall back to defaults.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .errors import ConfigError

APP_NAME = "micforge"
T = TypeVar("T")


def config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_NAME


def _build(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively build a dataclass from a dict, ignoring unknown keys."""
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value, hint = data[f.name], hints[f.name]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = _build(hint, value)
        elif typing.get_origin(hint) is list and isinstance(value, list):
            (inner,) = typing.get_args(hint)
            if dataclasses.is_dataclass(inner):
                value = [_build(inner, v) for v in value if isinstance(v, dict)]
        kwargs[f.name] = value
    return cls(**kwargs)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# DSP chain configuration
# --------------------------------------------------------------------------- #
@dataclass
class EQBand:
    kind: str = "peak"  # peak | lowshelf | highshelf | highpass | lowpass | bandpass
    freq: float = 1000.0
    gain_db: float = 0.0
    q: float = 0.707


@dataclass
class AECConfig:
    enabled: bool = False  # Enabled when loopback reference is provided or active in preset
    amount: float = 1.0  # 0..1 cancellation intensity
    filter_length_ms: float = 60.0  # acoustic tail memory (20..200 ms)
    delay_ms: float = 40.0  # speaker-to-mic loopback alignment (0..250 ms)
    suppression_db: float = 24.0  # residual echo suppression floor (6..40 dB)


@dataclass
class DeverbConfig:
    enabled: bool = True
    amount: float = 0.5  # 0..1 late reverberation reduction aggression
    room_decay: float = 0.4  # estimated RT60 decay time in seconds (0.15..1.5s)
    early_retention: float = 0.8  # 0..1 preserves natural direct voice warmth


@dataclass
class ProximityConfig:
    enabled: bool = True
    amount: float = 0.5  # 0..1 chest resonance boost intensity
    body_hz: float = 120.0  # center frequency of voice fundamental (70..250 Hz)
    warmth: float = 0.35  # 0..1 even-order low harmonic saturation
    mud_cleanup: float = 0.4  # 0..1 dynamic 300 Hz anti-boxiness cleanup


@dataclass
class DenoiseConfig:
    enabled: bool = True
    strength: float = 1.0  # 0..1, how aggressively noise is attenuated (1.0 = full neural suppression)
    max_reduction_db: float = 24.0  # attenuation floor per bin
    model_type: str = "cofilite"  # "cofilite" | "spectral"
    model_path: str | None = None  # path to swappable model file (.onnx or .tar)


@dataclass
class GateConfig:
    enabled: bool = True
    threshold_db: float = -48.0  # soft expander threshold (eliminates ambient room tone/hiss in speech pauses)
    range_db: float = 20.0  # how far the gate closes (soft expander, not a hard mute)
    attack_ms: float = 3.0
    hold_ms: float = 80.0
    release_ms: float = 150.0
    hysteresis_db: float = 4.0


@dataclass
class DeEsserConfig:
    enabled: bool = True
    freq_hz: float = 7000.0
    threshold_db: float = -32.0
    ratio: float = 3.0


@dataclass
class LevelerConfig:
    enabled: bool = True
    target_db: float = -20.0
    max_gain_db: float = 9.0
    max_cut_db: float = 6.0
    gate_db: float = -48.0


@dataclass
class CompressorConfig:
    enabled: bool = True
    threshold_db: float = -22.0
    ratio: float = 3.0
    attack_ms: float = 6.0
    release_ms: float = 90.0
    knee_db: float = 6.0
    makeup_db: float = 3.0


@dataclass
class LimiterConfig:
    enabled: bool = True
    ceiling_db: float = -1.5
    release_ms: float = 60.0


@dataclass
class ChainConfig:
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    highpass_hz: float = 90.0  # 0 disables
    aec: AECConfig = field(default_factory=AECConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    deverb: DeverbConfig = field(default_factory=DeverbConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    proximity: ProximityConfig = field(default_factory=ProximityConfig)
    eq_enabled: bool = True
    eq: list[EQBand] = field(default_factory=list)
    deesser: DeEsserConfig = field(default_factory=DeEsserConfig)
    leveler: LevelerConfig = field(default_factory=LevelerConfig)
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    limiter: LimiterConfig = field(default_factory=LimiterConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChainConfig":
        cfg = _build(cls, data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(msg)

        check(-40 <= self.input_gain_db <= 40, "input_gain_db must be within -40..40")
        check(-40 <= self.output_gain_db <= 40, "output_gain_db must be within -40..40")
        check(0 <= self.highpass_hz <= 400, "highpass_hz must be within 0..400")
        check(0.0 <= self.aec.amount <= 1.0, "aec.amount must be within 0..1")
        check(0.0 <= self.deverb.amount <= 1.0, "deverb.amount must be within 0..1")
        check(0.0 <= self.proximity.amount <= 1.0, "proximity.amount must be within 0..1")
        check(0.0 <= self.denoise.strength <= 1.0, "denoise.strength must be within 0..1")
        check(0 <= self.denoise.max_reduction_db <= 60, "denoise.max_reduction_db must be within 0..60")
        check(self.denoise.model_type in ("cofilite", "spectral"), "denoise.model_type must be 'cofilite' or 'spectral'")
        check(self.gate.range_db >= 0, "gate.range_db must be >= 0")
        check(self.compressor.ratio >= 1.0, "compressor.ratio must be >= 1")
        check(self.deesser.ratio >= 1.0, "deesser.ratio must be >= 1")
        check(self.limiter.ceiling_db <= 0, "limiter.ceiling_db must be <= 0")
        valid = {"peak", "lowshelf", "highshelf", "highpass", "lowpass", "bandpass"}
        for b in self.eq:
            check(b.kind in valid, f"unknown EQ band kind {b.kind!r}")
            check(b.freq > 0 and b.q > 0, "EQ band freq and q must be > 0")


# --------------------------------------------------------------------------- #
# Application configuration
# --------------------------------------------------------------------------- #
DeviceSpec = typing.Union[str, int, None]


@dataclass
class HotkeyConfig:
    toggle_mute: str = "ctrl+shift+m"
    push_to_mute: str = ""
    push_to_talk: str = ""
    toggle_bypass: str = "ctrl+shift+b"
    play_cues: bool = True
    cue_volume: float = 0.5
    start_in_tray: bool = False


@dataclass
class AppConfig:
    sample_rate: int = 48000
    block_size: int = 480  # 10 ms @ 48 kHz
    jitter_blocks: float = 3.0  # output prefill in blocks; 3.0 = 30 ms rock-solid cushion (glitch-proof for gaming)
    input_device: DeviceSpec = None  # name substring or index; None = system default
    output_device: DeviceSpec = None  # virtual cable (Discord reads its other end)
    monitor_device: DeviceSpec = None  # your headphones, for self-monitoring
    aec_reference_device: DeviceSpec = None  # speaker/loopback capture device for AEC (None = auto-detect loopback)
    input_channel: int = 0
    preset: str = "discord"
    denoise_model: str | None = None  # custom swappable model path (.onnx or .tar)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)

    @property
    def mic(self) -> DeviceSpec:
        return self.input_device

    @mic.setter
    def mic(self, val: DeviceSpec) -> None:
        self.input_device = val

    @property
    def cable(self) -> DeviceSpec:
        return self.output_device

    @cable.setter
    def cable(self, val: DeviceSpec) -> None:
        self.output_device = val

    @staticmethod
    def path() -> Path:
        return config_dir() / "config.json"

    @classmethod
    def load(cls) -> "AppConfig":
        p = cls.path()
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            cfg = _build(cls, data)
        except (OSError, ValueError, TypeError) as exc:
            raise ConfigError(
                f"Could not read {p}: {exc}",
                hint=f"Delete {p} to reset to defaults.",
            ) from exc
        # Ensure balanced, glitch-resistant default of 3.0 blocks (30ms cushion)
        jb = getattr(cfg, "jitter_blocks", 3.0)
        if jb < 1.0 or jb > 8.0 or jb in (1.5, 2.5):
            cfg.jitter_blocks = 3.0
        cfg.validate()
        return cfg

    def save(self) -> None:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")
        tmp.replace(p)

    def validate(self) -> None:
        if self.sample_rate not in (16000, 22050, 24000, 32000, 44100, 48000, 96000):
            raise ConfigError(f"Unsupported sample_rate {self.sample_rate}", hint="Use 48000 or 44100.")
        if not 64 <= self.block_size <= 4096:
            raise ConfigError("block_size must be within 64..4096")
        if not 1.0 <= float(self.jitter_blocks) <= 8.0:
            raise ConfigError("jitter_blocks must be within 1.0..8.0")
        if self.input_channel < 0:
            raise ConfigError("input_channel must be >= 0")
