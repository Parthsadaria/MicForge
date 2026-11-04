"""Built-in presets + user presets (JSON files in <config dir>/presets/)."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

from .config import ChainConfig, DenoiseConfig, EQBand, GateConfig, config_dir
from .errors import ConfigError


def _discord() -> ChainConfig:
    return ChainConfig(
        highpass_hz=90,
        eq=[
            EQBand("peak", 300, -2.5, 1.0),  # clear boxy "mud"
            EQBand("peak", 3200, 3.0, 0.9),  # presence / intelligibility
            EQBand("highshelf", 9000, 3.0, 0.7),  # air
        ],
    )


def _de_muffle() -> ChainConfig:
    cfg = _discord()
    cfg.eq = [
        EQBand("peak", 250, -3.5, 1.0),
        EQBand("peak", 2800, 4.5, 0.8),
        EQBand("peak", 5000, 3.0, 1.0),
        EQBand("highshelf", 8500, 5.0, 0.7),
    ]
    cfg.denoise = DenoiseConfig(strength=0.5, max_reduction_db=16)
    cfg.deesser.threshold_db = -34.0
    cfg.deesser.ratio = 4.0
    return cfg


def _noisy_room() -> ChainConfig:
    cfg = _discord()
    cfg.highpass_hz = 110
    cfg.denoise = DenoiseConfig(strength=0.85, max_reduction_db=28)
    cfg.gate = GateConfig(threshold_db=-46, range_db=30, hold_ms=100)
    return cfg


def _studio() -> ChainConfig:
    cfg = ChainConfig(
        highpass_hz=70,
        eq=[EQBand("peak", 3500, 1.5, 0.8), EQBand("highshelf", 10000, 1.5, 0.7)],
    )
    cfg.denoise = DenoiseConfig(strength=0.25, max_reduction_db=10)
    cfg.gate.enabled = False
    cfg.leveler.enabled = False
    cfg.compressor.ratio = 2.0
    cfg.compressor.makeup_db = 1.5
    return cfg


def _bypass() -> ChainConfig:
    cfg = ChainConfig(highpass_hz=0)
    cfg.denoise.enabled = False
    cfg.gate.enabled = False
    cfg.eq_enabled = False
    cfg.deesser.enabled = False
    cfg.leveler.enabled = False
    cfg.compressor.enabled = False
    cfg.limiter.enabled = False
    return cfg


BUILTIN = {
    "discord": (_discord, "Balanced default: de-mud, presence boost, noise suppression, leveling."),
    "de_muffle": (_de_muffle, "Stronger brightening for dull/muffled mics (headset, bad placement)."),
    "noisy_room": (_noisy_room, "Aggressive denoise + gate for fans, keyboards, thin walls."),
    "studio": (_studio, "Light touch for good mics in quiet rooms."),
    "bypass": (_bypass, "Everything off. Useful for A/B and routing checks."),
}


@dataclass(frozen=True)
class PresetInfo:
    name: str
    description: str
    builtin: bool


def user_preset_dir():
    return config_dir() / "presets"


def _user_presets() -> dict[str, ChainConfig]:
    out: dict[str, ChainConfig] = {}
    d = user_preset_dir()
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            out[p.stem] = ChainConfig.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError, ConfigError):
            continue  # a broken user file must never take the app down
    return out


def list_presets() -> list[PresetInfo]:
    infos = [PresetInfo(n, desc, True) for n, (_, desc) in BUILTIN.items()]
    infos += [PresetInfo(n, "User preset", False) for n in _user_presets() if n not in BUILTIN]
    return infos


def get_preset(name: str) -> ChainConfig:
    user = _user_presets()
    if name in user:
        return copy.deepcopy(user[name])
    if name in BUILTIN:
        return BUILTIN[name][0]()
    raise ConfigError(
        f"Unknown preset {name!r}.",
        hint="Available: " + ", ".join(p.name for p in list_presets()),
    )
