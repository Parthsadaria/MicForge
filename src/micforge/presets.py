"""Built-in presets + user presets (JSON files in <config dir>/presets/)."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    cfg.denoise = DenoiseConfig(strength=0.8, max_reduction_db=20)
    cfg.deesser.threshold_db = -34.0
    cfg.deesser.ratio = 4.0
    return cfg


def _noisy_room() -> ChainConfig:
    cfg = _discord()
    cfg.highpass_hz = 110
    cfg.denoise = DenoiseConfig(strength=1.0, max_reduction_db=28)
    cfg.gate = GateConfig(threshold_db=-44, range_db=30, hold_ms=100)
    return cfg


def _gaming() -> ChainConfig:
    cfg = _discord()
    cfg.aec.enabled = True  # Ready to cancel game speaker bleed
    cfg.deverb.amount = 0.45
    cfg.proximity.amount = 0.55
    cfg.proximity.warmth = 0.4
    return cfg


def _broadcast() -> ChainConfig:
    cfg = _discord()
    cfg.highpass_hz = 75
    cfg.deverb.amount = 0.5
    cfg.deverb.room_decay = 0.35
    cfg.proximity.amount = 0.75  # Deep radio chest resonance
    cfg.proximity.body_hz = 115.0
    cfg.proximity.warmth = 0.5
    cfg.proximity.mud_cleanup = 0.5
    cfg.compressor.makeup_db = 2.5
    return cfg


def _studio() -> ChainConfig:
    cfg = ChainConfig(
        highpass_hz=70,
        eq=[EQBand("peak", 3500, 1.5, 0.8), EQBand("highshelf", 10000, 1.5, 0.7)],
    )
    cfg.denoise = DenoiseConfig(strength=0.25, max_reduction_db=10)
    cfg.deverb.amount = 0.25
    cfg.proximity.amount = 0.35
    cfg.gate.enabled = False
    cfg.leveler.enabled = False
    cfg.compressor.ratio = 2.0
    cfg.compressor.makeup_db = 1.5
    return cfg


def _bypass() -> ChainConfig:
    cfg = ChainConfig(highpass_hz=0)
    cfg.aec.enabled = False
    cfg.denoise.enabled = False
    cfg.deverb.enabled = False
    cfg.gate.enabled = False
    cfg.proximity.enabled = False
    cfg.eq_enabled = False
    cfg.deesser.enabled = False
    cfg.leveler.enabled = False
    cfg.compressor.enabled = False
    cfg.limiter.enabled = False
    return cfg


BUILTIN: dict[str, tuple[Any, str]] = {
    "discord": (_discord, "Balanced default: de-mud, presence boost, noise suppression, leveling."),
    "gaming": (_gaming, "Optimized for gaming: AEC echo cancellation, room de-verb, voice punch."),
    "broadcast": (_broadcast, "Rich radio presence: deep chest resonance, warmth, studio de-verb."),
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


def user_preset_dir() -> Path:
    return config_dir() / "presets"


def sanitize_preset_name(name: str) -> str:
    """Sanitize preset name to be safe for filenames and CLI arguments."""
    clean = re.sub(r"[^\w\-. ]", "_", name.strip())
    clean = clean.replace(" ", "_")
    if not clean:
        raise ConfigError("Preset name cannot be empty.")
    return clean


def _user_presets() -> dict[str, ChainConfig]:
    out: dict[str, ChainConfig] = {}
    d = user_preset_dir()
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                out[p.stem] = ChainConfig.from_dict(raw)
        except (OSError, ValueError, ConfigError):
            continue  # a broken user file must never take the app down
    return out


def list_presets() -> list[PresetInfo]:
    """List all presets (built-in and custom user presets) with descriptions."""
    infos = [PresetInfo(n, desc, True) for n, (_, desc) in BUILTIN.items()]
    user_map = _user_presets()
    d = user_preset_dir()
    for name, cfg in user_map.items():
        if name in BUILTIN:
            continue
        desc = "Custom user preset"
        json_path = d / f"{name}.json"
        if json_path.is_file():
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and "_description" in raw and raw["_description"]:
                    desc = str(raw["_description"])
            except Exception:
                pass
        infos.append(PresetInfo(name, desc, False))
    return infos


def get_preset(name: str) -> ChainConfig:
    """Retrieve preset ChainConfig by name (user preset takes priority, then built-in)."""
    user = _user_presets()
    if name in user:
        return copy.deepcopy(user[name])
    if name in BUILTIN:
        return BUILTIN[name][0]()
    raise ConfigError(
        f"Unknown preset {name!r}.",
        hint="Available: " + ", ".join(p.name for p in list_presets()),
    )


def get_preset_description(name: str) -> str:
    """Retrieve preset human-readable description by name."""
    for p in list_presets():
        if p.name == name:
            return p.description
    return ""


def save_preset(
    name: str,
    config: ChainConfig,
    overwrite: bool = False,
    description: str = "",
) -> Path:
    """Save a custom user preset to disk."""
    clean_name = sanitize_preset_name(name)
    if clean_name.lower() in {k.lower() for k in BUILTIN}:
        raise ConfigError(
            f"Cannot overwrite built-in preset '{clean_name}'. Please choose a custom name."
        )

    target = user_preset_dir() / f"{clean_name}.json"
    if target.is_file() and not overwrite:
        raise ConfigError(
            f"Preset '{clean_name}' already exists. Use overwrite=True or pass --force to replace it."
        )

    config.validate()
    data = config.to_dict()
    data["_name"] = clean_name
    data["_description"] = description or "Custom user preset"
    data["_updated_at"] = datetime.now(timezone.utc).isoformat()

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target


def delete_preset(name: str) -> None:
    """Delete a custom user preset from disk."""
    clean_name = sanitize_preset_name(name)
    if clean_name.lower() in {k.lower() for k in BUILTIN}:
        raise ConfigError(f"Cannot delete built-in preset '{clean_name}'. Built-ins are permanent.")

    target = user_preset_dir() / f"{clean_name}.json"
    if not target.is_file():
        raise ConfigError(f"User preset '{clean_name}' does not exist.")

    target.unlink()


def export_preset(name: str, target_path: str | Path) -> Path:
    """Export a preset (built-in or user) to an external JSON file."""
    cfg = get_preset(name)
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    data = cfg.to_dict()
    data["_name"] = name
    data["_exported_from"] = "micforge"
    data["_exported_at"] = datetime.now(timezone.utc).isoformat()

    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target


def import_preset(
    source_path: str | Path,
    name: str | None = None,
    overwrite: bool = False,
) -> str:
    """Import an external preset JSON file into user presets."""
    src = Path(source_path).resolve()
    if not src.is_file():
        raise ConfigError(f"Preset file not found: {src}")

    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"Invalid JSON file in {src.name}: {exc}")

    if not isinstance(raw, dict):
        raise ConfigError(f"Preset file {src.name} must contain a JSON dictionary.")

    cfg = ChainConfig.from_dict(raw)
    target_name = name or raw.get("_name") or src.stem
    desc = raw.get("_description", f"Imported from {src.name}")

    save_preset(target_name, cfg, overwrite=overwrite, description=desc)
    return sanitize_preset_name(target_name)


def clone_preset(
    source_name: str,
    new_name: str,
    overwrite: bool = False,
    description: str = "",
) -> ChainConfig:
    """Clone an existing preset (built-in or user) as a new custom preset."""
    src_cfg = get_preset(source_name)
    cloned = copy.deepcopy(src_cfg)
    desc = description or f"Cloned from {source_name}"
    save_preset(new_name, cloned, overwrite=overwrite, description=desc)
    return cloned
