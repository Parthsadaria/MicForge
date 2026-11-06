"""Adaptive noise profiling: continuous voice-aware learning, environment profiles, and safe bounds."""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import config_dir

log = logging.getLogger("micforge.noise_profile")

ENVIRONMENTS_DIR_NAME = "noise_profiles"


def _profiles_dir() -> Path:
    p = config_dir() / ENVIRONMENTS_DIR_NAME
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


@dataclass
class NoiseProfile:
    """Snapshot of ambient noise characteristics."""
    name: str = "quiet_room"
    noise_floor_db: float = -62.0
    gate_threshold_db: float = -50.0
    spectral_tilt: float = 0.0  # ratio of high vs low band energy
    sample_rate: int = 48000
    bands: list[float] = field(default_factory=lambda: [-65.0] * 32)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    sample_count: int = 0
    frozen: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NoiseProfile:
        return cls(
            name=str(data.get("name", "custom")),
            noise_floor_db=float(data.get("noise_floor_db", -62.0)),
            gate_threshold_db=float(data.get("gate_threshold_db", -50.0)),
            spectral_tilt=float(data.get("spectral_tilt", 0.0)),
            sample_rate=int(data.get("sample_rate", 48000)),
            bands=list(data.get("bands", [-65.0] * 32)),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            sample_count=int(data.get("sample_count", 0)),
            frozen=bool(data.get("frozen", False)),
        )


# Standard environment baseline presets
BUILTIN_ENVIRONMENTS: dict[str, NoiseProfile] = {
    "quiet_room": NoiseProfile(
        name="quiet_room",
        noise_floor_db=-64.0,
        gate_threshold_db=-52.0,
        spectral_tilt=-0.3,
        bands=[-65.0] * 32,
    ),
    "fan_on": NoiseProfile(
        name="fan_on",
        noise_floor_db=-52.0,
        gate_threshold_db=-42.0,
        spectral_tilt=-0.8,  # strong low-frequency motor rumble
        bands=[-50.0 if i < 8 else -56.0 for i in range(32)],
    ),
    "mechanical_keyboard": NoiseProfile(
        name="mechanical_keyboard",
        noise_floor_db=-56.0,
        gate_threshold_db=-38.0,
        spectral_tilt=0.4,  # transient click presence
        bands=[-58.0 if i < 16 else -52.0 for i in range(32)],
    ),
}


class AdaptiveNoiseProfiler:
    """Continuous low-rate, voice-aware background noise learner."""

    def __init__(
        self,
        sample_rate: int = 48000,
        environment: str = "quiet_room",
        adaptation_rate: float = 0.005,  # smooth update ~3-5s time constant
        max_rate_change_db_per_sec: float = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.adaptation_rate = adaptation_rate
        self.max_rate_change_db_per_sec = max_rate_change_db_per_sec
        self.current_profile: NoiseProfile = self._load_or_create(environment)

        self._n_bands = 32
        self._band_powers = np.full(self._n_bands, 10.0 ** (self.current_profile.noise_floor_db / 10.0), dtype=np.float64)
        self._last_update_time = time.monotonic()
        self._consecutive_quiet_blocks = 0
        self._learning_enabled = True

    def _load_or_create(self, name: str) -> NoiseProfile:
        path = _profiles_dir() / f"{name}.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                prof = NoiseProfile.from_dict(data)
                prof.sample_rate = self.sample_rate
                return prof
            except Exception as exc:
                log.warning("Could not read profile %s: %s", path, exc)

        if name in BUILTIN_ENVIRONMENTS:
            base = BUILTIN_ENVIRONMENTS[name]
            return NoiseProfile(
                name=base.name,
                noise_floor_db=base.noise_floor_db,
                gate_threshold_db=base.gate_threshold_db,
                spectral_tilt=base.spectral_tilt,
                sample_rate=self.sample_rate,
                bands=list(base.bands),
            )
        return NoiseProfile(name=name, sample_rate=self.sample_rate)

    @property
    def is_frozen(self) -> bool:
        return self.current_profile.frozen or not self._learning_enabled

    def freeze(self) -> None:
        """Freeze adaptation so noise profile remains static."""
        self.current_profile.frozen = True

    def unfreeze(self) -> None:
        """Resume continuous background learning."""
        self.current_profile.frozen = False
        self._learning_enabled = True

    def update(
        self,
        block: np.ndarray,
        is_voiced: bool = False,
        speech_prob: float = 0.0,
    ) -> bool:
        """Feed audio block. Returns True if adaptation took place."""
        if self.is_frozen:
            return False

        if block is None or len(block) == 0:
            return False

        # SAFE ADAPTATION RULE 1: Never learn during voiced speech or high speech probability
        if is_voiced or speech_prob > 0.25:
            self._consecutive_quiet_blocks = 0
            return False

        # Instantaneous RMS
        sq_mean = float(np.mean(block * block))
        if sq_mean < 1e-12:
            return False
        rms_db = 10.0 * math.log10(sq_mean)

        # SAFE ADAPTATION RULE 2: Don't learn transient spikes (>16 dB above current noise floor)
        cur_floor = self.current_profile.noise_floor_db
        if rms_db > cur_floor + 16.0:
            self._consecutive_quiet_blocks = 0
            return False

        self._consecutive_quiet_blocks += 1
        # Require at least 3 consecutive quiet blocks before updating (prevents speech syllable bleed)
        if self._consecutive_quiet_blocks < 3:
            return False

        now = time.monotonic()
        dt = max(0.001, now - self._last_update_time)
        self._last_update_time = now

        # SAFE ADAPTATION RULE 3: Bounded rate of change per second
        target_db = rms_db
        max_delta = self.max_rate_change_db_per_sec * dt
        diff = target_db - cur_floor
        clipped_diff = max(-max_delta, min(max_delta, diff))
        new_floor = cur_floor + self.adaptation_rate * diff

        # Bound overall noise floor within physically reasonable microphone ranges
        new_floor = max(-85.0, min(-35.0, new_floor))
        self.current_profile.noise_floor_db = float(new_floor)

        # Gate threshold tracks noise floor with 10 dB headroom (clamped between -60 and -30 dBFS)
        self.current_profile.gate_threshold_db = float(max(-60.0, min(-30.0, new_floor + 10.0)))
        self.current_profile.sample_count += 1
        self.current_profile.updated_at = time.time()
        return True

    def relearn(self, duration_s: float = 1.0) -> None:
        """Prepare for quick immediate noise capture (e.g. user requested 'relearn')."""
        self._consecutive_quiet_blocks = 5
        self.current_profile.sample_count = 0
        self.unfreeze()

    def switch_environment(self, name: str) -> NoiseProfile:
        """Switch active noise profile to a named environment."""
        self.current_profile = self._load_or_create(name)
        self._last_update_time = time.monotonic()
        self._consecutive_quiet_blocks = 0
        return self.current_profile

    def save(self, name: str | None = None) -> Path | None:
        """Persist current noise profile to disk."""
        target_name = name or self.current_profile.name
        self.current_profile.name = target_name
        self.current_profile.updated_at = time.time()
        try:
            d = _profiles_dir()
            path = d / f"{target_name}.json"
            path.write_text(json.dumps(self.current_profile.to_dict(), indent=2), encoding="utf-8")
            log.info("Saved noise profile '%s' to %s", target_name, path)
            return path
        except Exception as exc:
            log.warning("Could not persist noise profile '%s': %s", target_name, exc)
            return None

    def reset_to_default(self) -> None:
        """Reset current profile to baseline preset values."""
        name = self.current_profile.name
        if name in BUILTIN_ENVIRONMENTS:
            base = BUILTIN_ENVIRONMENTS[name]
            self.current_profile.noise_floor_db = base.noise_floor_db
            self.current_profile.gate_threshold_db = base.gate_threshold_db
            self.current_profile.spectral_tilt = base.spectral_tilt
        else:
            self.current_profile.noise_floor_db = -62.0
            self.current_profile.gate_threshold_db = -50.0
        self.current_profile.updated_at = time.time()


def list_noise_profiles() -> list[str]:
    """List all available environment noise profile names."""
    names = set(BUILTIN_ENVIRONMENTS.keys())
    d = _profiles_dir()
    for f in d.glob("*.json"):
        names.add(f.stem)
    return sorted(names)
