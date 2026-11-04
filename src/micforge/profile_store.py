"""Persist the calibrated room-noise profile."""
from __future__ import annotations

import logging

import numpy as np

from .config import config_dir

log = logging.getLogger("micforge.profile")


def _path(sample_rate: int):
    return config_dir() / f"noise_profile_{sample_rate}.npz"


def save_profile(psd: np.ndarray, sample_rate: int, noise_db: float, gate_threshold_db: float) -> None:
    p = _path(sample_rate)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, psd=psd, sample_rate=sample_rate, noise_db=noise_db, gate_threshold_db=gate_threshold_db)


def load_profile(sample_rate: int) -> tuple[np.ndarray, float, float] | None:
    """Returns (psd, noise_db, gate_threshold_db) or None."""
    p = _path(sample_rate)
    if not p.exists():
        return None
    try:
        with np.load(p) as z:
            return z["psd"].astype(np.float64), float(z["noise_db"]), float(z["gate_threshold_db"])
    except (OSError, ValueError, KeyError):
        log.warning("ignoring unreadable noise profile %s", p)
        return None


def clear_profile(sample_rate: int) -> bool:
    p = _path(sample_rate)
    if p.exists():
        p.unlink()
        return True
    return False
