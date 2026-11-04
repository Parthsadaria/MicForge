"""Offline speech-signal statistics and plain-language diagnosis."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

_EPS = 1e-12


@dataclass(frozen=True)
class AudioStats:
    duration_s: float
    peak_db: float
    rms_db: float
    active_rms_db: float | None  # level while talking
    noise_floor_db: float
    snr_db: float | None
    clip_samples: int
    presence_db: float | None  # energy 2.5-8 kHz relative to 100-2.5 kHz (higher = brighter)
    centroid_hz: float | None
    active_pct: float


@dataclass(frozen=True)
class Finding:
    level: str  # ok | warn | bad
    text: str


def _db(p: float) -> float:
    return 10.0 * math.log10(max(p, _EPS))


def analyze(x: np.ndarray, sr: int) -> AudioStats:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < sr // 5:
        raise ValueError("recording too short to analyse")
    frame = max(1, int(0.02 * sr))
    m = n // frame
    frames = x[: m * frame].reshape(m, frame)
    fr_db = 10.0 * np.log10(np.maximum(np.mean(frames**2, axis=1), _EPS))
    noise_floor = float(np.percentile(fr_db, 10))
    loud = float(np.percentile(fr_db, 95))

    has_speech = (loud - noise_floor) >= 12.0
    active_rms = snr = presence = centroid = None
    active_pct = 0.0
    if has_speech:
        thr = max(noise_floor + 10.0, loud - 30.0)
        mask = fr_db > thr
        active_pct = 100.0 * float(np.mean(mask))
        if mask.any():
            active = frames[mask]
            active_rms = _db(float(np.mean(active**2)))
            snr = active_rms - noise_floor
            if active.size >= int(0.3 * sr):
                f, p = welch(active.reshape(-1), sr, nperseg=min(1024, active.size))
                hi_edge = min(8000.0, 0.45 * sr)
                lo = (f >= 100) & (f < 2500)
                hi = (f >= 2500) & (f < hi_edge)
                if lo.any() and hi.any():
                    presence = _db(float(p[hi].sum())) - _db(float(p[lo].sum()))
                band = (f >= 100) & (f <= hi_edge)
                if band.any() and p[band].sum() > 0:
                    centroid = float(np.sum(f[band] * p[band]) / np.sum(p[band]))

    return AudioStats(
        duration_s=n / sr,
        peak_db=_db(float(np.max(np.abs(x))) ** 2),
        rms_db=_db(float(np.mean(x**2))),
        active_rms_db=active_rms,
        noise_floor_db=noise_floor,
        snr_db=snr,
        clip_samples=int(np.count_nonzero(np.abs(x) >= 0.999)),
        presence_db=presence,
        centroid_hz=centroid,
        active_pct=active_pct,
    )


def band_levels_db(x: np.ndarray, sr: int, n_bands: int = 32, f_lo: float = 60.0, f_hi: float = 16000.0) -> np.ndarray:
    """Long-term average spectrum in log-spaced bands (dB, relative scale)."""
    f_hi = min(f_hi, 0.45 * sr)
    f, p = welch(np.asarray(x, dtype=np.float64), sr, nperseg=2048)
    edges = np.geomspace(f_lo, f_hi, n_bands + 1)
    out = np.full(n_bands, -120.0)
    for i in range(n_bands):
        sel = (f >= edges[i]) & (f < edges[i + 1])
        if sel.any():
            out[i] = _db(float(p[sel].mean()))
    return out


def clarity_label(presence_db: float | None) -> str:
    """Heuristic label; calibrated for typical close-mic speech, treat as a hint, not a measurement."""
    if presence_db is None:
        return "n/a"
    if presence_db >= -22.0:
        return "clear"
    if presence_db >= -30.0:
        return "a bit dull"
    return "muffled"


def diagnose(s: AudioStats) -> list[Finding]:
    out: list[Finding] = []
    if s.active_rms_db is None:
        out.append(Finding("warn", "Not enough speech detected - speak louder or closer to the mic for a reliable read."))
        return out
    if s.clip_samples > 0 or s.peak_db > -1.0:
        out.append(Finding("bad", "Clipping detected. Lower the mic level in system settings (and turn off mic boost)."))
    if s.active_rms_db < -35.0:
        out.append(Finding("warn", "Very quiet voice. Raise the mic level in system settings; the leveler can only add ~9 dB."))
    elif s.active_rms_db > -12.0:
        out.append(Finding("warn", "Very hot signal. Lower the mic level a little to leave headroom."))
    if s.noise_floor_db > -40.0:
        out.append(Finding("bad", "Very high noise floor. Use the `noisy_room` preset, or move the mic closer / away from fans."))
    elif s.noise_floor_db > -52.0:
        out.append(Finding("warn", "Noticeable background noise. Run `micforge calibrate` in your usual room conditions."))
    if s.snr_db is not None and s.snr_db < 20.0:
        out.append(Finding("warn", f"Low signal-to-noise ratio ({s.snr_db:.0f} dB). Closer mic placement helps more than any filter."))
    label = clarity_label(s.presence_db)
    if label == "muffled":
        out.append(Finding("bad", "Voice sounds muffled. Try the `de_muffle` preset, and check mic position (aim at your mouth, no cloth/foam blocking it)."))
    elif label == "a bit dull":
        out.append(Finding("warn", "Slightly dull top end. The `discord` or `de_muffle` preset should help."))
    if not out:
        out.append(Finding("ok", "Signal looks healthy: good level, low noise, clear top end."))
    return out
