"""Procedural audio feedback chimes (blip for mute, chirp for unmute).

Synthesized dynamically in NumPy with zero external sound files and zero disk I/O.
Mixed directly into the headphone/monitor sink so Discord/callers never hear them.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

log = logging.getLogger("micforge.cues")


class CuePlayer:
    """Thread-safe procedural audio cue synthesizer and mixer."""

    def __init__(self, sample_rate: int = 48000, volume: float = 0.5, enabled: bool = True) -> None:
        self.sr = sample_rate
        self.volume = max(0.0, min(1.0, float(volume)))
        self.enabled = enabled
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._cache: dict[str, np.ndarray] = {}
        self._pregenerate()

    def set_volume(self, vol: float) -> None:
        self.volume = max(0.0, min(1.0, float(vol)))

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)

    def _pregenerate(self) -> None:
        """Pre-render chime waveforms into cache to avoid any runtime allocation."""
        sr = self.sr
        # 1. Blip (Mute): gentle descending chirp 460 Hz -> 280 Hz over 70ms
        dur = 0.075
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        freq = np.linspace(460, 280, n)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        env = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
        blip = (0.28 * np.sin(phase) * env).astype(np.float32)

        # 2. Chirp (Unmute): gentle ascending chirp 280 Hz -> 540 Hz over 70ms
        freq_up = np.linspace(280, 540, n)
        phase_up = 2 * np.pi * np.cumsum(freq_up) / sr
        chirp = (0.28 * np.sin(phase_up) * env).astype(np.float32)

        # 3. Toggle (Bypass): subtle double-pip
        pip_dur = 0.02
        pip_n = int(sr * pip_dur)
        pip_t = np.linspace(0, pip_dur, pip_n, endpoint=False)
        pip_env = np.sin(np.pi * np.linspace(0, 1, pip_n)) ** 2
        pip = 0.22 * np.sin(2 * np.pi * 580 * pip_t) * pip_env
        gap = np.zeros(int(sr * 0.025), dtype=np.float32)
        toggle = np.concatenate([pip, gap, pip]).astype(np.float32)

        self._cache = {
            "mute": blip,
            "unmute": chirp,
            "toggle": toggle,
        }

    def play(self, kind: str) -> None:
        """Queue a chime to be mixed into the next outgoing headphone blocks."""
        if not self.enabled or self.volume <= 0.0:
            return
        wave = self._cache.get(kind)
        if wave is None:
            return
        scaled = wave * np.float32(self.volume)
        with self._lock:
            # Overwrite or append seamlessly
            self._buf = scaled.copy()

    def mix_into(self, out: np.ndarray) -> np.ndarray:
        """Mix queued chime samples directly into headphone/monitor audio block."""
        if not self.enabled or self._buf.size == 0:
            return out
        with self._lock:
            if self._buf.size == 0:
                return out
            n = min(out.shape[0], self._buf.shape[0])
            res = out.copy()
            res[:n] += self._buf[:n]
            self._buf = self._buf[n:]
            return res

    def play_standalone(self, kind: str, device: Any = None) -> None:
        """Play a cue directly through sounddevice if headphone monitoring is not active."""
        if not self.enabled or self.volume <= 0.0:
            return
        wave = self._cache.get(kind)
        if wave is None:
            return
        scaled = wave * np.float32(self.volume)

        def _worker():
            try:
                import sounddevice as sd

                sd.play(scaled, samplerate=self.sr, device=device, blocking=False)
            except Exception as exc:  # noqa: BLE001
                log.debug("Standalone cue playback failed: %s", exc)

        threading.Thread(target=_worker, daemon=True).start()

