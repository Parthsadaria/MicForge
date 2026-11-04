"""WAV reading/writing with mono down-mix and resampling."""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from ..errors import MicforgeError


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def read_wav(path: str | Path, target_rate: int | None = None) -> tuple[np.ndarray, int]:
    try:
        sr, data = wavfile.read(str(path))
    except (OSError, ValueError) as exc:
        raise MicforgeError(f"Cannot read {path}: {exc}", hint="Only PCM/float WAV files are supported (convert with ffmpeg).") from exc
    if np.issubdtype(data.dtype, np.integer):
        scale = float(np.iinfo(data.dtype).max) + 1.0
        if data.dtype == np.uint8:
            data = data.astype(np.float32) - 128.0
            scale = 128.0
        x = data.astype(np.float32) / scale
    else:
        x = data.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if target_rate and target_rate != sr:
        x = resample(x, sr, target_rate)
        sr = target_rate
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, pcm)
