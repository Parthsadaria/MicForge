"""Simple microphone recorder + file-style player used by `test` and `calibrate`."""
from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np

from ..errors import DeviceError
from .backend import get_sd
from .devices import DeviceInfo, extra_settings
from .metrics import LevelMeter, LevelSnapshot


class Recorder:
    def __init__(self, device: DeviceInfo, sample_rate: int, channel: int = 0, block: int = 480) -> None:
        self.device, self.sr, self.channel, self.block = device, sample_rate, channel, block
        self.meter = LevelMeter(sample_rate)
        self._chunks: list[np.ndarray] = []
        self._n = 0
        self._stream: Any = None

    def start(self) -> None:
        sd = get_sd()
        try:
            self._stream = sd.InputStream(
                device=self.device.index,
                samplerate=self.sr,
                blocksize=self.block,
                channels=max(1, min(self.device.max_input, self.channel + 1)),
                dtype="float32",
                callback=self._cb,
                extra_settings=extra_settings(self.device, "input"),
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise DeviceError(f"Cannot open microphone '{self.device.name}': {exc}") from exc

    def _cb(self, indata: np.ndarray, frames: int, _t: Any, _s: Any) -> None:
        x = indata[:, self.channel].copy()
        self._chunks.append(x)
        self._n += frames
        self.meter.update(x)

    @property
    def seconds(self) -> float:
        return self._n / self.sr

    @property
    def level(self) -> LevelSnapshot:
        return self.meter.snapshot

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)


def play(
    x: np.ndarray,
    sample_rate: int,
    device: DeviceInfo,
    on_tick: Callable[[float], None] | None = None,
) -> None:
    """Blocking playback. Ctrl+C stops it cleanly."""
    sd = get_sd()
    data = np.ascontiguousarray(x, dtype=np.float32)
    channels = max(1, min(2, device.max_output))
    pos = [0]
    done = threading.Event()

    def cb(outdata: np.ndarray, frames: int, _t: Any, _s: Any) -> None:
        chunk = data[pos[0] : pos[0] + frames]
        outdata[: chunk.shape[0]] = chunk[:, None]
        if chunk.shape[0] < frames:
            outdata[chunk.shape[0] :] = 0
            raise sd.CallbackStop
        pos[0] += frames

    try:
        stream = sd.OutputStream(
            device=device.index,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=cb,
            finished_callback=done.set,
            extra_settings=extra_settings(device, "output"),
        )
    except Exception as exc:  # noqa: BLE001
        raise DeviceError(f"Cannot open output '{device.name}': {exc}") from exc
    with stream:
        try:
            while not done.wait(0.05):
                if on_tick:
                    on_tick(pos[0] / sample_rate)
        except KeyboardInterrupt:
            pass
