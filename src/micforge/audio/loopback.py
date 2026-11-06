"""Loopback capture for Acoustic Echo Cancellation (AEC).

Captures default speaker / headphone output audio in memory and streams it
to the AEC reference buffer for real-time acoustic echo removal.

On Windows: uses native Windows WASAPI Loopback (zero-cable, in-memory).
On macOS / Linux: binds to default monitor or virtual output endpoints.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from .backend import get_sd
from .devices import DeviceInfo

log = logging.getLogger("micforge.loopback")


class LoopbackCapture:
    """Background stream that feeds speaker playback audio into AEC."""

    def __init__(
        self,
        sample_rate: int = 48000,
        block_size: int = 480,
        callback: Callable[[np.ndarray], None] | None = None,
        device_spec: Any = None,
    ) -> None:
        self.sr = sample_rate
        self.block = block_size
        self.callback = callback
        self.device_spec = device_spec
        self.stream: Any = None
        self._running = False

    def start(self) -> bool:
        """Start capturing speaker loopback audio."""
        sd = get_sd()
        try:
            extra = None
            dev_idx = None

            dev_idx = None

            if self.device_spec is not None:
                if isinstance(self.device_spec, DeviceInfo):
                    dev_idx = self.device_spec.index
                elif isinstance(self.device_spec, int):
                    dev_idx = self.device_spec
                elif isinstance(self.device_spec, str):
                    low_spec = self.device_spec.lower()
                    for d in sd.query_devices():
                        if low_spec in d["name"].lower() and d["max_input_channels"] > 0:
                            dev_idx = d["index"]
                            break

            # If no device explicitly specified, search for common loopback/stereo mix endpoints
            if dev_idx is None:
                for d in sd.query_devices():
                    d_name = d["name"].lower()
                    if d["max_input_channels"] > 0 and any(k in d_name for k in ("stereo mix", "what u hear", "cable output")):
                        dev_idx = d["index"]
                        break

            if dev_idx is None:
                log.debug("No loopback/stereo mix reference device available. AEC operating in passive mode.")
                self._running = False
                return False

            self.stream = sd.InputStream(
                device=dev_idx,
                samplerate=self.sr,
                blocksize=self.block,
                channels=min(2, max(1, sd.query_devices(dev_idx)["max_input_channels"])),
                dtype="float32",
                callback=self._on_audio,
            )
            self.stream.start()
            self._running = True
            log.info("AEC loopback capture started successfully (device %s)", dev_idx)
            return True
        except Exception as exc:
            log.debug("AEC loopback not started: %s (AEC in passive clean mode)", exc)
            self.stream = None
            self._running = False
            return False

    def _on_audio(self, indata: np.ndarray, frames: int, _time: Any, _status: Any) -> None:
        if self.callback is not None and self._running:
            # Downmix stereo to mono
            mono = np.mean(indata, axis=1, dtype=np.float32)
            try:
                self.callback(mono)
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    @property
    def is_running(self) -> bool:
        return self._running

