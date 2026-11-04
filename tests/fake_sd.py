"""Minimal in-memory stand-in for `sounddevice`: streams run callbacks on timer threads."""
from __future__ import annotations

import threading
import time
import types

import numpy as np


class CallbackStop(Exception):
    pass


class _Status:
    input_overflow = False


class _Stream:
    def __init__(self, kind, device, samplerate, blocksize, channels, callback, finished_callback=None, **_):
        self.kind, self.sr, self.bs, self.ch, self.cb = kind, samplerate, blocksize or 480, channels, callback
        self.finished = finished_callback
        self.latency = 0.01
        self._run = False
        self._t = None
        self.frames_done = 0

    def _loop(self):
        t = np.arange(self.bs)
        n = 0
        nxt = time.perf_counter()
        rng = np.random.default_rng(0)
        while self._run:
            if self.kind == "in":
                tt = (t + n * self.bs) / self.sr
                tone = 0.2 * np.sin(2 * np.pi * 220 * tt) * (np.sin(2 * np.pi * 1.5 * tt) > 0)
                data = (tone + rng.standard_normal(self.bs) * 0.003).astype(np.float32)[:, None]
                data = np.repeat(data, self.ch, axis=1)
                self.cb(data, self.bs, None, _Status())
            else:
                out = np.zeros((self.bs, self.ch), dtype=np.float32)
                try:
                    self.cb(out, self.bs, None, _Status())
                except CallbackStop:
                    self._run = False
                    if self.finished:
                        self.finished()
                    return
            self.frames_done += self.bs
            n += 1
            nxt += self.bs / self.sr
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)

    def start(self):
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._run = False
        if self._t:
            self._t.join(timeout=1)

    def close(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.close()


def make_fake_sd():
    m = types.ModuleType("fake_sounddevice")
    m.CallbackStop = CallbackStop
    m.default = types.SimpleNamespace(device=(0, 1))
    devices = [
        {"name": "Fake USB Mic", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000.0},
        {"name": "Fake Headphones", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
    ]
    m.query_devices = lambda: devices
    m.query_hostapis = lambda: [{"name": "Fake API"}]
    m.get_portaudio_version = lambda: (1, "fake portaudio")
    m.check_input_settings = lambda **k: None
    m.InputStream = lambda **k: _Stream("in", k.get("device"), k["samplerate"], k.get("blocksize"), k["channels"], k["callback"])
    m.OutputStream = lambda **k: _Stream("out", k.get("device"), k["samplerate"], k.get("blocksize") or 480, k["channels"], k["callback"], k.get("finished_callback"))
    return m
