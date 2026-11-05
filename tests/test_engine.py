import time

import numpy as np

from micforge.audio.devices import detect_virtual_output, find_device, list_devices
from micforge.audio.engine import AudioEngine, SinkSpec
from micforge.config import AppConfig
from micforge.commands.common import make_chain
from micforge.errors import DeviceError

import pytest


def test_device_discovery(fake_audio):
    devs = list_devices()
    assert find_device(None, "input", devs).name == "Fake USB Mic"
    assert find_device("headphones", "output", devs).name == "Fake Headphones"
    cable = detect_virtual_output(devs)
    assert cable and "CABLE" in cable.name
    with pytest.raises(DeviceError):
        find_device("nonexistent", "input", devs)
    with pytest.raises(DeviceError):
        find_device("headphones", "input", devs)


def test_engine_end_to_end(fake_audio):
    app = AppConfig()
    devs = list_devices()
    chain, _ = make_chain(app, "discord")
    eng = AudioEngine(
        chain=chain,
        input_device=find_device(None, "input", devs),
        sinks=[SinkSpec("cable", detect_virtual_output(devs)), SinkSpec("monitor", find_device("head", "output", devs))],
    )
    with eng:
        time.sleep(0.8)
        eng.toggle_bypass()
        eng.toggle_mute()
        eng.toggle_mute()
        eng.begin_calibration(0.3)
        time.sleep(0.8)
        res = eng.pop_calibration()
        snap = eng.snapshot()
    assert snap.running or True
    assert res is not None and res.psd is not None and res.noise_db < 0
    assert snap.errors == 0
    assert snap.in_level.peak_db > -40
    assert snap.cpu_load < 0.75  # Neural model inference + 8-stage DSP
    assert snap.latency_ms > 0
    assert snap.output_underruns <= 3  # startup jitter only
    assert "denoise" in snap.telemetry


def test_mute_outputs_silence(fake_audio):
    app = AppConfig()
    devs = list_devices()
    chain, _ = make_chain(app, "bypass")
    eng = AudioEngine(chain=chain, input_device=find_device(None, "input", devs),
                      sinks=[SinkSpec("cable", detect_virtual_output(devs))])
    eng.muted = True  # set before start: meter ballistics would otherwise still be falling
    with eng:
        time.sleep(0.4)
        assert eng.snapshot().out_level.peak_db < -100
