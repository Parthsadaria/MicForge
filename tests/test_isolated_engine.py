"""Tests for IsolatedAudioEngine, shared memory ring buffer, and crash watchdog."""
import time
import numpy as np
import pytest

from micforge.audio.isolated_engine import SharedAudioBuffer, IsolatedAudioEngine
from micforge.config import AppConfig
from micforge.audio.devices import DeviceInfo
from micforge.audio.engine import SinkSpec


def test_shared_audio_buffer():
    buf = SharedAudioBuffer(capacity_samples=1000, create=True)
    try:
        # Write 200 samples
        x = np.arange(200, dtype=np.float32)
        buf.write(x)
        # Read latest 100 samples
        read = buf.read_latest(100)
        np.testing.assert_allclose(read, np.arange(100, 200, dtype=np.float32))

        # Attach secondary reader from same shared memory name
        reader = SharedAudioBuffer(name=buf.name, capacity_samples=1000, create=False)
        try:
            read_sec = reader.read_latest(50)
            np.testing.assert_allclose(read_sec, np.arange(150, 200, dtype=np.float32))
        finally:
            reader.close()
    finally:
        buf.close()
        buf.unlink()


def test_isolated_engine_controls_and_snapshot(fake_audio):
    app = AppConfig()
    mic = DeviceInfo(index=0, name="Fake USB Mic", hostapi="MME", max_input=1, max_output=0, default_rate=48000.0, is_default_input=True, is_default_output=False)
    cable = DeviceInfo(index=1, name="Fake CABLE Input", hostapi="MME", max_input=0, max_output=2, default_rate=48000.0, is_default_input=False, is_default_output=True)
    sinks = [SinkSpec(name="cable", device=cable)]

    eng = IsolatedAudioEngine(app, preset="discord", input_device=mic, sinks=sinks)
    with eng:
        time.sleep(0.4)
        snap = eng.snapshot()
        assert snap is not None
        assert not snap.muted

        # Instant control
        eng.set_muted(True)
        assert eng.muted is True
        time.sleep(0.1)

        eng.toggle_bypass()
        assert eng.bypass is True
        time.sleep(0.1)

        eng.set_input_gain_db(3.0)
        time.sleep(0.1)


def test_isolated_engine_watchdog_restart(fake_audio):
    app = AppConfig()
    mic = DeviceInfo(index=0, name="Fake USB Mic", hostapi="MME", max_input=1, max_output=0, default_rate=48000.0, is_default_input=True, is_default_output=False)
    cable = DeviceInfo(index=1, name="Fake CABLE Input", hostapi="MME", max_input=0, max_output=2, default_rate=48000.0, is_default_input=False, is_default_output=True)
    sinks = [SinkSpec(name="cable", device=cable)]

    eng = IsolatedAudioEngine(app, preset="discord", input_device=mic, sinks=sinks)
    eng.start()
    try:
        time.sleep(0.3)
        assert eng._proc is not None and eng._proc.is_alive()
        old_pid = eng._proc.pid

        # Simulate crash of worker process
        eng._proc.kill()
        eng._proc.join(timeout=1.0)
        assert not eng._proc.is_alive()

        # Watchdog should detect and revive within ~0.8s
        time.sleep(0.8)
        assert eng._proc.is_alive()
        assert eng._proc.pid != old_pid
    finally:
        eng.stop()


def test_isolated_engine_dashboard_rendering(fake_audio):
    from micforge.ui.dashboard import LiveContext, render_dashboard

    app = AppConfig()
    mic = DeviceInfo(index=0, name="Fake USB Mic", hostapi="MME", max_input=1, max_output=0, default_rate=48000.0, is_default_input=True, is_default_output=False)
    cable = DeviceInfo(index=1, name="Fake CABLE Input", hostapi="MME", max_input=0, max_output=2, default_rate=48000.0, is_default_input=False, is_default_output=True)
    sinks = [SinkSpec(name="cable", device=cable)]

    eng = IsolatedAudioEngine(app, preset="discord", input_device=mic, sinks=sinks)
    with eng:
        time.sleep(0.3)
        # Verify spectrum properties
        assert isinstance(eng.spectrum_range, tuple)
        assert len(eng.spectrum_range) == 2
        assert len(eng.spectrum_centers) == 32
        assert "cable" in eng.sinks
        assert hasattr(eng.sinks["cable"], "notify_pipeline_change")

        # Test stage proxy
        aec = eng.chain.stage("aec")
        assert aec is not None
        aec.enabled = False
        time.sleep(0.05)

        # Build live context like live.py does
        ctx = LiveContext(
            title="MIC ENHANCER",
            preset="discord",
            input_name=mic.name,
            cable_name=cable.name,
            monitor_name=None,
            sample_rate=app.sample_rate,
            block_size=app.block_size,
            centers=eng.spectrum_range,
        )

        snap = eng.snapshot()
        # Ensure render_dashboard works both in normal mode and gamer HUD mode without error
        renderable_normal = render_dashboard(snap, ctx, 100, time.monotonic())
        assert renderable_normal is not None

        ctx.hud_mode = True
        renderable_hud = render_dashboard(snap, ctx, 100, time.monotonic())
        assert renderable_hud is not None


def test_isolated_engine_switch_model(fake_audio):
    from micforge.dsp.denoise import SpectralDenoiser
    from micforge.dsp.cofilite import CoFiLiteDenoiser

    app = AppConfig()
    mic = DeviceInfo(index=0, name="Fake USB Mic", hostapi="MME", max_input=1, max_output=0, default_rate=48000.0, is_default_input=True, is_default_output=False)
    cable = DeviceInfo(index=1, name="Fake CABLE Input", hostapi="MME", max_input=0, max_output=2, default_rate=48000.0, is_default_input=False, is_default_output=True)
    sinks = [SinkSpec(name="cable", device=cable)]

    eng = IsolatedAudioEngine(app, preset="discord", input_device=mic, sinks=sinks)
    with eng:
        time.sleep(0.3)
        dn = eng.chain.stage("denoise")
        actual_dn = getattr(dn, "_stage", dn)
        assert isinstance(actual_dn, CoFiLiteDenoiser)
        assert actual_dn.model_type == "cofilite"

        # 1. Swap to Spectral Wiener
        new_dn = SpectralDenoiser(app.sample_rate, 1.0, 24.0)
        eng.chain.replace_stage("denoise", new_dn)
        time.sleep(0.2)

        dn_swapped = eng.chain.stage("denoise")
        actual_swapped = getattr(dn_swapped, "_stage", dn_swapped)
        assert isinstance(actual_swapped, SpectralDenoiser)
        assert actual_swapped.model_type == "spectral"

        snap = eng.snapshot()
        assert snap.telemetry["denoise"]["model"] == "spectral"
        assert snap.telemetry["denoise"]["is_neural"] == 0.0

        # 2. Swap back to CoFi-Lite
        cofi = CoFiLiteDenoiser(app.sample_rate, 1.0, 24.0)
        eng.chain.replace_stage("denoise", cofi)
        time.sleep(0.2)

        dn_back = eng.chain.stage("denoise")
        actual_back = getattr(dn_back, "_stage", dn_back)
        assert isinstance(actual_back, CoFiLiteDenoiser)
        assert actual_back.model_type == "cofilite"

        snap2 = eng.snapshot()
        assert snap2.telemetry["denoise"]["model"] == "cofilite"
        assert snap2.telemetry["denoise"]["is_neural"] == 1.0

