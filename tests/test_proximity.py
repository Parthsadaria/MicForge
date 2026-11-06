"""Tests for VoiceProximity chest resonance and warmth synthesizer."""
import numpy as np
import pytest

from micforge.dsp.proximity import VoiceProximity


def test_proximity_init_and_params():
    p = VoiceProximity(48000, amount=0.6, body_hz=110.0, warmth=0.4, mud_cleanup=0.5)
    assert p.amount == 0.6
    assert p.body_hz == 110.0
    assert p.warmth == 0.4
    assert p.mud_cleanup == 0.5
    assert p.latency_samples == 0


def test_proximity_block_invariance():
    p = VoiceProximity(48000, amount=0.5)
    for block_size in (128, 240, 480, 960):
        x = np.random.randn(block_size).astype(np.float32) * 0.1
        y = p.process(x)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


def test_proximity_boost_fundamental():
    sr = 48000
    p = VoiceProximity(sr, amount=0.8, body_hz=120.0, warmth=0.0, mud_cleanup=0.0)
    t = np.arange(4800) / float(sr)
    tone = (0.1 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)

    # Process blocks
    outputs = []
    for i in range(0, len(tone), 480):
        outputs.append(p.process(tone[i : i + 480]))
    out = np.concatenate(outputs)

    # Steady-state check: output at 120 Hz should be louder than input
    in_rms = np.sqrt(np.mean(tone[2400:] ** 2))
    out_rms = np.sqrt(np.mean(out[2400:] ** 2))
    assert out_rms > in_rms * 1.25, f"Expected boost, got out_rms={out_rms} vs in_rms={in_rms}"


def test_proximity_mud_cleanup():
    sr = 48000
    # Enable mud cleanup at 300 Hz
    p = VoiceProximity(sr, amount=0.8, body_hz=120.0, warmth=0.0, mud_cleanup=0.8)
    t = np.arange(4800) / float(sr)
    mud_tone = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    outputs = []
    for i in range(0, len(mud_tone), 480):
        outputs.append(p.process(mud_tone[i : i + 480]))
    out = np.concatenate(outputs)

    in_rms = np.sqrt(np.mean(mud_tone[2400:] ** 2))
    out_rms = np.sqrt(np.mean(out[2400:] ** 2))
    assert out_rms < in_rms * 1.05, "Mud frequency should be controlled / attenuated"


def test_proximity_silence_stability():
    p = VoiceProximity(48000, amount=0.8)
    silence = np.zeros(480, dtype=np.float32)
    for _ in range(50):
        y = p.process(silence)
        assert np.max(np.abs(y)) < 1e-6
    tel = p.telemetry()
    assert tel["boost_db"] == 0.0


def test_proximity_bypass_and_reset():
    p = VoiceProximity(48000, amount=0.7, enabled=False)
    x = np.ones(480, dtype=np.float32) * 0.2
    y = p.process(x)
    np.testing.assert_array_equal(x, y)

    p.enabled = True
    p.process(x)
    p.reset()
    assert p._env == 0.0

