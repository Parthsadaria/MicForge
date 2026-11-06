"""Tests for RoomDereverberation processor."""
import numpy as np
import pytest

from micforge.dsp.deverb import RoomDereverberation


def test_deverb_init():
    d = RoomDereverberation(48000, amount=0.7, room_decay=0.5, early_retention=0.75)
    assert d.amount == 0.7
    assert d.room_decay == 0.5
    assert d.early_retention == 0.75
    assert d.latency_samples == 128


def test_deverb_block_invariance():
    d = RoomDereverberation(48000, amount=0.5)
    for block_size in (128, 240, 480):
        x = np.random.randn(block_size).astype(np.float32) * 0.05
        y = d.process(x)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


def test_deverb_attenuates_late_tail():
    sr = 48000
    d = RoomDereverberation(sr, amount=0.9, room_decay=0.4)

    # Create artificial sound: speech burst followed by 0.5s decaying reverberation tail
    t = np.arange(9600) / float(sr)
    # Direct sound in first 2400 samples
    direct = np.zeros(9600, dtype=np.float32)
    direct[:2400] = 0.3 * np.sin(2 * np.pi * 350 * t[:2400])
    # Exponential reverb tail convolved into the signal
    decay = np.exp(-t * 6.0).astype(np.float32)
    reverb_signal = direct + 0.15 * decay * np.sin(2 * np.pi * 350 * t)

    outputs = []
    for i in range(0, len(reverb_signal), 480):
        outputs.append(d.process(reverb_signal[i : i + 480]))
    out = np.concatenate(outputs)

    # Check tail energy (samples 4800 to 8000)
    in_tail_rms = float(np.sqrt(np.mean(reverb_signal[4800:8000] ** 2)))
    out_tail_rms = float(np.sqrt(np.mean(out[4800:8000] ** 2)))

    # Output tail must be lower than input tail
    assert out_tail_rms < in_tail_rms * 0.75, f"Expected tail reduction: {out_tail_rms:.5f} vs {in_tail_rms:.5f}"
    tel = d.telemetry()
    assert "reverb_reduction_db" in tel


def test_deverb_silence_stability():
    d = RoomDereverberation(48000, amount=0.8)
    silence = np.zeros(480, dtype=np.float32)
    for _ in range(30):
        y = d.process(silence)
        assert np.max(np.abs(y)) < 1e-5


def test_deverb_bypass_and_reset():
    d = RoomDereverberation(48000, amount=0.6, enabled=False)
    x = np.full(480, 0.1, dtype=np.float32)
    y = d.process(x)
    np.testing.assert_array_equal(x, y)

    d.enabled = True
    d.process(x)
    d.reset()
    assert d._in_len == 0
    assert d._out_len == 0

