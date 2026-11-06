"""Tests for AcousticEchoCanceller (AEC) module."""
import numpy as np
import pytest

from micforge.dsp.aec import AcousticEchoCanceller, AECDelayEstimator, AECDoubleTalkDetector


def test_aec_init():
    aec = AcousticEchoCanceller(48000, amount=0.9, filter_length_ms=80.0, delay_ms=30.0, suppression_db=20.0)
    assert aec.amount == 0.9
    assert aec.filter_length_ms == 80.0
    assert aec.delay_ms == 30.0
    assert aec.suppression_db == 20.0
    assert aec.latency_samples == 128


def test_aec_passive_passthrough_when_no_reference():
    aec = AcousticEchoCanceller(48000, amount=1.0)
    x = np.random.randn(480).astype(np.float32) * 0.1
    # Without feeding reference, AEC must pass audio through cleanly with 0 latency
    y = aec.process(x)
    np.testing.assert_array_equal(x, y)
    assert aec.telemetry()["erle_db"] == 0.0


def test_aec_block_invariance():
    aec = AcousticEchoCanceller(48000, amount=1.0)
    ref = np.random.randn(480).astype(np.float32) * 0.1
    aec.feed_reference(ref)

    for block_size in (128, 240, 480):
        mic = np.random.randn(block_size).astype(np.float32) * 0.05
        y = aec.process(mic)
        assert y.shape == mic.shape
        assert np.all(np.isfinite(y))


def test_aec_cancels_echo():
    sr = 48000
    delay_ms = 20.0
    aec = AcousticEchoCanceller(sr, amount=1.0, filter_length_ms=60.0, delay_ms=delay_ms, suppression_db=24.0)

    delay_samples = int(delay_ms * 1e-3 * sr)
    total_samples = 48000  # 1 second of audio

    t = np.arange(total_samples) / float(sr)
    # Speaker reference signal: mix of tones
    ref_signal = (0.25 * np.sin(2 * np.pi * 320 * t) + 0.15 * np.sin(2 * np.pi * 640 * t)).astype(np.float32)

    # Microphone picks up delayed attenuated echo
    mic_signal = np.zeros_like(ref_signal)
    mic_signal[delay_samples:] = 0.5 * ref_signal[:-delay_samples]

    outputs = []
    block = 480
    for i in range(0, total_samples, block):
        aec.feed_reference(ref_signal[i : i + block])
        out = aec.process(mic_signal[i : i + block])
        outputs.append(out)

    out_signal = np.concatenate(outputs)

    # After initial convergence (>24000 samples / 0.5s), echo should be significantly cancelled
    in_echo_rms = float(np.sqrt(np.mean(mic_signal[30000:] ** 2)))
    out_echo_rms = float(np.sqrt(np.mean(out_signal[30000:] ** 2)))

    assert out_echo_rms < in_echo_rms * 0.6, (
        f"AEC failed to cancel echo: output RMS {out_echo_rms:.5f} vs input RMS {in_echo_rms:.5f}"
    )


def test_aec_double_talk_detector():
    dtd = AECDoubleTalkDetector()
    # When mic has loud near-end speech compared to reference playback, DTD must be active
    assert dtd.check(mic_rms=0.1, ref_rms=0.01, err_rms=0.1) is True

    # When mic only contains echo (mic <= reference), DTD is inactive
    dtd._hangover = 0
    assert dtd.check(mic_rms=0.02, ref_rms=0.05, err_rms=0.02) is False


def test_aec_reset():
    aec = AcousticEchoCanceller(48000)
    ref = np.ones(480, dtype=np.float32)
    aec.feed_reference(ref)
    aec.process(ref)
    assert aec._ref_available is True

    aec.reset()
    assert aec._ref_available is False
    assert aec._in_mic_len == 0
    assert aec._out_len == 0
