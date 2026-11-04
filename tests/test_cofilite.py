import numpy as np
import pytest

from micforge.dsp.cofilite import CoFiLiteDenoiser, find_model_file
from micforge.dsp.chain import Chain, process_offline
from micforge.dsp.util import rms_db


def test_find_model_file():
    model_path = find_model_file()
    assert model_path is not None
    assert model_path.is_file()
    assert model_path.suffix.lower() == ".onnx"


def test_cofilite_init_and_properties():
    denoiser = CoFiLiteDenoiser(48000, strength=0.7, max_reduction_db=20.0)
    assert denoiser.enabled
    assert denoiser.strength == 0.7
    assert denoiser.max_reduction_db == 20.0
    assert denoiser.latency_samples > 0
    assert denoiser.model_name.endswith(".onnx")


def test_cofilite_suppresses_noise_48k():
    denoiser = CoFiLiteDenoiser(48000, strength=0.8)
    rng = np.random.default_rng(42)

    # 1.5 seconds of noise + tone burst
    t = np.arange(48000 * 2) / 48000
    noise = (rng.standard_normal(len(t)) * 0.05).astype(np.float32)
    # Speech tone active in second half
    speech = (0.2 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    speech[t < 1.0] = 0
    signal = noise + speech

    chain = Chain([denoiser])
    out = process_offline(chain, signal, block=480)

    # In the noise-only region (first second, after initial latency priming)
    noise_in_rms = rms_db(signal[4800:40000])
    noise_out_rms = rms_db(out[4800:40000])

    assert noise_out_rms < noise_in_rms - 10.0  # At least 10 dB noise suppression


def test_cofilite_strength_control():
    d_off = CoFiLiteDenoiser(48000, strength=0.0)
    d_full = CoFiLiteDenoiser(48000, strength=1.0)
    rng = np.random.default_rng(123)
    noise = (rng.standard_normal(48000) * 0.05).astype(np.float32)

    out_off = process_offline(Chain([d_off]), noise, block=480)
    out_full = process_offline(Chain([d_full]), noise, block=480)

    # Full strength should suppress significantly more noise than 0.0 strength
    assert rms_db(out_full[4800:]) < rms_db(out_off[4800:]) - 8.0


def test_cofilite_16k_native():
    denoiser = CoFiLiteDenoiser(16000, strength=0.7)
    x = (np.random.default_rng(0).standard_normal(16000) * 0.02).astype(np.float32)
    out = process_offline(Chain([denoiser]), x, block=160)
    assert out.shape == x.shape
    assert rms_db(out[3200:]) < rms_db(x[3200:])


def test_cofilite_calibration_compatibility():
    denoiser = CoFiLiteDenoiser(48000)
    noise = (np.random.default_rng(0).standard_normal(48000 * 2) * 0.02).astype(np.float32)
    psd = denoiser.learn(noise)
    assert psd.shape == (denoiser.n_bins,)
    assert denoiser.set_profile(psd) is True
    assert denoiser.set_profile(np.zeros(10)) is False
    assert denoiser.noise_floor_db() < -20.0

