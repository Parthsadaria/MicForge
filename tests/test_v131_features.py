import numpy as np
import pytest

from micforge.dsp.cofilite import CoFiLiteDenoiser, find_model_file
from micforge.dsp.denoise import SpectralDenoiser
from micforge.dsp.gate import NoiseGate
from micforge.dsp.chain import build_chain
from micforge.config import ChainConfig, DenoiseConfig


def test_gate_mask_vad():
    gate = NoiseGate(sample_rate=48000, threshold_db=-30.0, range_db=40.0)
    # Whisper or low level signal below threshold (-40 dBFS)
    t = np.linspace(0, 0.1, 4800, endpoint=False, dtype=np.float32)
    sig = 0.01 * np.sin(2 * np.pi * 440.0 * t)  # ~ -40 dBFS

    # Gate without speech prob stays closed (attenuated)
    gate.speech_prob = 0.0
    out_closed = gate.process(sig)
    assert np.max(np.abs(out_closed)) < np.max(np.abs(sig)) * 0.1

    # Gate with neural speech prob > 0.45 opens up!
    gate.reset()
    gate.speech_prob = 0.85
    out_open = gate.process(sig)
    assert np.max(np.abs(out_open)) > np.max(np.abs(sig)) * 0.8


def test_spectral_denoiser_vad_and_naturalness():
    sd = SpectralDenoiser(sample_rate=16000, naturalness=0.08)
    t = np.linspace(0, 0.2, 3200, endpoint=False, dtype=np.float32)
    speech = 0.2 * np.sin(2 * np.pi * 500.0 * t)

    out = sd.process(speech)
    assert out.shape == speech.shape
    assert not np.isnan(out).any()
    # Telemetry should include speech_prob and is_speech
    tel = sd.telemetry()
    assert "speech_prob" in tel
    assert "is_speech" in tel
    assert 0.0 <= tel["speech_prob"] <= 1.0


def test_chain_naturalness_config():
    cfg = ChainConfig(denoise=DenoiseConfig(naturalness=0.10))
    chain = build_chain(cfg, sample_rate=48000)
    dn = chain.stage("denoise")
    assert getattr(dn, "naturalness", None) == 0.10

