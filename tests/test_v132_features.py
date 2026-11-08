"""Tests for Micforge v1.3.2:
- Zero speech onset clipping (consonant preservation via gate lookahead & dual-trigger)
- Anti-burst latency ceiling & clean underrun recovery (prevents Discord WebRTC 2x speed catch-up)
- Universal sample rate support in SpectralRack (48k, 32k, 16k, 44.1k)
- Seamless 16 kHz STFT grid alignment across model toggling
"""
import numpy as np
import pytest

from micforge.dsp.gate import NoiseGate
from micforge.dsp.spectral_rack import SpectralRack
from micforge.dsp.aec import AcousticEchoCanceller
from micforge.dsp.deverb import RoomDereverberation
from micforge.dsp.cofilite import CoFiLiteDenoiser
from micforge.dsp.denoise import SpectralDenoiser
from micforge.audio.engine import OutputSink, SinkSpec
from micforge.audio.devices import DeviceInfo
from micforge.audio.ring import RingBuffer


def test_gate_speech_onset_consonant_preservation():
    """Verify that an initial short consonant burst following silence is NOT clipped."""
    sr = 48000
    gate = NoiseGate(sample_rate=sr, threshold_db=-45.0, range_db=30.0, lookahead_ms=10.0)

    # 1. Feed 400 ms of background silence (-60 dBFS)
    silence = (np.random.randn(sr * 4 // 10) * 1e-4).astype(np.float32)
    gate.speech_prob = 0.0
    out_silence = gate.process(silence)
    assert not gate.is_open, "Gate must be closed during silence"

    # 2. Simulate speech onset: a 25 ms unvoiced consonant burst (/t/ or /p/ at -25 dBFS)
    # During unvoiced consonant onset, raw mask is modest (~0.22) but energy is high
    consonant_len = int(0.025 * sr)  # 25 ms
    t = np.linspace(0, 0.025, consonant_len, endpoint=False, dtype=np.float32)
    consonant = 0.05 * np.sin(2 * np.pi * 3000.0 * t)  # ~ -26 dBFS

    # Process consonant block by block (10 ms blocks = 480 samples)
    gate.speech_prob = 0.22  # Typical consonant mask VAD
    block = 480
    consonant_padded = np.pad(consonant, (0, block - (len(consonant) % block)))
    outputs = []
    for i in range(0, len(consonant_padded), block):
        outputs.append(gate.process(consonant_padded[i : i + block]))
    out_consonant = np.concatenate(outputs)[:consonant_len]

    # The gate must detect speech onset and preserve the consonant without eating it
    # Without lookahead, the consonant is attenuated by > 20 dB.
    # With lookahead & dual-trigger, peak output must be > 80% of original consonant peak.
    assert gate.is_open, "Gate must open on speech onset"
    assert np.max(np.abs(out_consonant)) > np.max(np.abs(consonant)) * 0.80, (
        f"Consonant was clipped: max in={np.max(np.abs(consonant)):.4f}, max out={np.max(np.abs(out_consonant)):.4f}"
    )


def test_output_sink_underrun_rearm_and_cushion():
    """Verify that an underrun rearms prefill cushion without runaway buffer bloat."""
    dev = DeviceInfo(
        index=0, name="Fake Sink", hostapi="MME", max_input=0, max_output=2,
        default_rate=48000.0, is_default_input=False, is_default_output=True,
    )
    spec = SinkSpec(name="test_sink", device=dev)
    sink = OutputSink(spec, sample_rate=48000, block_size=480, prefill_blocks=1.5)

    # Prime sink with initial block
    sink.push(np.zeros(480, dtype=np.float32))
    assert sink._active

    # Force an underrun by requesting audio when ring is nearly empty
    sink.ring.clear()
    outdata = np.zeros((480, 2), dtype=np.float32)
    sink._callback(outdata, 480, None, None)

    assert sink.underruns == 1
    assert sink.ring.fill == sink.prefill


def test_output_sink_latency_ceiling_prevents_webrtc_accelerate():
    """Verify that bursts of delayed blocks never inflate ring buffer fill beyond latency ceiling."""
    dev = DeviceInfo(
        index=0, name="Fake Sink", hostapi="MME", max_input=0, max_output=2,
        default_rate=48000.0, is_default_input=False, is_default_output=True,
    )
    spec = SinkSpec(name="test_sink", device=dev)
    sink = OutputSink(spec, sample_rate=48000, block_size=480, prefill_blocks=1.5)

    # Initial block
    sink.push(np.zeros(480, dtype=np.float32))

    # Simulate DSP stall where 10 blocks (100 ms) arrive in quick succession
    for _ in range(10):
        sink.push(np.ones(480, dtype=np.float32) * 0.1)

    # Fill must be bounded by max_cushion (target_fill + 1.0 block = ~1200 samples = 25 ms)
    # rather than accumulating 4800 samples (100 ms delay), which triggers WebRTC 2x speed catch-up.
    max_allowed = int(sink.prefill + 1.0 * sink.block)
    assert sink.ring.fill <= max_allowed + 480, (
        f"Ring fill {sink.ring.fill} exceeded latency ceiling {max_allowed + 480}"
    )


def test_spectral_rack_universal_sample_rates():
    """Verify SpectralRack operates seamlessly at 48k, 32k, 16k, and 44.1k without speed distortion."""
    for sr in [48000, 32000, 16000, 44100]:
        aec = AcousticEchoCanceller(sr, enabled=False)
        dn = SpectralDenoiser(sr, enabled=True)
        dv = RoomDereverberation(sr, enabled=False)
        rack = SpectralRack(sr, aec, dn, dv, enabled=True)

        block_len = int(round(0.010 * sr))  # 10 ms
        x = (np.random.randn(block_len) * 0.1).astype(np.float32)

        # Process 5 blocks
        for _ in range(5):
            y = rack.process(x)
            assert y.shape == x.shape, f"Failed shape match at {sr} Hz: got {y.shape}, expected {x.shape}"
            assert not np.isnan(y).any()


def test_spectral_rack_model_swap_grid_alignment():
    """Verify swapping from CoFi-Lite to Spectral Wiener and back preserves 16 kHz grid alignment."""
    sr = 48000
    aec = AcousticEchoCanceller(sr, enabled=False)
    cofi = CoFiLiteDenoiser(sr, enabled=True)
    dv = RoomDereverberation(sr, enabled=False)
    rack = SpectralRack(sr, aec, cofi, dv, enabled=True)

    assert rack.grid_sr == 16000
    assert rack.hop == 256
    assert rack.n_bins == 257

    # Swap to SpectralDenoiser
    spec_dn = SpectralDenoiser(sr, enabled=True)
    rack.replace_sub_stage("denoise", spec_dn)

    assert spec_dn.sample_rate == 16000
    assert spec_dn.n_bins == 257
    assert spec_dn.hop == 256

    # Swap back to CoFiLite
    rack.replace_sub_stage("denoise", cofi)
    assert rack.denoise is cofi
