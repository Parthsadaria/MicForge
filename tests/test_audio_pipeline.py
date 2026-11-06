"""Test full real-time audio pipeline processing and spectrogram metrics on test_audios/1.wav."""
import wave
from pathlib import Path
import numpy as np
import pytest

from micforge.presets import get_preset
from micforge.dsp.chain import build_chain, process_offline


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
        if wf.getsampwidth() == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            data = np.frombuffer(raw, dtype=np.float32)
        if ch > 1:
            data = data.reshape(-1, ch)[:, 0]
        return data, sr


def test_full_pipeline_on_audio_file():
    raw_path = Path("test_audios/1.wav")
    assert raw_path.is_file(), "test_audios/1.wav must exist"

    x_raw, sr = read_wav(raw_path)
    assert sr == 16000
    assert len(x_raw) > 0
    assert not np.isnan(x_raw).any()
    assert not np.isinf(x_raw).any()

    # Build default discord chain with all 11 stages
    cfg = get_preset("discord")
    chain = build_chain(cfg, sr)

    # Process in realistic 160-sample (10 ms) blocks
    block_size = 160
    lat = chain.latency_samples
    padded_in = np.concatenate((x_raw, np.zeros(lat + block_size, dtype=np.float32)))
    processed_buf = np.empty_like(padded_in)

    for i in range(0, padded_in.shape[0], block_size):
        chunk = padded_in[i : i + block_size]
        out_chunk = chain.process(chunk)
        processed_buf[i : i + len(out_chunk)] = out_chunk

    x_proc = processed_buf[lat : lat + len(x_raw)]

    # 1. Output sanity checks
    assert len(x_proc) == len(x_raw)
    assert not np.isnan(x_proc).any()
    assert not np.isinf(x_proc).any()

    # Peak must never clip above limiter ceiling (-1.5 dBFS ~ 0.841)
    peak = np.max(np.abs(x_proc))
    assert peak <= 0.85, f"Limiter must keep peak below 0.85, got {peak}"

    # 2. Check noise suppression during inter-word silence pause
    # Between 4.1s and 4.4s in 1.wav, there is an inter-sentence breathing pause
    pause_raw = x_raw[int(4.1 * sr) : int(4.4 * sr)]
    pause_proc = x_proc[int(4.1 * sr) : int(4.4 * sr)]

    pause_raw_p = np.mean(pause_raw**2) + 1e-12
    pause_proc_p = np.mean(pause_proc**2) + 1e-12
    suppression_db = 10.0 * np.log10(pause_raw_p / pause_proc_p)

    # Full pipeline with CoFi-Lite and NoiseGate must suppress pause noise by at least 15 dB (typically ~25 dB)
    assert suppression_db >= 15.0, f"Expected >= 15 dB pause noise suppression, got {suppression_db:.2f} dB"

    # 3. Check voice activity preservation
    # Total voice power must be healthy and audible
    rms_raw = np.sqrt(np.mean(x_raw**2))
    rms_proc = np.sqrt(np.mean(x_proc**2))
    assert rms_proc >= 0.03, f"Voice must not be over-attenuated, got rms={rms_proc:.4f}"


def test_pressure_cooker_noise_suppression():
    """Verify that loud broadband steam hiss / pressure cooker noise is aggressively suppressed (> 35 dB)."""
    sr = 16000
    cfg = get_preset("discord")
    chain = build_chain(cfg, sr)

    # Simulate 2 seconds of stationary pressure cooker hiss:
    # broadband white noise with high-frequency emphasis (2 kHz - 7 kHz)
    np.random.seed(42)
    noise = np.random.randn(2 * sr).astype(np.float32)
    # Target -30 dBFS RMS
    target_rms = 10.0 ** (-30.0 / 20.0)
    noise = noise * (target_rms / (np.sqrt(np.mean(noise**2)) + 1e-12))

    out = process_offline(chain, noise)
    # Measure steady-state suppression after initial convergence (skip first 0.5s)
    ss_noise = noise[int(0.5 * sr) :]
    ss_out = out[int(0.5 * sr) :]

    rms_in = np.sqrt(np.mean(ss_noise**2))
    rms_out = np.sqrt(np.mean(ss_out**2))
    suppression_db = 20.0 * np.log10(rms_in / (rms_out + 1e-12))

    assert suppression_db >= 35.0, f"Expected >= 35 dB suppression on steam hiss, got {suppression_db:.2f} dB"
    assert rms_out < 0.001, f"Residual hiss must be below -60 dBFS, got {20 * np.log10(rms_out):.2f} dBFS"


def test_docs_audio_and_images_exist():
    # Verify generated documentation assets
    assert Path("docs/audio/1_raw.wav").is_file()
    assert Path("docs/audio/1_processed.wav").is_file()
    assert Path("docs/images/spectrogram_comparison.png").is_file()
    assert Path("docs/images/spectrogram_raw.png").is_file()
    assert Path("docs/images/spectrogram_processed.png").is_file()
