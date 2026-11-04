import json

import numpy as np
import pytest

from micforge.analysis.stats import analyze, clarity_label, diagnose
from micforge.audio.fileio import read_wav, write_wav
from micforge.audio.metrics import LevelMeter, SpectrumAnalyzer
from micforge.audio.ring import RingBuffer
from micforge.config import AppConfig, ChainConfig, EQBand
from micforge.errors import ConfigError
from micforge.presets import get_preset, list_presets, user_preset_dir
from micforge.profile_store import load_profile, save_profile

SR = 48000


def test_chain_config_roundtrip():
    cfg = get_preset("de_muffle")
    again = ChainConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg and isinstance(again.eq[0], EQBand)


def test_config_validation_and_unknown_keys():
    with pytest.raises(ConfigError):
        ChainConfig.from_dict({"compressor": {"ratio": 0.5}})
    assert ChainConfig.from_dict({"nonsense": 1, "highpass_hz": 100}).highpass_hz == 100


def test_app_config_save_load(tmp_path):
    a = AppConfig(input_device="USB", preset="studio")
    a.save()
    b = AppConfig.load()
    assert (b.input_device, b.preset) == ("USB", "studio")


def test_bad_app_config_gives_hint(tmp_path):
    AppConfig().path().parent.mkdir(parents=True, exist_ok=True)
    AppConfig.path().write_text("{not json")
    with pytest.raises(ConfigError) as e:
        AppConfig.load()
    assert e.value.hint


def test_user_preset_loaded_and_broken_one_ignored():
    d = user_preset_dir()
    d.mkdir(parents=True)
    (d / "mine.json").write_text(json.dumps({"highpass_hz": 120}))
    (d / "broken.json").write_text("{{{")
    names = [p.name for p in list_presets()]
    assert "mine" in names and "broken" not in names
    assert get_preset("mine").highpass_hz == 120


def test_unknown_preset_hint():
    with pytest.raises(ConfigError) as e:
        get_preset("nope")
    assert "discord" in e.value.hint


def test_ring_buffer_wrap_overflow_underrun():
    r = RingBuffer(10)
    r.write(np.arange(8, dtype=np.float32))
    r.read(6)
    assert r.write(np.arange(100, 109, dtype=np.float32)) == 1  # 2 + 9 = 11 > 10 -> drops 1
    data, got = r.read(20)
    assert got == 10 and data[0] == 7 and data[9] == 108 and data[10:].sum() == 0


def test_level_meter_peak_hold_clip():
    m = LevelMeter(SR)
    s = m.update(np.full(480, 0.5, dtype=np.float32))
    assert s.peak_db == pytest.approx(-6.02, abs=0.1) and not s.clipping
    s = m.update(np.ones(480, dtype=np.float32))
    assert s.clipping and s.clip_count == 480
    s2 = m.update(np.zeros(480, dtype=np.float32))
    assert s2.hold_db == pytest.approx(0.0, abs=0.1) and s2.peak_db < 0.0


def test_spectrum_peak_lands_in_right_band():
    sp = SpectrumAnalyzer(SR)
    t = np.arange(4096) / SR
    sp.feed((0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
    assert abs(sp.band_centers()[int(np.argmax(sp.bands))] - 1000) < 250


def test_wav_roundtrip_and_resample(tmp_path):
    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    p = tmp_path / "a.wav"
    write_wav(p, x, SR)
    y, sr = read_wav(p, target_rate=16000)
    assert sr == 16000 and abs(len(y) - 16000) <= 1


def test_profile_store_roundtrip():
    psd = np.linspace(1, 2, 257)
    save_profile(psd, SR, -55.0, -46.0)
    got = load_profile(SR)
    assert got is not None and np.allclose(got[0], psd) and got[1:] == (-55.0, -46.0)
    assert load_profile(44100) is None


def test_analyze_flags_noise_and_clipping():
    rng = np.random.default_rng(1)
    t = np.arange(SR * 4) / SR
    voice = 0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 1 * t) > 0)
    noisy = (voice + rng.standard_normal(len(t)) * 0.05).astype(np.float32)
    s = analyze(noisy, SR)
    assert s.snr_db is not None and s.noise_floor_db > -40
    assert any(f.level == "bad" for f in diagnose(s))
    clipped = np.clip(noisy * 10, -1, 1)
    assert analyze(clipped, SR).clip_samples > 0


def test_analyze_handles_silence_and_short():
    with pytest.raises(ValueError):
        analyze(np.zeros(100), SR)
    s = analyze(np.zeros(SR, dtype=np.float32) + 1e-5, SR)
    assert s.active_rms_db is None and diagnose(s)[0].level == "warn"
    assert clarity_label(None) == "n/a"
