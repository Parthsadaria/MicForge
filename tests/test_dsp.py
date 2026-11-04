import numpy as np
import pytest
from scipy.signal import sosfreqz

from micforge.config import EQBand
from micforge.dsp.biquad import BiquadFilter, design_biquad
from micforge.dsp.chain import Chain, build_chain, process_offline
from micforge.dsp.compressor import Compressor, Limiter
from micforge.dsp.deesser import DeEsser
from micforge.dsp.denoise import SpectralDenoiser
from micforge.dsp.gate import NoiseGate
from micforge.dsp.leveler import Leveler
from micforge.dsp.util import rms_db
from micforge.presets import BUILTIN, get_preset

SR = 48000
rng = np.random.default_rng(42)


def tone(f, dur=1.0, amp=0.1):
    t = np.arange(int(SR * dur)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def run(proc, x, block=480):
    return process_offline(Chain([proc]), x, block)


@pytest.mark.parametrize("kind,f0,gain", [("peak", 3000, 6.0), ("peak", 300, -4.0), ("highshelf", 8000, 5.0), ("lowshelf", 200, 4.0)])
def test_biquad_gain_at_center(kind, f0, gain):
    sos = design_biquad(kind, SR, f0, 0.9, gain)[None, :]
    probe = {"peak": f0, "highshelf": 18000, "lowshelf": 30}[kind]
    _, h = sosfreqz(sos, worN=[probe], fs=SR)
    assert 20 * np.log10(abs(h[0])) == pytest.approx(gain, abs=0.6)


def test_biquad_block_size_independent():
    x = (rng.standard_normal(SR) * 0.1).astype(np.float32)
    bands = [EQBand("peak", 3000, 4, 1.0), EQBand("highpass", 100, 0, 0.707)]
    a = run(BiquadFilter.from_bands(SR, bands), x, 480)
    b = run(BiquadFilter.from_bands(SR, bands), x, 137)
    assert np.allclose(a, b, atol=1e-5)


def test_highpass_removes_rumble_keeps_voice():
    f = BiquadFilter.highpass(SR, 90)
    assert rms_db(run(f, tone(30))[SR // 2 :]) < rms_db(tone(30)) - 20
    f.reset()
    assert rms_db(run(f, tone(1000))[SR // 2 :]) == pytest.approx(rms_db(tone(1000)), abs=0.5)


def test_denoiser_perfect_reconstruction_when_unity():
    d = SpectralDenoiser(SR)
    d._gmin = 1.0
    x = (rng.standard_normal(SR) * 0.1).astype(np.float32)
    y = run(d, x)
    assert np.max(np.abs(y[1000:] - x[1000:])) < 1e-6  # first n_fft samples are the startup fade-in


def test_denoiser_reduces_noise_preserves_speechlike():
    d = SpectralDenoiser(SR, strength=0.6)
    noise = (rng.standard_normal(SR * 6) * 0.01).astype(np.float32)
    y = run(d, noise)
    assert rms_db(y[SR * 3 :]) < rms_db(noise[SR * 3 :]) - 8
    assert d.noise_floor_db() == pytest.approx(-40, abs=3.5)

    # A stationary tone is (correctly) indistinguishable from noise; speech is bursty, so test with bursts.
    d2 = SpectralDenoiser(SR, strength=0.6)
    on = (np.arange(SR * 6) // int(0.4 * SR)) % 2 == 0
    sig = tone(500, 6, 0.1) * on + noise
    out = run(d2, sig)
    sel = on.copy()
    sel[: SR * 3] = False
    assert rms_db(out[sel]) == pytest.approx(rms_db(sig[sel]), abs=2.5)
    assert rms_db(out[~on][SR * 2 :]) < rms_db(sig[~on][SR * 2 :]) - 6


def test_denoiser_profile_roundtrip_and_shape_guard():
    d = SpectralDenoiser(SR)
    psd = d.learn((rng.standard_normal(SR) * 0.01).astype(np.float32))
    assert psd.shape == (d.n_bins,)
    assert d.set_profile(psd) is True
    assert d.set_profile(np.ones(10)) is False


def test_denoiser_latency_matches_report():
    d = SpectralDenoiser(SR)
    d._gmin = 1.0
    x = np.zeros(SR, dtype=np.float32)
    x[10000] = 1.0
    y = np.concatenate([d.process(x[i : i + 480]) for i in range(0, SR - 480 + 1, 480)])
    assert int(np.argmax(np.abs(y))) - 10000 == d.latency_samples


def test_gate_closes_on_quiet_opens_on_loud():
    g = NoiseGate(SR, threshold_db=-40, range_db=30)
    quiet = tone(300, 1, 0.001)
    loud = tone(300, 1, 0.3)
    yq = run(g, quiet)
    assert rms_db(yq[SR // 2 :]) < rms_db(quiet) - 20
    yl = run(g, np.concatenate([quiet, loud]))
    assert rms_db(yl[-SR // 2 :]) == pytest.approx(rms_db(loud), abs=1.0)


def test_compressor_reduces_loud_not_quiet():
    c = Compressor(SR, threshold_db=-20, ratio=4, makeup_db=0, knee_db=0)
    loud = tone(500, 1, 0.5)
    y = run(c, loud)
    assert rms_db(y[SR // 2 :]) < rms_db(loud) - 6
    c2 = Compressor(SR, threshold_db=-20, ratio=4, makeup_db=0)
    quiet = tone(500, 1, 0.01)
    assert rms_db(run(c2, quiet)[SR // 2 :]) == pytest.approx(rms_db(quiet), abs=0.3)


def test_limiter_never_exceeds_ceiling():
    li = Limiter(SR, ceiling_db=-3.0)
    y = run(li, (rng.standard_normal(SR) * 2.0).astype(np.float32))
    assert np.max(np.abs(y)) <= 10 ** (-3 / 20) + 1e-6


def test_leveler_lifts_quiet_voice():
    lv = Leveler(SR, target_db=-20, max_gain_db=12)
    x = tone(300, 8, 0.03)
    y = run(lv, x)
    assert rms_db(y[-SR:]) > rms_db(x[-SR:]) + 4


def test_deesser_attenuates_sibilance_only():
    d = DeEsser(SR, freq_hz=7000, threshold_db=-40, ratio=4)
    sib, low = tone(7000, 1, 0.2), tone(300, 1, 0.2)
    assert rms_db(run(d, sib)[SR // 2 :]) < rms_db(sib) - 3
    d.reset()
    assert rms_db(run(d, low)[SR // 2 :]) == pytest.approx(rms_db(low), abs=0.5)


@pytest.mark.parametrize("name", list(BUILTIN))
@pytest.mark.parametrize("block", [480, 441, 1024, 64])
def test_every_preset_is_stable(name, block):
    chain = build_chain(get_preset(name), SR)
    x = (rng.standard_normal(SR) * 0.2).astype(np.float32)
    y = process_offline(chain, x, block)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 1.0


def test_bypass_preset_is_transparent():
    chain = build_chain(get_preset("bypass"), SR)
    x = (rng.standard_normal(SR) * 0.1).astype(np.float32)
    assert np.allclose(process_offline(chain, x), x, atol=1e-7)
