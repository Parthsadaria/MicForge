"""Regression tests ensuring zero audio clicking/popping artifacts.

Covers:
1. Lock-free ring buffer concurrency
2. Ring buffer zero-allocation read_into()
3. CoFi-Lite STFT buffer shift continuity
4. OutputSink underrun pop concealment & no silence injection
5. Clock drift correction fractional interpolation smoothness
6. Unity gain zero-allocation fast path
"""
import numpy as np
import threading
import time

from micforge.audio.devices import DeviceInfo
from micforge.audio.engine import OutputSink, SinkSpec
from micforge.audio.ring import RingBuffer
from micforge.dsp.cofilite import CoFiLiteDenoiser


def test_ring_buffer_lockfree_concurrent():
    rb = RingBuffer(4800)
    errors = []
    iterations = 2000

    def writer():
        for i in range(iterations):
            block = np.full(480, float(i % 1000), dtype=np.float32)
            rb.write(block)
            time.sleep(0.00005)

    def reader():
        for _ in range(iterations):
            data, got = rb.read(480)
            if got > 0 and (np.any(np.isnan(data[:got])) or np.any(np.isinf(data[:got]))):
                errors.append("NaN or Inf in read data")
            time.sleep(0.00005)

    tw = threading.Thread(target=writer)
    tr = threading.Thread(target=reader)
    tw.start()
    tr.start()
    tw.join()
    tr.join()

    assert not errors, f"Ring buffer corruption: {errors}"


def test_ring_buffer_read_into():
    rb = RingBuffer(4800)
    data = np.arange(960, dtype=np.float32)
    rb.write(data)

    dest = np.zeros(480, dtype=np.float32)
    got = rb.read_into(dest)
    assert got == 480
    np.testing.assert_array_equal(dest, data[:480])

    got = rb.read_into(dest)
    assert got == 480
    np.testing.assert_array_equal(dest, data[480:960])

    got = rb.read_into(dest)
    assert got == 0
    np.testing.assert_array_equal(dest, np.zeros(480, dtype=np.float32))


def test_cofilite_buffer_shift_continuity():
    dn = CoFiLiteDenoiser(48000, strength=0.6)
    outputs = []

    for i in range(100):
        t = np.arange(i * 480, (i + 1) * 480) / 48000.0
        block = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        outputs.append(dn.process(block))

    full = np.concatenate(outputs)
    signal = full[2400:]  # skip priming

    diff = np.abs(np.diff(signal))
    large_jumps = np.sum(diff > 0.15)
    assert large_jumps == 0, f"Discontinuities ({large_jumps}) detected in CoFi-Lite output"


def test_output_sink_underrun_no_synthetic_silence():
    dev = DeviceInfo(
        index=0, name="test", hostapi="test", max_input=0, max_output=2,
        default_rate=48000.0, is_default_input=False, is_default_output=True
    )
    sink = OutputSink(SinkSpec("test", dev, enabled=True, gain_db=0.0), 48000, 480, 2.5)
    sink._active = True

    outdata = np.zeros((480, 2), dtype=np.float32)
    sink._callback(outdata, 480, None, None)

    assert sink.ring.fill == 0, "Underrun should not inject silence into live ring buffer"
    assert sink.underruns == 1


def test_drift_correction_smoothness():
    dev = DeviceInfo(
        index=0, name="test", hostapi="test", max_input=0, max_output=2,
        default_rate=48000.0, is_default_input=False, is_default_output=True
    )
    sink = OutputSink(SinkSpec("test", dev, enabled=True, gain_db=0.0), 48000, 480, 2.5)
    sink._active = True

    t = np.arange(48000, dtype=np.float32) / 48000.0
    smooth = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sink.ring.write(smooth)

    sink._drift_cooldown = 200
    sink._avg_fill = sink.prefill + 480 + int(3.0 * 480)

    outdata = np.zeros((480, 2), dtype=np.float32)
    sink._callback(outdata, 480, None, None)

    diff = np.abs(np.diff(outdata[:, 0]))
    assert diff.max() < 0.1, f"Drift correction produced click: max jump {diff.max()}"


def test_unity_gain_direct_pass():
    dev = DeviceInfo(
        index=0, name="test", hostapi="test", max_input=0, max_output=2,
        default_rate=48000.0, is_default_input=False, is_default_output=True
    )
    sink = OutputSink(SinkSpec("test", dev, enabled=True, gain_db=0.0), 48000, 480, 2.5)
    assert sink._gain_is_unity

    signal = np.arange(480, dtype=np.float32) * 0.001
    sink.push(signal)

    data, got = sink.ring.read(480 + sink.prefill)
    actual = data[sink.base_prefill : sink.base_prefill + 480]
    np.testing.assert_array_equal(actual, signal)


def test_cofilite_session_cache_fast_reinit():
    # First ensure session is loaded
    d1 = CoFiLiteDenoiser(48000)
    t0 = time.perf_counter()
    d2 = CoFiLiteDenoiser(48000)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # Re-initialization from session cache should be sub-millisecond (not 200ms)
    assert elapsed_ms < 10.0, f"Session caching failed: took {elapsed_ms:.1f}ms"
    assert d1._session is d2._session, "Models should share cached ONNX session"


def test_cofilite_smart_nap_and_instant_wakeup():
    dn = CoFiLiteDenoiser(48000, strength=1.0)
    dn.nap_after_sec = 0.5  # 50 blocks for fast unit testing
    dn._nap_blocks_limit = 50

    silence_block = np.zeros(480, dtype=np.float32)
    # Feed 60 blocks of silence (> 50 blocks)
    for _ in range(60):
        dn.process(silence_block)

    assert dn._is_napping, "Model should be in nap state after prolonged silence"
    assert dn.telemetry()["napping"] == 1.0

    # Feed speech block -> model must wake up on the very first block!
    t = np.arange(480) / 48000.0
    speech_block = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    out = dn.process(speech_block)

    assert not dn._is_napping, "Model must wake up immediately upon detecting speech"
    assert dn.telemetry()["napping"] == 0.0
    assert out.shape == speech_block.shape


def test_output_sink_notify_pipeline_change():
    dev = DeviceInfo(
        index=0, name="test", hostapi="test", max_input=0, max_output=2,
        default_rate=48000.0, is_default_input=False, is_default_output=True
    )
    sink = OutputSink(SinkSpec("test", dev, enabled=True, gain_db=0.0), 48000, 480, 2.5)
    sink._drift_cooldown = 200
    sink._consecutive_underruns = 5

    sink.notify_pipeline_change()

    assert sink._drift_cooldown == 0, "Pipeline change must reset drift cooldown"
    assert sink._consecutive_underruns == 0, "Pipeline change must clear consecutive underruns"

