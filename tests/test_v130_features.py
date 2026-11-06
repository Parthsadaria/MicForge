"""Unit tests for Micforge v1.3.0 features:
- Adaptive noise profiling & environment profiles
- Hot-plug device watcher & auto-recovery with fallback
- Diagnostic collection, secret redaction, and zip export
- End-to-end latency probe chirp & cross-correlation
- Simple/Advanced tuning macro mappings
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from micforge.config import AppConfig, ChainConfig, EQBand
from micforge.audio.noise_profile import (
    AdaptiveNoiseProfiler,
    BUILTIN_ENVIRONMENTS,
    NoiseProfile,
    list_noise_profiles,
)
from micforge.audio.devices import DeviceInfo
from micforge.audio.device_watch import (
    DeviceEventType,
    DeviceWatchEvent,
    DeviceWatcher,
)
from micforge.diagnostics import (
    collect_diagnostics,
    export_diagnostics_zip,
    redact_secrets,
)
from micforge.audio.latency import (
    compute_cross_correlation_delay,
    generate_probe_chirp,
    measure_end_to_end_latency,
)
from micforge.ui.tuning import (
    MacroKnobs,
    chain_config_to_macro,
    macro_to_chain_config,
)


# ==============================================================================
# 1. ADAPTIVE NOISE PROFILING TESTS
# ==============================================================================


def test_builtin_environments_present():
    """Verify standard baseline noise environments are available."""
    assert "quiet_room" in BUILTIN_ENVIRONMENTS
    assert "fan_on" in BUILTIN_ENVIRONMENTS
    assert "mechanical_keyboard" in BUILTIN_ENVIRONMENTS

    qr = BUILTIN_ENVIRONMENTS["quiet_room"]
    assert qr.noise_floor_db < -60.0
    assert qr.gate_threshold_db < -45.0


def test_adaptive_noise_profiler_quiet_adaptation(tmp_path, monkeypatch):
    """Profiler adapts smoothly to quiet ambient noise over consecutive quiet blocks."""
    profiler = AdaptiveNoiseProfiler(sample_rate=48000, environment="quiet_room", adaptation_rate=0.2)
    initial_floor = profiler.current_profile.noise_floor_db

    # Feed quiet ambient noise (-55 dBFS RMS) for 15 blocks
    rng = np.random.default_rng(42)
    target_amplitude = 10.0 ** (-55.0 / 20.0)
    for _ in range(15):
        block = (rng.standard_normal(480) * target_amplitude).astype(np.float32)
        profiler.update(block, is_voiced=False, speech_prob=0.0)

    # After quiet blocks, floor should adapt towards target
    assert profiler.current_profile.noise_floor_db != initial_floor
    assert profiler.current_profile.sample_count > 0


def test_adaptive_noise_profiler_speech_rejection():
    """SAFE ADAPTATION RULE: Must reject voiced speech and never update floor on speech."""
    profiler = AdaptiveNoiseProfiler(sample_rate=48000, environment="quiet_room")
    floor_before = profiler.current_profile.noise_floor_db

    # Strong speech block with is_voiced=True
    speech_block = (np.sin(2 * np.pi * 300 * np.linspace(0, 1, 480)) * 0.5).astype(np.float32)
    adapted = profiler.update(speech_block, is_voiced=True, speech_prob=0.95)

    assert adapted is False
    assert profiler.current_profile.noise_floor_db == floor_before
    assert profiler.current_profile.sample_count == 0


def test_adaptive_noise_profiler_transient_spike_rejection():
    """SAFE ADAPTATION RULE: Transient energy spikes (>16 dB above floor) must be rejected."""
    profiler = AdaptiveNoiseProfiler(sample_rate=48000, environment="quiet_room")
    profiler.current_profile.noise_floor_db = -60.0

    # Sudden loud burst (e.g. keyboard click or book dropping: -20 dBFS)
    loud_transient = (np.ones(480) * 0.1).astype(np.float32)
    adapted = profiler.update(loud_transient, is_voiced=False, speech_prob=0.0)

    assert adapted is False


def test_adaptive_noise_profiler_freeze_and_switch(tmp_path):
    """Test freeze/unfreeze and environment profile switching."""
    profiler = AdaptiveNoiseProfiler(sample_rate=48000, environment="quiet_room")
    assert not profiler.is_frozen

    profiler.freeze()
    assert profiler.is_frozen

    block = (np.random.randn(480) * 0.001).astype(np.float32)
    assert profiler.update(block, is_voiced=False) is False

    profiler.unfreeze()
    assert not profiler.is_frozen

    # Switch to fan_on
    fan_prof = profiler.switch_environment("fan_on")
    assert fan_prof.name == "fan_on"
    assert profiler.current_profile.spectral_tilt == BUILTIN_ENVIRONMENTS["fan_on"].spectral_tilt

    # Save profile and list profiles
    saved_path = profiler.save("test_custom_env")
    assert saved_path is not None
    assert saved_path.exists()
    all_profs = list_noise_profiles()
    assert "test_custom_env" in all_profs


# ==============================================================================
# 2. DEVICE WATCHER & AUTO-RECOVERY TESTS
# ==============================================================================


def test_device_watcher_diff_detection():
    """Watcher accurately identifies added and removed audio devices."""
    watcher = DeviceWatcher(poll_interval_s=0.1)

    devices_v1 = [
        DeviceInfo(index=0, name="Realtek Mic", hostapi=0, max_input=2, max_output=0, default_rate=48000, is_default_input=True, is_default_output=False),
        DeviceInfo(index=1, name="Speakers", hostapi=0, max_input=0, max_output=2, default_rate=48000, is_default_input=False, is_default_output=True),
    ]
    devices_v2 = [
        DeviceInfo(index=0, name="Realtek Mic", hostapi=0, max_input=2, max_output=0, default_rate=48000, is_default_input=True, is_default_output=False),
        DeviceInfo(index=2, name="USB Headset", hostapi=0, max_input=1, max_output=2, default_rate=48000, is_default_input=False, is_default_output=False),  # Added
    ]

    events_received: list[DeviceWatchEvent] = []
    watcher.add_listener(lambda ev: events_received.append(ev))

    # Seed baseline
    watcher.check_devices(devices_v1)
    events_received.clear()

    # Diff with new device added & speakers removed
    watcher.check_devices(devices_v2)
    assert len(events_received) >= 2

    types = {ev.event_type for ev in events_received}
    assert DeviceEventType.DEVICE_ADDED in types
    assert DeviceEventType.DEVICE_REMOVED in types


def test_device_watcher_recovery_success():
    """Recovery succeeds within bounded retries when recover_fn returns True."""
    call_count = 0
    def mock_recover(kind, dev):
        nonlocal call_count
        call_count += 1
        return call_count >= 2  # Succeeds on 2nd attempt

    watcher = DeviceWatcher(
        poll_interval_s=0.5,
        max_retries=3,
        backoff_base_s=0.01,
        on_recovery_attempt=mock_recover,
    )

    success = watcher._perform_recovery("input", "Realtek Mic")

    assert success is True
    assert call_count == 2
    assert not watcher.is_recovering


def test_device_watcher_recovery_fallback_to_bypass():
    """When all recovery retries fail, trigger fallback callback so user is never muted."""
    mock_recover = MagicMock(return_value=False)
    mock_fallback = MagicMock()

    watcher = DeviceWatcher(
        poll_interval_s=0.5,
        max_retries=2,
        backoff_base_s=0.01,
        on_recovery_attempt=mock_recover,
        on_fallback=mock_fallback,
    )

    success = watcher._perform_recovery("input", "Realtek Mic")

    assert success is False
    assert mock_recover.call_count == 2
    mock_fallback.assert_called_once()
    assert not watcher.is_recovering


# ==============================================================================
# 3. DIAGNOSTICS & SECRET REDACTION TESTS
# ==============================================================================


def test_redact_secrets_recursive():
    """Verify passwords, tokens, API keys, URLs and home paths are completely redacted."""
    raw_data = {
        "user_name": "developer",
        "api_token": "sk-ant-api03-secretkey1234567890",
        "password": "supersecretpassword",
        "webhook_url": "https://discord.com/api/webhooks/12345/token_xyz",
        "server_url": "https://admin:superpass@streaming.micforge.local/feed",
        "nested": {
            "credentials": {"auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"},
            "log_path": str(Path.home() / "micforge" / "debug.log"),
            "normal_value": 48000,
        },
    }

    redacted = redact_secrets(raw_data)

    assert redacted["api_token"] == "***REDACTED***"
    assert redacted["password"] == "***REDACTED***"
    assert redacted["webhook_url"] == "***REDACTED***"
    assert "***:***@" in redacted["server_url"]
    assert redacted["nested"]["credentials"] == "***REDACTED***"
    assert str(Path.home()) not in redacted["nested"]["log_path"]
    assert "<USER_HOME>" in redacted["nested"]["log_path"]
    assert redacted["nested"]["normal_value"] == 48000


def test_collect_diagnostics_and_export_zip(tmp_path):
    """Verify diagnostic collection and zip bundling creates all expected internal files."""
    app = AppConfig()
    app.preset = "streaming"

    report = collect_diagnostics(app)
    assert report.micforge_version is not None
    assert "os" in report.platform_info
    assert len(report.doctor_checks) > 0

    zip_path = tmp_path / "diagnostics_test.zip"
    created_zip = export_diagnostics_zip(report, zip_path)

    assert created_zip.exists()
    assert zipfile.is_zipfile(created_zip)

    with zipfile.ZipFile(created_zip, "r") as z:
        names = z.namelist()
        assert "diagnostic_report.json" in names
        assert "doctor_summary.txt" in names
        assert "devices.txt" in names
        assert "config_redacted.json" in names
        assert "recent_logs.txt" in names

        # Validate that diagnostic_report.json inside zip is valid JSON
        report_data = json.loads(z.read("diagnostic_report.json").decode("utf-8"))
        assert "platform_info" in report_data


# ==============================================================================
# 4. LATENCY ESTIMATION & CHIRP CORRELATION TESTS
# ==============================================================================


def test_probe_chirp_generation():
    """Verify logarithmic test chirp generation has correct length, type, and window edges."""
    sr = 48000
    duration_s = 0.06
    chirp = generate_probe_chirp(sample_rate=sr, duration_s=duration_s)

    expected_len = int(sr * duration_s)
    assert len(chirp) == expected_len
    assert chirp.dtype == np.float32
    # Hanning window edges must taper to 0
    assert abs(chirp[0]) < 1e-4
    assert abs(chirp[-1]) < 1e-4


def test_cross_correlation_delay_detection():
    """Synthetic test: chirp delayed by exact offset is accurately resolved via cross-correlation."""
    sr = 48000
    chirp = generate_probe_chirp(sample_rate=sr, duration_s=0.05)

    delay_samples = 480  # Exactly 10.0 ms delay
    recv_signal = np.zeros(sr, dtype=np.float32)
    recv_signal[delay_samples : delay_samples + len(chirp)] = chirp

    # Add small background noise
    rng = np.random.default_rng(123)
    recv_signal += rng.standard_normal(len(recv_signal)).astype(np.float32) * 0.005

    delay_s, confidence = compute_cross_correlation_delay(chirp, recv_signal, sample_rate=sr)
    est_delay_samples = round(delay_s * sr)

    # Must resolve delay within +/- 1 sample
    assert abs(est_delay_samples - delay_samples) <= 1
    # Confidence score should be high (> 0.7) for clean synthetic signal
    assert confidence > 0.7


def test_latency_measurement_fallback():
    """End-to-end latency measurement produces structured breakdown and recommendations."""
    res = measure_end_to_end_latency(sample_rate=48000, block_size=256, jitter_blocks=3)

    assert res.one_way_ms > 0
    assert res.roundtrip_ms >= res.one_way_ms
    assert res.breakdown.dsp_pipeline_ms > 0
    assert res.breakdown.jitter_cushion_ms > 0
    assert len(res.recommendations) > 0


# ==============================================================================
# 5. SIMPLE / ADVANCED TUNING KNOBS TESTS
# ==============================================================================


def test_macro_knobs_clamping():
    """Macro knob values are clamped strictly between 0 and 100."""
    knobs = MacroKnobs(clarity=150.0, warmth=-20.0, noise_removal=85.0, room_cleanup=50.0)
    assert knobs.clarity == 100.0
    assert knobs.warmth == 0.0
    assert knobs.noise_removal == 85.0
    assert knobs.room_cleanup == 50.0


def test_macro_to_chain_config_mapping():
    """Tuning macro knobs deterministically updates DSP chain parameters without breaking presets."""
    base_chain = ChainConfig()
    knobs = MacroKnobs(clarity=80.0, warmth=70.0, noise_removal=90.0, room_cleanup=60.0)

    updated_chain = macro_to_chain_config(knobs, base_chain)

    # Clarity (80) -> high shelf boost, air boost, de-esser active
    assert updated_chain.deesser.enabled is True
    assert len(updated_chain.eq) > 0

    # Warmth (70) -> proximity enabled with boost
    assert updated_chain.proximity.enabled is True
    assert updated_chain.proximity.amount > 0.5

    # Noise Removal (90) -> aggressive denoiser strength & gate threshold
    assert updated_chain.denoise.enabled is True
    assert updated_chain.denoise.strength > 0.8
    assert updated_chain.gate.threshold_db > -45.0

    # Room Cleanup (60) -> deverb enabled with suppression
    assert updated_chain.deverb.enabled is True
    assert updated_chain.deverb.amount > 0.3
    assert updated_chain.highpass_hz >= 80


def test_chain_config_to_macro_reverse_mapping():
    """Reverse mapping produces reasonable macro knob values from existing chain config."""
    chain = ChainConfig()
    chain.proximity.enabled = True
    chain.proximity.amount = 0.7
    chain.denoise.enabled = True
    chain.denoise.strength = 0.85

    macros = chain_config_to_macro(chain)
    assert 0.0 <= macros.clarity <= 100.0
    assert macros.warmth >= 50.0  # High proximity amount reflects in Warmth
    assert macros.noise_removal >= 70.0  # High denoise strength reflects in Noise Removal
    assert 0.0 <= macros.room_cleanup <= 100.0


# ==============================================================================
# 6. DYNAMIC HELP COMMAND TESTS
# ==============================================================================


def test_dynamic_help_overview():
    """Verify micforge help outputs complete categorized directory and exits with 0."""
    from micforge.commands.help_cmd import run_help
    rc = run_help(None)
    assert rc == 0


def test_dynamic_help_command_deep_dive():
    """Verify micforge help <command> introspects flags, examples, and exits with 0."""
    from micforge.commands.help_cmd import run_help
    for cmd in ["run", "latency", "profile", "doctor", "settings"]:
        rc = run_help(cmd)
        assert rc == 0


def test_dynamic_help_unknown_command():
    """Verify micforge help <unknown> returns exit code 1 with friendly error."""
    from micforge.commands.help_cmd import run_help
    rc = run_help("nonexistent_subcommand")
    assert rc == 1
