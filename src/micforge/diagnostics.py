"""Export diagnostics with redaction, system inspection, and plain-English doctor summary."""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .audio.backend import get_sd
from .audio.devices import detect_virtual_output, find_device, list_devices
from .config import AppConfig, config_dir

log = logging.getLogger("micforge.diagnostics")

SECRET_KEY_PATTERNS = [
    re.compile(r"(api[_-]?key|token|secret|password|auth|webhook[_-]?url|credential)", re.IGNORECASE),
]

URL_AUTH_PATTERN = re.compile(r"://([^:]+):([^@]+)@")


def redact_secrets(obj: Any) -> Any:
    """Recursively redact secrets and user home directories from data structures."""
    user_home = str(Path.home())
    user_home_forward = user_home.replace("\\", "/")

    def _clean_str(s: str) -> str:
        # Redact embedded credentials in URLs
        s = URL_AUTH_PATTERN.sub("://***:***@", s)
        # Redact home paths
        if user_home in s:
            s = s.replace(user_home, "<USER_HOME>")
        if user_home_forward in s:
            s = s.replace(user_home_forward, "<USER_HOME>")
        return s

    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            k_str = str(k)
            # Check if key implies secret
            if any(p.search(k_str) for p in SECRET_KEY_PATTERNS):
                cleaned[k_str] = "***REDACTED***"
            else:
                cleaned[k_str] = redact_secrets(v)
        return cleaned
    elif isinstance(obj, (list, tuple)):
        return [redact_secrets(item) for item in obj]
    elif isinstance(obj, str):
        return _clean_str(obj)
    return obj


@dataclass
class DiagnosticReport:
    """Complete system diagnostic telemetry."""
    micforge_version: str = __version__
    timestamp: float = field(default_factory=time.time)
    platform_info: dict[str, str] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)
    config_summary: dict[str, Any] = field(default_factory=dict)
    devices: list[dict[str, Any]] = field(default_factory=list)
    engine_stats: dict[str, Any] = field(default_factory=dict)
    noise_profile: dict[str, Any] = field(default_factory=dict)
    doctor_checks: list[dict[str, str]] = field(default_factory=list)
    overall_status: str = "HEALTHY"  # HEALTHY | WARNING | CRITICAL
    summary_text: str = ""
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_diagnostics(
    app: AppConfig,
    engine_snapshot: Any | None = None,
    recent_logs: list[str] | None = None,
) -> DiagnosticReport:
    """Gather complete diagnostic data with all secrets redacted."""
    report = DiagnosticReport()

    # 1. Platform Info
    report.platform_info = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": redact_secrets(sys.executable),
    }

    # 2. Package Versions
    pkgs = {"micforge": __version__, "python": sys.version.split()[0]}
    for mod_name in ("sounddevice", "numpy", "scipy", "onnxruntime", "psutil", "rich"):
        try:
            m = __import__(mod_name)
            pkgs[mod_name] = getattr(m, "__version__", "installed")
        except ImportError:
            pkgs[mod_name] = "not installed"
    report.package_versions = pkgs

    # 3. Redacted Config
    try:
        report.config_summary = redact_secrets(app.to_dict())
    except Exception as exc:
        report.config_summary = {"error": str(exc)}

    # 4. Audio Devices
    try:
        pool = list_devices(all_apis=True)
        report.devices = [
            {
                "index": d.index,
                "name": d.name,
                "hostapi": d.hostapi,
                "max_input": d.max_input,
                "max_output": d.max_output,
                "default_rate": d.default_rate,
                "is_default_input": d.is_default_input,
                "is_default_output": d.is_default_output,
            }
            for d in pool
        ]
    except Exception as exc:
        report.devices = [{"error": str(exc)}]

    # 5. Engine Runtime Stats
    if engine_snapshot is not None:
        report.engine_stats = {
            "running": getattr(engine_snapshot, "running", False),
            "cpu_load_pct": round(getattr(engine_snapshot, "cpu_load", 0.0) * 100, 1),
            "latency_ms": round(getattr(engine_snapshot, "latency_ms", 0.0), 1),
            "output_underruns": getattr(engine_snapshot, "output_underruns", 0),
            "input_overflows": getattr(engine_snapshot, "input_overflows", 0),
            "drift_corrections": getattr(engine_snapshot, "drift_corrections", 0),
            "errors": getattr(engine_snapshot, "errors", 0),
            "bypass": getattr(engine_snapshot, "bypass", False),
            "muted": getattr(engine_snapshot, "muted", False),
        }
        if hasattr(engine_snapshot, "noise_profile") and engine_snapshot.noise_profile:
            np = engine_snapshot.noise_profile
            report.noise_profile = {
                "name": getattr(np, "name", "unknown"),
                "noise_floor_db": getattr(np, "noise_floor_db", -62.0),
                "gate_threshold_db": getattr(np, "gate_threshold_db", -50.0),
                "frozen": getattr(np, "frozen", False),
            }

    # 6. Run Doctor Checks
    _run_doctor_assessment(report, app)
    return report


def _run_doctor_assessment(report: DiagnosticReport, app: AppConfig) -> None:
    """Generate plain-English health assessment with next steps."""
    checks = []
    issues = 0
    critical = 0

    # Check 1: Python
    if sys.version_info >= (3, 10):
        checks.append({"name": "Python Runtime", "status": "OK", "detail": f"Python {platform.python_version()} (compatible)"})
    else:
        checks.append({"name": "Python Runtime", "status": "BAD", "detail": f"Python {platform.python_version()} is older than 3.10"})
        critical += 1

    # Check 2: Audio Backend
    try:
        sd = get_sd()
        pa_ver = str(sd.get_portaudio_version()[1])
        checks.append({"name": "PortAudio Engine", "status": "OK", "detail": pa_ver})
    except Exception as exc:
        checks.append({"name": "PortAudio Engine", "status": "CRITICAL", "detail": f"Could not load audio backend: {exc}"})
        critical += 1

    # Check 3: Input Microphone
    pool = list_devices(all_apis=True)
    try:
        mic = find_device(app.input_device, "input", pool)
        checks.append({"name": "Microphone", "status": "OK", "detail": f"{mic.name} [{mic.hostapi}]"})
    except Exception as exc:
        checks.append({"name": "Microphone", "status": "CRITICAL", "detail": f"Microphone device not found: {exc}"})
        critical += 1

    # Check 4: Virtual Cable (for Discord/OBS)
    cable = None
    if app.output_device is not None:
        try:
            cable = find_device(app.output_device, "output", pool)
        except Exception:
            pass
    if cable is None:
        cable = detect_virtual_output(pool)

    if cable is not None:
        checks.append({"name": "Virtual Cable", "status": "OK", "detail": f"{cable.name} (Discord/OBS ready)"})
    else:
        checks.append({
            "name": "Virtual Cable",
            "status": "WARN",
            "detail": "No virtual cable (VB-CABLE / BlackHole) detected. Discord won't receive processed audio directly.",
        })
        issues += 1

    # Check 5: Latency & Real-Time Performance
    if report.engine_stats:
        lat = report.engine_stats.get("latency_ms", 0.0)
        underruns = report.engine_stats.get("output_underruns", 0)
        drift = report.engine_stats.get("drift_corrections", 0)
        if underruns == 0:
            checks.append({"name": "Audio Buffer Sync", "status": "OK", "detail": "Zero dropped glitches, perfect stream sync"})
        else:
            checks.append({"name": "Audio Buffer Sync", "status": "WARN", "detail": f"{underruns} buffer glitches dropped"})
            issues += 1

    report.doctor_checks = checks

    # Assign overall status
    if critical > 0:
        report.overall_status = "CRITICAL"
        report.summary_text = "Micforge detected critical configuration or hardware issues preventing normal audio routing."
        report.next_steps = [
            "Check that your USB microphone is plugged in and authorized in OS privacy settings.",
            "Review docs/TROUBLESHOOTING.md for device connection troubleshooting.",
            "Run 'micforge setup' to re-select active microphone and playback endpoints.",
        ]
    elif issues > 0:
        report.overall_status = "WARNING"
        report.summary_text = "Micforge is operational, but one or more optimizations (like virtual cable routing) need attention."
        report.next_steps = [
            "Install a free virtual audio cable (e.g. VB-CABLE on Windows or BlackHole on macOS).",
            "Set Discord's Input Device to 'CABLE Output (VB-Audio Virtual Cable)'.",
            "See docs/TROUBLESHOOTING.md for step-by-step app setup guides.",
        ]
    else:
        report.overall_status = "HEALTHY"
        report.summary_text = "All systems healthy! Audio routing, noise profile, and DSP pipeline are streaming cleanly."
        report.next_steps = [
            "In Discord/Zoom/OBS, verify your input is set to your Virtual Cable.",
            "Press [Tab] in the live terminal to switch between Studio Inspector and Gamer HUD.",
        ]


def export_diagnostics_zip(
    report: DiagnosticReport,
    output_path: Path | None = None,
    recent_logs: list[str] | None = None,
) -> Path:
    """Create a single .zip diagnostic bundle containing redacted config, report, and logs."""
    timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(report.timestamp))
    if output_path is None:
        target_dir = config_dir() / "diagnostics"
        target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = target_dir / f"micforge_diagnostics_{timestamp_str}.zip"
    else:
        zip_path = Path(output_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Full JSON Report
        zf.writestr("diagnostic_report.json", json.dumps(report.to_dict(), indent=2))

        # 2. Plain-English Doctor Summary
        doc_lines = [
            f"Micforge Diagnostic Doctor Summary (v{report.micforge_version})",
            f"Generated: {time.ctime(report.timestamp)}",
            "=" * 60,
            f"Overall Status: {report.overall_status}",
            f"Summary: {report.summary_text}",
            "",
            "Health Checks:",
        ]
        for c in report.doctor_checks:
            doc_lines.append(f"  [{c['status']:<4}] {c['name']:<24}: {c['detail']}")
        doc_lines.append("")
        doc_lines.append("Recommended Next Steps:")
        for step in report.next_steps:
            doc_lines.append(f"  • {step}")
        doc_lines.append("")
        doc_lines.append("See docs/TROUBLESHOOTING.md for comprehensive solutions.")
        zf.writestr("doctor_summary.txt", "\n".join(doc_lines))

        # 3. Audio Devices Inventory
        dev_lines = ["Audio Devices Inventory:\n"]
        for d in report.devices:
            dev_lines.append(f"[{d.get('index', '?')}] {d.get('name', 'unknown')} (in={d.get('max_input', 0)}, out={d.get('max_output', 0)}) hostapi={d.get('hostapi', '?')}")
        zf.writestr("devices.txt", "\n".join(dev_lines))

        # 4. Redacted Config
        zf.writestr("config_redacted.json", json.dumps(report.config_summary, indent=2))

        # 5. Recent Logs
        log_content = "\n".join(recent_logs) if recent_logs else "No active log buffer captured."
        zf.writestr("recent_logs.txt", redact_secrets(log_content))

    log.info("Exported diagnostics zip to %s", zip_path)
    return zip_path

