"""Shared plumbing for commands: resolving devices and building a chain with the saved noise profile."""
from __future__ import annotations

from dataclasses import dataclass

from ..audio.devices import DeviceInfo, detect_virtual_output, find_device, list_devices
from ..config import AppConfig
from ..dsp.chain import Chain, build_chain
from ..errors import DeviceError
from ..presets import get_preset
from ..profile_store import load_profile


def make_chain(app: AppConfig, preset: str, model_override: str | None = None) -> tuple[Chain, bool]:
    """Build a chain for `preset`, applying the saved room-noise profile. Returns (chain, profile_applied)."""
    cfg = get_preset(preset)
    prof = load_profile(app.sample_rate)
    model = model_override or app.denoise_model
    chain = build_chain(cfg, app.sample_rate, noise_profile=prof[0] if prof else None, model_override=model)
    applied = False
    if prof:
        dn = chain.stage("denoise")
        applied = bool(dn and getattr(dn, "profile", None) is not None)
        gate = chain.stage("gate")
        if applied and gate is not None and cfg.gate.enabled:
            gate.threshold_db = prof[2]  # type: ignore[attr-defined]
    return chain, applied


@dataclass
class Devices:
    mic: DeviceInfo
    cable: DeviceInfo | None
    monitor: DeviceInfo | None


def resolve_devices(
    app: AppConfig,
    *,
    need_cable: bool,
    want_monitor: bool,
    mic_override=None,
    cable_override=None,
    monitor_override=None,
) -> Devices:
    pool = list_devices(all_apis=True)
    mic = find_device(mic_override if mic_override is not None else app.input_device, "input", pool)

    cable = None
    if need_cable:
        spec = cable_override if cable_override is not None else app.output_device
        if spec is not None:
            cable = find_device(spec, "output", pool)
        else:
            cable = detect_virtual_output(pool)
        if cable is None:
            raise DeviceError(
                "No virtual audio cable found - Discord needs one to receive the processed mic.",
                hint="Run `micforge doctor` for install steps (VB-CABLE on Windows, BlackHole on macOS), "
                "or use `micforge loopback` to just hear yourself.",
            )

    monitor = None
    if want_monitor:
        spec = monitor_override if monitor_override is not None else app.monitor_device
        monitor = find_device(spec, "output", pool)
        if cable is not None and monitor.index == cable.index:
            raise DeviceError("Monitor and virtual cable are the same device - pick your headphones for monitoring.")
    return Devices(mic, cable, monitor)
