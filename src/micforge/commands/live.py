"""`run` and `loopback`: the live dashboard."""
from __future__ import annotations

import os
import threading
import time

from rich.live import Live

from ..audio.engine import AudioEngine, SinkSpec
from ..config import AppConfig
from ..presets import list_presets
from ..profile_store import save_profile
from ..ui.dashboard import LiveContext, render_dashboard
from ..ui.hotkeys import GlobalHotkeyManager
from ..ui.keys import KeyReader
from ..ui.theme import console
from ..ui.tray import SystemTrayDaemon
from .common import make_chain, resolve_devices

KEYS = [
    ("b", "A/B bypass"), ("x", "mute"), ("l", "monitor"), ("p", "preset"),
    ("m", "model"), ("c", "calibrate"), ("+/-", "denoise"), ("[ ]", "gain"), ("q", "quit"),
]


def run_live(
    app: AppConfig,
    *,
    preset: str | None = None,
    loopback: bool = False,
    monitor: bool = False,
    mic=None,
    cable=None,
    monitor_dev=None,
    model: str | None = None,
    tray: bool = False,
    no_tray: bool = False,
) -> None:
    # Tune GC thresholds to prevent garbage collection pauses during real-time streaming
    import gc
    try:
        gc.freeze()
        gc.set_threshold(60000, 60, 60)
    except Exception:
        pass

    preset = preset or app.preset
    model_override = model or app.denoise_model
    devs = resolve_devices(
        app,
        need_cable=not loopback,
        want_monitor=loopback or monitor,
        mic_override=mic,
        cable_override=cable,
        monitor_override=monitor_dev,
    )
    chain, has_profile = make_chain(app, preset, model_override=model_override)

    sinks: list[SinkSpec] = []
    if devs.cable:
        sinks.append(SinkSpec("cable", devs.cable))
    if devs.monitor:
        sinks.append(SinkSpec("monitor", devs.monitor, enabled=True))
    engine = AudioEngine(
        chain=chain,
        input_device=devs.mic,
        sinks=sinks,
        input_channel=app.input_channel,
        sample_rate=app.sample_rate,
        block_size=app.block_size,
        jitter_blocks=app.jitter_blocks,
        play_cues=app.hotkeys.play_cues,
        cue_volume=app.hotkeys.cue_volume,
    )
    names = [p.name for p in list_presets()]
    ctx = LiveContext(
        title="LOOPBACK TEST" if loopback else "MIC ENHANCER",
        preset=preset,
        input_name=devs.mic.name,
        cable_name=devs.cable.name if devs.cable else None,
        monitor_name=devs.monitor.name if devs.monitor else None,
        sample_rate=app.sample_rate,
        block_size=app.block_size,
        centers=engine.spectrum_range,
        has_profile=has_profile,
        keys=KEYS if not loopback else [k for k in KEYS if k[0] not in ("x", "l")],
    )

    def flash(msg: str, secs: float = 2.5) -> None:
        ctx.message, ctx.message_until = msg, time.monotonic() + secs

    exit_event = threading.Event()
    tray_daemon: SystemTrayDaemon | None = None

    def _on_toggle_mute():
        is_muted = engine.toggle_mute()
        if tray_daemon:
            tray_daemon.update_icon()
        flash("MIC MUTED" if is_muted else "Mic live")

    def _on_set_mute(val: bool):
        engine.set_muted(val)
        if tray_daemon:
            tray_daemon.update_icon()
        flash("MIC MUTED" if val else "Mic live")

    def _on_toggle_bypass():
        is_bp = engine.toggle_bypass()
        if tray_daemon:
            tray_daemon.update_icon()
        flash("BYPASS - raw mic" if is_bp else "PROCESSED")

    def _on_switch_preset(p_name: str):
        ctx.preset = p_name
        new_chain, ctx.has_profile = make_chain(app, p_name, model_override=model_override)
        engine.set_chain(new_chain)
        flash(f"Preset: {p_name}")

    def _on_switch_model(m_type: str):
        dn = engine.chain.stage("denoise")
        from ..dsp.cofilite import CoFiLiteDenoiser
        from ..dsp.denoise import SpectralDenoiser

        if m_type == "spectral":
            new_dn = SpectralDenoiser(
                app.sample_rate,
                getattr(dn, "strength", 1.0),
                getattr(dn, "max_reduction_db", 24.0),
                enabled=getattr(dn, "enabled", True),
            )
            engine.chain.stages = [new_dn if s.name == "denoise" else s for s in engine.chain.stages]
            for s in engine.sinks.values():
                s.notify_pipeline_change()
            flash("Swapped denoiser: Spectral Wiener (classical)")
        else:
            try:
                new_dn = CoFiLiteDenoiser(
                    app.sample_rate,
                    getattr(dn, "strength", 1.0),
                    getattr(dn, "max_reduction_db", 24.0),
                    model_path=model_override,
                )
                engine.chain.stages = [new_dn if s.name == "denoise" else s for s in engine.chain.stages]
                for s in engine.sinks.values():
                    s.notify_pipeline_change()
                flash(f"Swapped denoiser: CoFi-Lite ({new_dn.model_name})")
            except Exception as e:
                flash(f"CoFi-Lite load error: {e}", 3.0)

    # Initialize global hotkeys
    hotkeys = GlobalHotkeyManager(
        app.hotkeys,
        on_toggle_mute=_on_toggle_mute,
        on_set_mute=_on_set_mute,
        on_toggle_bypass=_on_toggle_bypass,
    )
    hotkeys.start()

    # Initialize system tray daemon (unless disabled)
    enable_tray = not no_tray and (tray or app.hotkeys.start_in_tray or os.name == "nt")
    if enable_tray:
        tray_daemon = SystemTrayDaemon(
            app=app,
            engine=engine,
            on_exit=exit_event.set,
            on_switch_preset=_on_switch_preset,
            on_switch_model=_on_switch_model,
        )
        tray_daemon.start()

    # If run in pure tray background mode, run headless
    if tray and not loopback:
        console.print("[bold cyan]✔ Micforge is running in background system tray mode.[/]")
        console.print("  [dim]Right-click the microphone icon in your taskbar for controls, or use your hotkeys.[/]")
        if tray_daemon:
            tray_daemon.toggle_console()
        with engine:
            try:
                while not exit_event.is_set():
                    time.sleep(0.3)
            except KeyboardInterrupt:
                pass
        hotkeys.stop()
        if tray_daemon:
            tray_daemon.stop()
        return

    if loopback:
        flash("Use headphones, or you will get feedback. Press b to A/B raw vs processed.", 6.0)
    elif has_profile:
        flash("Loaded your saved room-noise profile.", 3.0)

    is_cmd = os.name == "nt" and "WT_SESSION" not in os.environ and "TERM_PROGRAM" not in os.environ
    # 10 FPS on cmd.exe avoids console buffer saturation and eliminates reprint stutter
    refresh_interval = 0.10 if is_cmd else 0.07
    last_render = 0.0

    try:
        with engine, KeyReader() as keys, Live(console=console, screen=True, auto_refresh=False) as live:
            while not exit_event.is_set():
                now = time.monotonic()
                cal = engine.pop_calibration()
                if cal is not None:
                    if cal.psd is not None:
                        save_profile(cal.psd, cal.sample_rate, cal.noise_db, cal.gate_threshold_db)
                    flash(f"Calibrated: room noise {cal.noise_db:.1f} dBFS, gate set to {cal.gate_threshold_db:.1f} dB", 5.0)

                should_refresh = (now - last_render >= refresh_interval) or (cal is not None)
                if should_refresh:
                    snap = engine.snapshot()
                    live.update(render_dashboard(snap, ctx, console.width, now), refresh=True)
                    last_render = now

                k = keys.poll(0.02)
                if k is None:
                    continue
                k = k.lower() if k.isalpha() else k
                if k in ("q", "\x03"):
                    break
                if k == "b":
                    _on_toggle_bypass()
                elif k == "x" and not loopback:
                    _on_toggle_mute()
                elif k == "l" and not loopback:
                    if not engine.monitor_available:
                        flash("No monitor device. Restart with --monitor, or run `micforge setup`.", 4.0)
                    else:
                        flash("Monitor ON (use headphones!)" if engine.toggle_monitor() else "Monitor off")
                elif k == "p":
                    i = names.index(ctx.preset) if ctx.preset in names else -1
                    _on_switch_preset(names[(i + 1) % len(names)])
                elif k == "m":
                    dn = engine.chain.stage("denoise")
                    from ..dsp.cofilite import CoFiLiteDenoiser
                    _on_switch_model("spectral" if isinstance(dn, CoFiLiteDenoiser) else "cofilite")
                elif k == "c":
                    engine.begin_calibration(3.0)
                elif k in ("+", "="):
                    dn = engine.chain.stage("denoise")
                    if dn is not None:
                        dn.set_strength(dn.strength + 0.1)  # type: ignore[attr-defined]
                        dn.enabled = True
                        flash(f"Denoise strength {dn.strength:.1f}")  # type: ignore[attr-defined]
                elif k in ("-", "_"):
                    dn = engine.chain.stage("denoise")
                    if dn is not None:
                        dn.set_strength(dn.strength - 0.1)  # type: ignore[attr-defined]
                        flash(f"Denoise strength {dn.strength:.1f}")  # type: ignore[attr-defined]
                elif k == "]":
                    engine.chain.set_input_gain_db(engine.chain.input_gain_db + 1.0)
                    flash(f"Input gain {engine.chain.input_gain_db:+.0f} dB")
                elif k == "[":
                    engine.chain.set_input_gain_db(engine.chain.input_gain_db - 1.0)
                    flash(f"Input gain {engine.chain.input_gain_db:+.0f} dB")

                # Force immediate UI update after key press
                snap = engine.snapshot()
                live.update(render_dashboard(snap, ctx, console.width, time.monotonic()), refresh=True)
                last_render = time.monotonic()
    finally:
        hotkeys.stop()
        if tray_daemon:
            tray_daemon.stop()

    if ctx.preset != app.preset:
        app.preset = ctx.preset
        app.save()
        console.print(f"[dim]Saved preset '{ctx.preset}' as default.[/]")
