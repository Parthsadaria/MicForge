"""Command-line entry point."""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import AppConfig
from .errors import MicforgeError
from .logging_setup import setup_logging
from .ui.theme import console


def _device_arg(v: str):
    return int(v) if v.isdigit() else v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="micforge",
        description="Real-time mic clean-up for Discord & co: de-muffle, denoise, level, route via a virtual cable.",
    )
    p.add_argument("-v", "--version", action="version", version=f"micforge {__version__}")
    p.add_argument("--debug", action="store_true", help="verbose log file")
    p.add_argument("--model", help="path to swappable CoFi-Lite model file (.onnx or .tar)")
    p.add_argument("--tray", action="store_true", help="start live enhancer in background system tray mode")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="live: mic -> processing -> virtual cable (what Discord hears)")
    r.add_argument("-p", "--preset", help="DSP preset to load (e.g. streaming, gaming, podcast, noisy_room)")
    r.add_argument("--mic", type=_device_arg, help="input device (index or name part)")
    r.add_argument("--cable", type=_device_arg, help="virtual cable output (index or name part)")
    r.add_argument("--monitor", nargs="?", const="", type=_device_arg, metavar="DEVICE",
                   help="also play the processed mic to your headphones")
    r.add_argument("--rate", type=int, help="sample rate (default from config: 48000)")
    r.add_argument("--block", type=int, help="block size in samples (default 480)")
    r.add_argument("--model", help="swappable CoFi-Lite model file (.onnx or .tar)")
    r.add_argument("--tray", action="store_true", help="run in background with system tray icon")
    r.add_argument("--no-tray", action="store_true", help="disable system tray icon")

    l = sub.add_parser("loopback", help="hear yourself live through headphones, A/B raw vs processed")
    l.add_argument("-p", "--preset", help="DSP preset to load (default: active preset)")
    l.add_argument("--mic", type=_device_arg, help="input microphone (index or substring)")
    l.add_argument("--out", type=_device_arg, help="headphones device (index or substring)")
    l.add_argument("--model", help="swappable CoFi-Lite model file (.onnx or .tar)")

    t = sub.add_parser("test", help="record a clip, show before/after analysis, play both back")
    t.add_argument("-p", "--preset", help="DSP preset to test (default: active preset)")
    t.add_argument("-s", "--seconds", type=float, default=8.0, help="duration in seconds to record sample")
    t.add_argument("--mic", type=_device_arg, help="microphone device (index or substring)")
    t.add_argument("--out", type=_device_arg, help="playback headphones device (index or substring)")
    t.add_argument("--model", help="swappable CoFi-Lite model file (.onnx or .tar)")

    c = sub.add_parser("calibrate", help="learn your room noise (improves denoise + sets the gate)")
    c.add_argument("-s", "--seconds", type=float, default=4.0, help="duration in seconds of ambient silence to sample")
    c.add_argument("--mic", type=_device_arg, help="microphone device (index or substring)")
    c.add_argument("--reset", action="store_true", help="delete the saved noise profile")

    d = sub.add_parser("devices", help="list audio devices")
    d.add_argument("--all", action="store_true", help="show every host API (MME, DirectSound, ...)")

    sub.add_parser("setup", help="choose mic / cable / headphones interactively")

    # Preset management suite
    p_cmd = sub.add_parser("preset", help="manage and create custom presets")
    p_sub = p_cmd.add_subparsers(dest="preset_action")
    p_sub.add_parser("list", help="list all presets")
    p_save = p_sub.add_parser("save", help="create or save a custom preset")
    p_save.add_argument("name", help="new preset name")
    p_save.add_argument("--from", dest="from_preset", help="base preset to clone from")
    p_save.add_argument("--desc", default="", help="human description for the preset")
    p_save.add_argument("--force", action="store_true", help="overwrite if exists")
    p_copy = p_sub.add_parser("copy", help="duplicate an existing preset")
    p_copy.add_argument("source", help="source preset")
    p_copy.add_argument("target", help="target new preset name")
    p_copy.add_argument("--force", action="store_true", help="overwrite if exists")
    p_del = p_sub.add_parser("delete", help="delete a custom preset")
    p_del.add_argument("name", help="preset name to delete")
    p_del.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    p_exp = p_sub.add_parser("export", help="export preset to portable JSON")
    p_exp.add_argument("name", help="preset to export")
    p_exp.add_argument("target", nargs="?", help="output file path")
    p_imp = p_sub.add_parser("import", help="import preset from JSON file")
    p_imp.add_argument("source", help="path to preset JSON file")
    p_imp.add_argument("--name", default=None, help="custom name for imported preset")
    p_imp.add_argument("--force", action="store_true", help="overwrite if exists")
    p_show = p_sub.add_parser("show", help="show preset details")
    p_show.add_argument("name", help="preset name")

    pr = sub.add_parser("presets", help="list presets, or print one as JSON (alias for `micforge preset`)")
    pr.add_argument("name", nargs="?")

    f = sub.add_parser("process", help="process a WAV file offline")
    f.add_argument("input")
    f.add_argument("output")
    f.add_argument("-p", "--preset")
    f.add_argument("--model", help="swappable CoFi-Lite model file (.onnx or .tar)")

    a = sub.add_parser("analyze", help="analyse a WAV file (level, noise, clarity) and give advice")
    a.add_argument("input")

    m_cmd = sub.add_parser("model", help="inspect or export swappable CoFi-Lite models")
    m_cmd.add_argument("action", choices=["info", "export"], nargs="?", default="info")
    m_cmd.add_argument("checkpoint", nargs="?", help="source .tar / .pth checkpoint")
    m_cmd.add_argument("output", nargs="?", help="output .onnx file")

    doc = sub.add_parser("doctor", help="check your setup and performance (plain-English diagnostics)")
    doc.add_argument("--export", nargs="?", const="", help="export diagnostic report as a redacted .zip bundle")

    lat = sub.add_parser("latency", help="measure end-to-end audio pipeline latency with loopback probe")
    lat.add_argument("--mic", type=_device_arg, help="input microphone device (index or substring)")
    lat.add_argument("--out", type=_device_arg, help="output playback device (index or substring)")

    prof = sub.add_parser("profile", help="manage noise profiles (quiet_room, fan_on, mechanical_keyboard)")
    prof.add_argument("action", choices=["list", "show", "relearn", "switch"], default="list", nargs="?",
                      help="profile action to perform (list, show, relearn, switch)")
    prof.add_argument("name", nargs="?", help="profile / environment name")

    sub.add_parser("version", help="show micforge version and system information")
    sub.add_parser("settings", help="open GUI to customize global hotkeys, audio cues, and startup")
    sub.add_parser("hotkeys", help="alias for settings")
    help_p = sub.add_parser("help", help="show comprehensive command guide and dynamic usage examples")
    help_p.add_argument("command", nargs="?", help="specific command to inspect (e.g. run, latency, profile, doctor)")
    return p


def main(argv: list[str] | None = None) -> int:
    import multiprocessing
    multiprocessing.freeze_support()
    args = build_parser().parse_args(argv)
    setup_logging(args.debug)
    try:
        app = AppConfig.load()
        cmd = args.cmd
        model_arg = getattr(args, "model", None)
        if model_arg:
            app.denoise_model = model_arg
        if cmd is None and getattr(args, "tray", False):
            cmd = "run"
        if cmd is None:
            from .commands.wizard import run_wizard
            run_wizard(app)
        elif cmd == "version":
            from . import __version__
            from .dsp.cofilite import find_model_file
            console.print(f"[bold cyan]micforge[/] [bold green]v{__version__}[/]")
            active = find_model_file(app.denoise_model)
            model_str = f"CoFi-Lite ({active.name})" if active else "Spectral Wiener"
            console.print(f"  [dim]Active Model:[/] {model_str}")
            console.print(f"  [dim]Python:[/] {sys.version.split()[0]} on {sys.platform}")
            return 0
        elif cmd == "run":
            from .commands.live import run_live
            if args.rate:
                app.sample_rate = args.rate
            if args.block:
                app.block_size = args.block
            app.validate()
            run_live(app, preset=args.preset, monitor=args.monitor is not None, mic=args.mic, cable=args.cable,
                     monitor_dev=(args.monitor or None) if args.monitor is not None else None,
                     model=model_arg, tray=getattr(args, "tray", False), no_tray=getattr(args, "no_tray", False))
        elif cmd == "loopback":
            from .commands.live import run_live
            run_live(app, preset=args.preset, loopback=True, mic=args.mic, monitor_dev=args.out, model=model_arg)
        elif cmd == "test":
            from .commands.test_mic import run_test
            run_test(app, preset=args.preset, seconds=max(4.0, args.seconds), mic=args.mic, out=args.out)
        elif cmd == "calibrate":
            from .commands.calibrate import run_calibrate
            run_calibrate(app, seconds=max(2.0, args.seconds), mic=args.mic, reset=args.reset)
        elif cmd == "devices":
            from .commands.devices_cmd import show_devices
            show_devices(args.all)
        elif cmd == "setup":
            from .commands.devices_cmd import run_setup, show_devices
            run_setup(app)
        elif cmd in ("preset", "presets"):
            from .commands.preset_cmd import (
                run_preset_copy,
                run_preset_delete,
                run_preset_export,
                run_preset_import,
                run_preset_list,
                run_preset_save,
                run_preset_show,
            )
            action = getattr(args, "preset_action", None)
            if cmd == "presets":
                if args.name:
                    run_preset_show(args.name)
                else:
                    run_preset_list(app)
            elif action in (None, "list"):
                run_preset_list(app)
            elif action == "save":
                run_preset_save(app, args.name, from_preset=args.from_preset, description=args.desc, force=args.force)
            elif action == "copy":
                run_preset_copy(args.source, args.target, force=args.force)
            elif action == "delete":
                run_preset_delete(args.name, force=args.yes)
            elif action == "export":
                run_preset_export(args.name, args.target)
            elif action == "import":
                run_preset_import(args.source, name=args.name, force=args.force)
            elif action == "show":
                run_preset_show(args.name)
        elif cmd == "process":
            from .commands.files import run_process
            run_process(app, args.input, args.output, args.preset)
        elif cmd == "analyze":
            from .commands.files import run_analyze
            run_analyze(app, args.input)
        elif cmd == "model":
            from pathlib import Path
            from .dsp.cofilite import find_model_file
            if args.action == "info":
                active = find_model_file(model_arg or app.denoise_model)
                if active and active.is_file():
                    size_kb = active.stat().st_size / 1024
                    console.print(f"[bold green]✔ Active CoFi-Lite model:[/] {active} ({size_kb:.1f} KB)")
                    console.print("  [dim]Type:[/] ONNX neural speech enhancement (~83k params)")
                    console.print("  [dim]Memory:[/] ~65-75 MB RAM (well under 100 MB limit)")
                else:
                    console.print("[yellow]! No CoFi-Lite model file found. Operating with fallback Spectral Wiener denoiser.[/]")
            elif args.action == "export":
                if not args.checkpoint:
                    console.print("[red]✖ Please provide checkpoint path: micforge model export <checkpoint.tar> [out.onnx][/]")
                    return 1
                from .dsp.export_cofilite import export_checkpoint_to_onnx
                target = args.output or Path(args.checkpoint).with_suffix(".onnx")
                export_checkpoint_to_onnx(args.checkpoint, target)
                console.print(f"[bold green]✔ Successfully exported CoFi-Lite model to:[/] {target}")
        elif cmd == "doctor":
            from .commands.doctor import run_doctor
            export_flag = args.export is not None
            export_path = args.export if (args.export and len(args.export) > 0) else None
            return run_doctor(app, export=export_flag, export_path=export_path)
        elif cmd == "latency":
            from .commands.latency_test import run_latency_test
            return run_latency_test(app, mic=args.mic, out=args.out)
        elif cmd == "profile":
            action = args.action or "list"
            if action == "list":
                from .audio.noise_profile import BUILTIN_ENVIRONMENTS, list_noise_profiles
                profiles = list_noise_profiles()
                console.print("\n[bold cyan]🎙️ Available Noise Profiles & Environments:[/]")
                for p in profiles:
                    tag = " [green](builtin)[/]" if p in BUILTIN_ENVIRONMENTS else " [dim](custom)[/]"
                    console.print(f"  • [bold white]{p}[/]{tag}")
                console.print("\n[dim]Usage: micforge profile switch <name>  │  micforge profile show <name>[/]\n")
                return 0
            elif action == "show":
                from .audio.noise_profile import BUILTIN_ENVIRONMENTS, AdaptiveNoiseProfiler
                prof_name = args.name or "quiet_room"
                profiler = AdaptiveNoiseProfiler(sample_rate=app.sample_rate, environment=prof_name)
                p = profiler.current_profile
                console.print(f"\n[bold cyan]Noise Profile:[/] [bold green]{p.name}[/]")
                console.print(f"  • Estimated Floor:    [yellow]{p.noise_floor_db:.1f} dBFS[/]")
                console.print(f"  • Gate Threshold:     [yellow]{p.gate_threshold_db:.1f} dBFS[/]")
                console.print(f"  • Spectral Tilt:      [dim]{p.spectral_tilt:+.2f}[/]")
                console.print(f"  • Status:             {'[cyan]Frozen[/]' if p.frozen else '[green]Active Learner[/]'}\n")
                return 0
            elif action == "switch":
                if not args.name:
                    console.print("[red]✖ Please specify environment profile name (e.g. quiet_room, fan_on, mechanical_keyboard)[/]")
                    return 1
                from .audio.noise_profile import AdaptiveNoiseProfiler
                profiler = AdaptiveNoiseProfiler(sample_rate=app.sample_rate, environment=args.name)
                profiler.save()
                console.print(f"[bold green]✔ Switched active noise profile environment to:[/] [bold cyan]{args.name}[/]")
                return 0
            elif action == "relearn":
                console.print("[bold cyan]🔄 Requesting immediate noise profile relearn…[/]")
                console.print("[dim]Next live or test session will automatically sample ambient silence.[/]")
                return 0
        elif cmd in ("settings", "hotkeys"):
            from .ui.settings_gui import launch_settings_gui
            launch_settings_gui(app, standalone=True)
            return 0
        elif cmd == "help":
            from .commands.help_cmd import run_help
            return run_help(args.command)
        return 0
    except MicforgeError as exc:
        console.print(f"[bold red]✖ {exc}[/]")
        if exc.hint:
            console.print(f"[yellow]  → {exc.hint}[/]")
        return 2
    except KeyboardInterrupt:
        console.print("\n[dim]bye[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
