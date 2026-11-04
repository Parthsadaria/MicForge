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
    r.add_argument("-p", "--preset")
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
    l.add_argument("-p", "--preset")
    l.add_argument("--mic", type=_device_arg)
    l.add_argument("--out", type=_device_arg, help="headphones device")
    l.add_argument("--model", help="swappable CoFi-Lite model file (.onnx or .tar)")

    t = sub.add_parser("test", help="record a clip, show before/after analysis, play both back")
    t.add_argument("-p", "--preset")
    t.add_argument("-s", "--seconds", type=float, default=8.0)
    t.add_argument("--mic", type=_device_arg)
    t.add_argument("--out", type=_device_arg)
    t.add_argument("--model", help="swappable CoFi-Lite model file (.onnx or .tar)")

    c = sub.add_parser("calibrate", help="learn your room noise (improves denoise + sets the gate)")
    c.add_argument("-s", "--seconds", type=float, default=4.0)
    c.add_argument("--mic", type=_device_arg)
    c.add_argument("--reset", action="store_true", help="delete the saved noise profile")

    d = sub.add_parser("devices", help="list audio devices")
    d.add_argument("--all", action="store_true", help="show every host API (MME, DirectSound, ...)")

    sub.add_parser("setup", help="choose mic / cable / headphones interactively")

    pr = sub.add_parser("presets", help="list presets, or print one as JSON")
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

    sub.add_parser("doctor", help="check your setup and performance")
    sub.add_parser("version", help="show micforge version and system information")
    sub.add_parser("settings", help="open GUI to customize global hotkeys, audio cues, and startup")
    sub.add_parser("hotkeys", help="alias for settings")
    return p


def main(argv: list[str] | None = None) -> int:
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
        elif cmd == "presets":
            from .commands.devices_cmd import show_presets
            show_presets(args.name)
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
            return run_doctor(app)
        elif cmd in ("settings", "hotkeys"):
            from .ui.settings_gui import launch_settings_gui
            launch_settings_gui(app, standalone=True)
            return 0
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
