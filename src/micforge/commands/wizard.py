"""Interactive menu shown when `micforge` is run with no arguments."""
from __future__ import annotations

from rich.panel import Panel
from rich.prompt import Prompt

from .. import __version__
from ..config import AppConfig
from ..errors import MicforgeError
from ..ui.theme import console
from .calibrate import run_calibrate
from .devices_cmd import run_setup, show_devices, show_presets
from .doctor import run_doctor
from .live import run_live
from .test_mic import run_test

MENU = [
    ("1", "Start enhancer  (mic → virtual cable → Discord)"),
    ("2", "Mic test       (record, compare, play back raw vs processed)"),
    ("3", "Loopback       (hear yourself live, A/B raw vs processed)"),
    ("4", "Calibrate room noise"),
    ("5", "Setup devices"),
    ("6", "Presets"),
    ("7", "Settings & Hotkeys (global hotkeys, audio cues, autostart)"),
    ("8", "Doctor         (diagnose problems)"),
    ("q", "Quit"),
]


def run_wizard(app: AppConfig) -> None:
    while True:
        body = "\n".join(f"[bold cyan]{k}[/]  {v}" for k, v in MENU)
        console.print(Panel(body, title=f"[bold]micforge[/] v{__version__}", subtitle=f"preset: {app.preset}",
                            border_style="cyan", padding=(1, 3)))
        choice = Prompt.ask("Choose", choices=[k for k, _ in MENU], default="1", console=console)
        try:
            if choice == "q":
                return
            if choice == "1":
                run_live(app)
            elif choice == "2":
                run_test(app)
            elif choice == "3":
                run_live(app, loopback=True)
            elif choice == "4":
                run_calibrate(app)
            elif choice == "5":
                show_devices()
                run_setup(app)
            elif choice == "6":
                show_presets(None)
                name = Prompt.ask("Set default preset (Enter to skip)", default="", console=console).strip()
                if name:
                    from ..presets import get_preset
                    get_preset(name)  # validates
                    app.preset = name
                    app.save()
                    console.print(f"[green]Default preset: {name}[/]")
            elif choice == "7":
                from ..ui.settings_gui import launch_settings_gui
                launch_settings_gui(app, standalone=True)
            elif choice == "8":
                run_doctor(app)
        except MicforgeError as exc:
            console.print(f"\n[bold red]✖ {exc}[/]")
            if exc.hint:
                console.print(f"[yellow]  → {exc.hint}[/]")
        except KeyboardInterrupt:
            console.print()
        console.print()
