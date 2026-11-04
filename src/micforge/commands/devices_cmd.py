"""`devices`, `setup`, `presets`."""
from __future__ import annotations

import json

from rich import box
from rich.prompt import Prompt
from rich.table import Table

from ..audio.devices import DeviceInfo, detect_virtual_output, list_devices
from ..config import AppConfig
from ..presets import get_preset, list_presets, user_preset_dir
from ..ui.theme import console


def show_devices(all_apis: bool = False) -> list[DeviceInfo]:
    devs = list_devices(all_apis)
    cable = detect_virtual_output(devs)
    t = Table(box=box.SIMPLE_HEAD, header_style="bold cyan")
    for c in ("#", "name", "api", "in", "out", ""):
        t.add_column(c, justify="right" if c in ("#", "in", "out") else "left")
    for d in devs:
        tags = []
        if d.is_default_input:
            tags.append("default in")
        if d.is_default_output:
            tags.append("default out")
        if cable is not None and d.index == cable.index:
            tags.append("[green]virtual cable[/]")
        t.add_row(str(d.index), d.name, d.hostapi, str(d.max_input or ""), str(d.max_output or ""), ", ".join(tags))
    console.print(t)
    return devs


def _pick(devs: list[DeviceInfo], kind: str, prompt: str, current, allow_skip: bool = False):
    pool = [d for d in devs if (d.max_input if kind == "input" else d.max_output) > 0]
    t = Table(box=box.SIMPLE_HEAD, header_style="bold cyan")
    t.add_column("#", justify="right")
    t.add_column("name")
    t.add_column("api")
    for d in pool:
        t.add_row(str(d.index), d.name, d.hostapi)
    console.print(t)
    valid = {d.index for d in pool}
    hint = f" (Enter = keep {current!r})" if current not in (None, "") else " (Enter = system default)"
    while True:
        raw = Prompt.ask(prompt + hint, default="", show_default=False, console=console).strip()
        if raw == "":
            return current
        if raw.isdigit() and int(raw) in valid:
            return next(d.name for d in pool if d.index == int(raw))  # store the name: indices change between boots
        console.print("[red]Not a valid device number.[/]")


def run_setup(app: AppConfig) -> None:
    devs = list_devices()
    console.rule("[bold cyan]Setup")
    console.print("[bold]1/3  Your microphone[/]")
    app.input_device = _pick(devs, "input", "Mic device #", app.input_device)
    cable = detect_virtual_output(devs)
    console.print("\n[bold]2/3  Virtual cable output[/] (this is what Discord listens to)")
    if cable:
        console.print(f"Detected: [green]{cable.name}[/]")
    app.output_device = _pick(devs, "output", "Cable device #", app.output_device or (cable.name if cable else None))
    console.print("\n[bold]3/3  Your headphones[/] (for monitoring and test playback)")
    app.monitor_device = _pick(devs, "output", "Headphones #", app.monitor_device)
    app.save()
    console.print("\n[green]✔ Saved.[/]")
    console.print(
        "\n[bold]Now in Discord:[/] Settings → Voice & Video → Input Device = the cable's [bold]recording[/] end "
        "(Windows: [cyan]CABLE Output[/]). Set Noise Suppression to [bold]None[/], and turn off Automatic Gain Control "
        "and Echo Cancellation so your voice isn't processed twice."
    )


def show_presets(name: str | None) -> None:
    if name:
        console.print_json(json.dumps(get_preset(name).to_dict()))
        return
    t = Table(box=box.SIMPLE_HEAD, header_style="bold cyan")
    t.add_column("preset")
    t.add_column("description")
    for p in list_presets():
        t.add_row(p.name + ("" if p.builtin else " *"), p.description)
    console.print(t)
    console.print(f"[dim]`micforge presets NAME` prints a preset as JSON. Drop edited copies into {user_preset_dir()} to use them.[/]")
