"""`doctor`: environment, device and performance checks with fix-it hints."""
from __future__ import annotations

import platform
import sys
import time

import numpy as np
from rich import box
from rich.table import Table

from ..audio.backend import get_sd
from ..audio.devices import detect_virtual_output, find_device, list_devices
from ..config import AppConfig
from ..dsp.chain import process_offline
from ..errors import MicforgeError
from ..ui.theme import console
from .common import make_chain

OK, WARN, BAD = "[green]✔[/]", "[yellow]![/]", "[red]✖[/]"

CABLE_HELP = {
    "win32": "Install VB-CABLE (free): https://vb-audio.com/Cable/ - reboot, then re-run `micforge setup`.",
    "darwin": "Install BlackHole 2ch: `brew install blackhole-2ch` - then re-run `micforge setup`.",
    "linux": "Create a virtual sink: `pactl load-module module-null-sink sink_name=micforge` "
    "- Discord then uses 'Monitor of micforge' as its mic.",
}


def run_doctor(app: AppConfig, export: bool = False, export_path: str | None = None) -> int:
    t = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", show_lines=False)
    t.add_column("", width=2)
    t.add_column("check")
    t.add_column("result")
    problems = 0

    def row(mark: str, name: str, result: str) -> None:
        t.add_row(mark, name, result)

    row(OK if sys.version_info >= (3, 10) else BAD, "python", f"{platform.python_version()} on {platform.system()}")
    try:
        import scipy
        row(OK, "numpy / scipy", f"{np.__version__} / {scipy.__version__}")
    except ImportError:
        row(BAD, "scipy", "missing - pip install scipy")
        problems += 1

    try:
        sd = get_sd()
        row(OK, "PortAudio", str(sd.get_portaudio_version()[1]))
    except MicforgeError as exc:
        row(BAD, "PortAudio", f"{exc} {exc.hint or ''}")
        console.print(t)
        return 1

    pool = list_devices(all_apis=True)
    try:
        mic = find_device(app.input_device, "input", pool)
        row(OK, "microphone", f"{mic.name} [{mic.hostapi}]")
        try:
            sd.check_input_settings(device=mic.index, channels=1, samplerate=app.sample_rate)
            row(OK, f"mic @ {app.sample_rate} Hz", "supported")
        except Exception as exc:  # noqa: BLE001
            row(WARN, f"mic @ {app.sample_rate} Hz", f"not accepted ({exc}); WASAPI auto-convert may still work, or try 44100")
    except MicforgeError as exc:
        row(BAD, "microphone", str(exc))
        problems += 1

    cable = detect_virtual_output(pool) if app.output_device is None else None
    if app.output_device is not None:
        try:
            cable = find_device(app.output_device, "output", pool)
        except MicforgeError as exc:
            row(BAD, "virtual cable", str(exc))
            problems += 1
    if cable:
        row(OK, "virtual cable", cable.name)
    elif app.output_device is None:
        row(BAD, "virtual cable", "none found - " + CABLE_HELP.get(sys.platform if sys.platform in CABLE_HELP else "linux", ""))
        problems += 1

    try:
        hp = find_device(app.monitor_device, "output", pool)
        row(OK, "headphones / monitor", hp.name)
    except MicforgeError as exc:
        row(WARN, "headphones / monitor", str(exc))

    from ..dsp.cofilite import find_model_file
    model_file = find_model_file(app.denoise_model)
    if model_file:
        row(OK, "denoise model", f"CoFi-Lite ({model_file.name}, {model_file.stat().st_size / 1024:.0f} KB) [<100MB RAM]")
    else:
        row(OK, "denoise model", "Spectral Wiener (classical fallback)")

    chain, _ = make_chain(app, app.preset)
    x = (np.random.default_rng(0).standard_normal(app.sample_rate * 2) * 0.05).astype(np.float32)
    process_offline(chain, x[: app.sample_rate // 2], app.block_size)  # warm-up (numpy/FFT plan caches)
    chain, _ = make_chain(app, app.preset)
    t0 = time.perf_counter()
    process_offline(chain, x, app.block_size)
    rt = (x.shape[0] / app.sample_rate) / (time.perf_counter() - t0)
    row(OK if rt > 5 else WARN if rt > 2 else BAD, "dsp speed", f"{rt:.0f}x realtime ({100 / rt:.1f}% of one core)")
    if rt <= 2:
        problems += 1
    row(OK, "latency (dsp only)", f"{chain.latency_samples / app.sample_rate * 1000:.1f} ms")

    console.print(t)
    console.print(
        "[bold]Discord checklist:[/] Input Device = cable's recording end · Noise Suppression: None · "
        "Automatic Gain Control: off · Echo Cancellation: off · Input Sensitivity: manual, low."
    )
    console.print(
        "[bold]Windows tip:[/] Sound settings → your mic → Properties → Advanced: turn off 'Audio Enhancements' "
        "and exclusive mode, or other apps can fight for the mic."
    )
    console.print(
        "[dim]📖 Troubleshooting guide:[/] [bold cyan]docs/TROUBLESHOOTING.md[/]"
    )

    if export or export_path:
        from pathlib import Path
        from ..diagnostics import collect_diagnostics, export_diagnostics_zip
        report = collect_diagnostics(app)
        out_zip = export_diagnostics_zip(report, Path(export_path) if export_path else None)
        console.print(f"\n[bold green]✔ Diagnostics exported to:[/] [white]{out_zip}[/]")

    return 1 if problems else 0
