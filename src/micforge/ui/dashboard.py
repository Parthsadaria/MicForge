"""Live dashboard renderer: pure function of (snapshot, context, width)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import __version__
from ..audio.engine import EngineSnapshot
from .widgets import fmt_db, freq_axis, level_bar, load_bar, reduction_bar, spectrum_rows


@dataclass
class LiveContext:
    title: str = "MIC ENHANCER"
    preset: str = "discord"
    input_name: str = ""
    cable_name: str | None = None
    monitor_name: str | None = None
    sample_rate: int = 48000
    block_size: int = 480
    message: str = ""
    message_until: float = 0.0
    centers: tuple[float, float] = (60.0, 16000.0)
    n_bands: int = 32
    has_profile: bool = False
    keys: list[tuple[str, str]] = field(default_factory=list)


def _status(snap: EngineSnapshot) -> Text:
    t = Text()
    if snap.muted:
        t.append(" ■ MUTED ", style="bold white on red")
    elif snap.bypass:
        t.append(" ◌ BYPASS (raw mic) ", style="bold black on yellow")
    else:
        t.append(" ● PROCESSED ", style="bold black on green")
    if snap.monitor_on:
        t.append("  ")
        t.append(" 🎧 MONITOR ", style="bold black on cyan")
    return t


def _meters(snap: EngineSnapshot, width: int) -> Table:
    bar_w = max(16, width - 30)
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(width=4, style="bold")
    tbl.add_column(width=bar_w)
    tbl.add_column(width=9, justify="right")
    tbl.add_column(width=6)
    for label, lv in (("IN", snap.in_level), ("OUT", snap.out_level)):
        clip = Text("CLIP", style="bold white on red") if lv.clipping else Text("")
        tbl.add_row(label, level_bar(lv.peak_db, lv.hold_db, bar_w), Text(f"{fmt_db(lv.peak_db)} dB"), clip)
    return tbl


def _reductions(snap: EngineSnapshot, width: int) -> Table:
    bar_w = max(10, min(28, (width - 40) // 2))
    tel = snap.telemetry
    rows = [
        ("noise", "denoise", "reduction_db", "magenta"),
        ("de-ess", "deesser", "gain_reduction_db", "magenta"),
        ("comp", "compressor", "gain_reduction_db", "cyan"),
        ("limit", "limiter", "gain_reduction_db", "red"),
    ]
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(width=7, style="grey70")
    tbl.add_column(width=bar_w)
    tbl.add_column(width=8, justify="right")
    for label, stage, key, color in rows:
        if stage in tel:
            if stage == "denoise" and tel[stage].get("napping", 0.0) > 0.5:
                tbl.add_row(label, Text("💤 IDLE NAP (0% CPU)".ljust(bar_w), style="dim cyan"), "-40 dB")
            else:
                v = tel[stage].get(key, 0.0)
                tbl.add_row(label, reduction_bar(v, bar_w, 24.0, color), f"-{v:4.1f} dB")
        else:
            tbl.add_row(label, Text("off".ljust(bar_w), style="grey37"), "")
    gate = tel.get("gate")
    if gate is not None:
        state = Text("OPEN ", style="bold green") if gate["open"] > 0.5 else Text("closed", style="grey50")
        tbl.add_row("gate", state, f"{gate['gain_db']:5.1f} dB")
    lev = tel.get("leveler")
    if lev is not None:
        tbl.add_row("level", Text("auto gain", style="grey50"), f"{lev['gain_db']:+5.1f} dB")
    return tbl


def _spectrum(snap: EngineSnapshot, ctx: LiveContext, width: int) -> Table:
    panel_w = max(20, (width - 10) // 2)
    rep = max(1, min(2, panel_w // ctx.n_bands))
    f_lo, f_hi = ctx.centers
    height = 5
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column()
    tbl.add_column()
    tbl.add_row(Text("raw mic", style="grey70"), Text("processed", style="bold green"))
    left = spectrum_rows(snap.in_bands, height, rep, style="blue")
    right = spectrum_rows(snap.out_bands, height, rep, style="green")
    for a, b in zip(left, right):
        tbl.add_row(a, b)
    axis = freq_axis(ctx.n_bands, rep, f_lo, f_hi)
    tbl.add_row(axis, axis)
    return tbl


def _stats(snap: EngineSnapshot, ctx: LiveContext) -> Table:
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style="grey70")
    tbl.add_column()
    tbl.add_column(style="grey70")
    tbl.add_column()
    noise = snap.telemetry.get("denoise", {}).get("noise_floor_db")
    xr = snap.input_overflows + snap.output_underruns
    tbl.add_row("latency", f"~{snap.latency_ms:4.0f} ms", "noise floor", f"{noise:5.1f} dBFS" if noise is not None else "-")
    cpu = Text()
    cpu.append_text(load_bar(snap.cpu_load, 10))
    cpu.append(f" {snap.cpu_load * 100:3.0f}%")
    tbl.add_row("dsp load", cpu, "input gain", f"{snap.input_gain_db:+.1f} dB")
    glitch_style = "green" if xr == 0 else "yellow" if xr < 5 else "bold red"
    tbl.add_row("glitches", Text(f"{xr}", style=glitch_style), "clock fixes", f"{snap.drift_corrections}")
    return tbl


def _routing(ctx: LiveContext) -> Text:
    t = Text()
    t.append("mic ", style="grey70")
    t.append(ctx.input_name[:34], style="white")
    if ctx.cable_name:
        t.append("  →  ", style="cyan")
        t.append(ctx.cable_name[:34], style="white")
    if ctx.monitor_name:
        t.append("   🎧 ", style="grey70")
        t.append(ctx.monitor_name[:28], style="white")
    return t


def render_dashboard(snap: EngineSnapshot, ctx: LiveContext, width: int, now: float) -> RenderableType:
    header = Text.assemble((f" {ctx.title} ", "bold white on blue"), "  ", _status(snap), "  ",
                           ("preset ", "grey70"), (ctx.preset, "bold cyan"))
    parts: list[RenderableType] = [header, _routing(ctx), Text("")]
    parts.append(_meters(snap, width))
    parts.append(Text(""))
    parts.append(Panel(_reductions(snap, width), title="processing", title_align="left", box=box.ROUNDED,
                       border_style="grey37", padding=(0, 1)))
    parts.append(Panel(_spectrum(snap, ctx, width), title="spectrum", title_align="left", box=box.ROUNDED,
                       border_style="grey37", padding=(0, 1)))
    parts.append(_stats(snap, ctx))

    if snap.calibrating is not None:
        pct = int(snap.calibrating * 100)
        parts.append(Text(f"\n  ◉ Calibrating room noise… stay quiet ({pct}%)", style="bold yellow"))
    elif ctx.message and now < ctx.message_until:
        parts.append(Text(f"\n  {ctx.message}", style="bold cyan"))
    else:
        parts.append(Text(""))

    foot = Text()
    for k, label in ctx.keys:
        foot.append(f" {k} ", style="key")
        foot.append(f" {label}  ", style="grey70")
    parts.append(foot)
    return Panel(Group(*parts), title=f"[bold]micforge[/] [dim]v{__version__}[/]", border_style="cyan", box=box.ROUNDED)
