"""Live dashboard renderer: pure function of (snapshot, context, width)."""
from __future__ import annotations

import time
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

BLOCKS = " ▁▂▃▄▅▆▇█"


def get_process_ram_mb() -> float:
    """Read process RAM working set safely via psutil."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def format_uptime(start_time: float, now: float) -> str:
    """Format elapsed session time into human readable string."""
    elapsed = max(0, int(now - start_time))
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}h {mins:02d}m {secs:02d}s"
    return f"{mins:02d}m {secs:02d}s"


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
    start_time: float = field(default_factory=time.monotonic)
    hud_mode: bool = False  # Toggle between Studio Inspector and Gamer HUD
    show_help: bool = False  # Toggle full hotkey cheatsheet overlay


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


def _pipeline_ribbon(snap: EngineSnapshot) -> Text:
    """Render a sleek, single-line broadcast DSP rack ribbon that never wraps."""
    tel = snap.telemetry
    t = Text()
    t.append("Rack: ", style="grey50")

    is_neural = tel.get("denoise", {}).get("is_neural", 1.0) > 0.5
    dn_label = "AI" if is_neural else "WNR"
    stages = [
        ("HPF", True),
        ("AEC", "aec" in tel and tel["aec"].get("enabled", 1.0) > 0.5),
        (dn_label, "denoise" in tel and tel["denoise"].get("enabled", 1.0) > 0.5),
        ("VERB", "deverb" in tel and tel["deverb"].get("enabled", 1.0) > 0.5),
        ("GATE", "gate" in tel and tel["gate"].get("enabled", 1.0) > 0.5),
        ("BODY", "proximity" in tel and tel["proximity"].get("enabled", 1.0) > 0.5),
        ("EQ", True),
        ("DEESS", "deesser" in tel and tel["deesser"].get("enabled", 1.0) > 0.5),
        ("LVL", "leveler" in tel and tel["leveler"].get("enabled", 1.0) > 0.5),
        ("COMP", "compressor" in tel and tel["compressor"].get("enabled", 1.0) > 0.5),
        ("LIM", "limiter" in tel and tel["limiter"].get("enabled", 1.0) > 0.5),
    ]

    for i, (name, is_on) in enumerate(stages):
        if not is_on:
            t.append(f"{name}·", style="grey37")
        elif name in ("AI", "WNR") and tel.get("denoise", {}).get("napping", 0) > 0.5:
            t.append(f"{name}💤", style="dim cyan")
        elif name == "GATE" and tel.get("gate", {}).get("open", 0) <= 0.5:
            t.append(f"{name}·", style="grey50")
        else:
            t.append(f"{name}●", style="bold green" if name == "AI" else "bold yellow" if name == "WNR" else "cyan")

        if i < len(stages) - 1:
            t.append(" ", style="grey23")
    return t


def _meters(snap: EngineSnapshot, width: int) -> Table:
    """Render IN, OUT, and AEC REF meters with target sweet-spot indications."""
    bar_w = max(14, width - 36)
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(width=4, style="bold")
    tbl.add_column(width=bar_w)
    tbl.add_column(width=10, justify="right")
    tbl.add_column(width=16)

    # 1. Input Level
    clip_in = Text("CLIP", style="bold white on red") if snap.in_level.clipping else Text("")
    tbl.add_row(
        "IN",
        level_bar(snap.in_level.peak_db, snap.in_level.hold_db, bar_w),
        Text(f"{fmt_db(snap.in_level.peak_db)} dB"),
        clip_in,
    )

    # 2. Output Level with Target Sweet-Spot Indicator
    out_db = snap.out_level.peak_db
    if snap.out_level.clipping:
        target_badge = Text("CLIP", style="bold white on red")
    elif -24.0 <= out_db <= -10.0:
        target_badge = Text("[SWEET SPOT]", style="bold green")
    elif out_db > -10.0:
        target_badge = Text("[HOT / PEAK]", style="bold yellow")
    else:
        target_badge = Text("")

    tbl.add_row(
        "OUT",
        level_bar(out_db, snap.out_level.hold_db, bar_w),
        Text(f"{fmt_db(out_db)} dB"),
        target_badge,
    )

    # 3. Far-End Reference Stream Monitor (AEC speaker/game loopback)
    aec_active = ("aec" in snap.telemetry and snap.telemetry["aec"].get("enabled", 1.0) > 0.5)
    if aec_active and snap.ref_level is not None:
        ref_db = snap.ref_level.peak_db
        if ref_db <= -80.0:
            ref_text = "   silent"
            ref_badge = Text("(Clean / Headphones)", style="dim green")
        elif ref_db > -55.0:
            ref_text = f"{fmt_db(ref_db)} dB"
            ref_badge = Text("(Speaker Bleed Captured)", style="bold cyan")
        else:
            ref_text = f"{fmt_db(ref_db)} dB"
            ref_badge = Text("(Speaker Active)", style="dim cyan")

        tbl.add_row(
            "REF",
            level_bar(ref_db, snap.ref_level.hold_db, bar_w),
            Text(ref_text, style="grey50" if ref_db <= -80.0 else "white"),
            ref_badge,
        )
    return tbl


def _reductions(snap: EngineSnapshot, width: int) -> Table:
    """Render processing reductions cleanly, hiding idle stages to eliminate clutter."""
    bar_w = max(10, min(26, (width - 36) // 2))
    tel = snap.telemetry
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(width=8, style="grey70")
    tbl.add_column(width=bar_w)
    tbl.add_column(width=10, justify="right")

    # 1. AI Noise Suppression (always shown)
    if "denoise" in tel:
        if tel["denoise"].get("napping", 0.0) > 0.5:
            tbl.add_row("noise", Text("💤 IDLE NAP (0% CPU)".ljust(bar_w), style="dim cyan"), "-40 dB")
        else:
            v = abs(tel["denoise"].get("reduction_db", 0.0))
            tbl.add_row("noise", reduction_bar(v, bar_w, 28.0, "magenta"), f"-{v:4.1f} dB")

    # 2. Gate (always shown)
    gate = tel.get("gate")
    if gate is not None:
        is_open = gate.get("open", 0.0) > 0.5
        state = Text("OPEN ", style="bold green") if is_open else Text("closed", style="grey50")
        g_gain = abs(gate.get("gain_db", 0.0))
        tbl.add_row("gate", state, f"-{g_gain:4.1f} dB" if not is_open else " 0.0 dB")

    # 3. Room De-Verb (only shown if actively reducing > 0.5 dB)
    if "deverb" in tel:
        v = abs(tel["deverb"].get("reverb_reduction_db", 0.0))
        if v >= 0.5:
            tbl.add_row("de-verb", reduction_bar(v, bar_w, 18.0, "green"), f"-{v:4.1f} dB")

    # 4. Voice Proximity (only shown if boosting > 0.5 dB)
    if "proximity" in tel:
        v = abs(tel["proximity"].get("boost_db", 0.0))
        if v >= 0.5:
            tbl.add_row("body", reduction_bar(v, bar_w, 9.0, "yellow"), f"+{v:4.1f} dB")

    # 5. AEC Echo (only shown if AEC is enabled and actively working)
    if "aec" in tel and tel["aec"].get("enabled", 1.0) > 0.5:
        v = abs(tel["aec"].get("erle_db", 0.0))
        if tel["aec"].get("dtd_active", 0.0) > 0.5:
            tbl.add_row("echo", Text("TALKING".ljust(bar_w), style="bold yellow"), f"{v:4.1f} dB")
        elif v >= 1.0 or tel["aec"].get("echo_detected", 0.0) > 0.5:
            tbl.add_row("echo", reduction_bar(v, bar_w, 30.0, "blue"), f"-{v:4.1f} dB")

    # 6. Dynamics (Comp / Limit / De-Esser) - only light up when actively clamping > 0.5 dB
    comp_v = abs(tel.get("compressor", {}).get("gain_reduction_db", 0.0))
    if comp_v >= 0.5:
        tbl.add_row("comp", reduction_bar(comp_v, bar_w, 12.0, "cyan"), f"-{comp_v:4.1f} dB")

    lim_v = abs(tel.get("limiter", {}).get("gain_reduction_db", 0.0))
    if lim_v >= 0.5:
        tbl.add_row("limit", reduction_bar(lim_v, bar_w, 12.0, "red"), f"-{lim_v:4.1f} dB")

    deess_v = abs(tel.get("deesser", {}).get("gain_reduction_db", 0.0))
    if deess_v >= 0.5:
        tbl.add_row("de-ess", reduction_bar(deess_v, bar_w, 12.0, "magenta"), f"-{deess_v:4.1f} dB")

    return tbl


def _spectrum(snap: EngineSnapshot, ctx: LiveContext, width: int) -> Table:
    panel_w = max(20, (width - 10) // 2)
    rep = max(1, min(2, panel_w // ctx.n_bands))
    if isinstance(ctx.centers, (list, tuple)) and len(ctx.centers) == 2:
        f_lo, f_hi = float(ctx.centers[0]), float(ctx.centers[1])
    elif isinstance(ctx.centers, (list, tuple)) and len(ctx.centers) > 2:
        f_lo, f_hi = float(ctx.centers[0]), float(ctx.centers[-1])
    else:
        f_lo, f_hi = 60.0, 16000.0
    height = 3
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column()
    tbl.add_column()
    tbl.add_row(Text("raw mic", style="grey70"), Text("processed voice", style="bold green"))
    left = spectrum_rows(snap.in_bands, height, rep, style="blue")
    right = spectrum_rows(snap.out_bands, height, rep, style="green")
    for a, b in zip(left, right):
        tbl.add_row(a, b)
    axis = freq_axis(ctx.n_bands, rep, f_lo, f_hi)
    tbl.add_row(axis, axis)
    return tbl


def _stats(snap: EngineSnapshot, ctx: LiveContext, now: float) -> Table:
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style="grey70")
    tbl.add_column()
    tbl.add_column(style="grey70")
    tbl.add_column()
    tbl.add_column(style="grey70")
    tbl.add_column()

    noise = snap.telemetry.get("denoise", {}).get("noise_floor_db")
    xr = snap.input_overflows + snap.output_underruns
    uptime_str = format_uptime(ctx.start_time, now)
    ram_mb = get_process_ram_mb()

    tbl.add_row(
        "latency", f"~{snap.latency_ms:4.0f} ms",
        "noise floor", f"{noise:5.1f} dBFS" if noise is not None else "-",
        "uptime", Text(uptime_str, style="cyan"),
    )

    cpu = Text()
    cpu.append_text(load_bar(snap.cpu_load, 8))
    cpu.append(f" {snap.cpu_load * 100:3.0f}%")

    ram_str = f"{ram_mb:.1f} MB" if ram_mb > 0 else "-"
    tbl.add_row(
        "dsp load", cpu,
        "input gain", f"{snap.input_gain_db:+.1f} dB",
        "memory", Text(ram_str, style="dim white"),
    )

    glitch_style = "green" if xr == 0 else "yellow" if xr < 5 else "bold red"
    glitch_text = Text("PERFECT (0 drops)" if xr == 0 else f"{xr} (DROPPED)", style=glitch_style)
    drift_text = Text("0 (STABLE)" if snap.drift_corrections == 0 else f"{snap.drift_corrections} fixes", style="green" if snap.drift_corrections == 0 else "yellow")
    tbl.add_row(
        "audio sync", glitch_text,
        "clock drift", drift_text,
        "profile", Text("Active" if ctx.has_profile else "None", style="green" if ctx.has_profile else "grey50"),
    )

    timings = snap.telemetry.get("stage_timings", {})
    tot = timings.get("total", {})
    tot_p50 = tot.get("p50_us", 0.0)
    tot_p99 = tot.get("p99_us", 0.0)
    dn = timings.get("denoise", {})
    dn_p99 = dn.get("p99_us", 0.0)

    timing_style = "green" if tot_p99 < 3500 else "yellow" if tot_p99 < 5000 else "bold red"
    timing_text = Text(f"{tot_p50:.0f}µs (p99 {tot_p99:.0f}µs)" if tot_p99 > 0 else "measuring…", style=timing_style)
    ai_tail_text = Text(f"{dn_p99:.0f}µs" if dn_p99 > 0 else "-", style="dim cyan")

    tbl.add_row(
        "dsp timing", timing_text,
        "ai tail", ai_tail_text,
        "budget", Text("5333 µs", style="dim grey50"),
    )
    return tbl


def _routing(ctx: LiveContext) -> Text:
    t = Text()
    t.append("mic ", style="grey70")
    t.append(ctx.input_name[:32], style="white")
    if ctx.cable_name:
        t.append("  →  ", style="cyan")
        t.append(ctx.cable_name[:32], style="white")
    if ctx.monitor_name:
        t.append("   🎧 ", style="grey70")
        t.append(ctx.monitor_name[:24], style="white")
    return t


def _render_mini_presence_spectrum(out_bands: np.ndarray, width: int = 24) -> Text:
    """Render a single-line high-resolution vocal presence wave for Gamer HUD."""
    t = Text()
    t.append("Vocals: ", style="grey50")
    if len(out_bands) == 0:
        t.append("·" * width, style="grey37")
        return t

    # Subsample bands to target width
    step = max(1, len(out_bands) // width)
    sampled = out_bands[::step][:width]
    norm = np.clip((sampled + 90.0) / 55.0, 0.0, 1.0)
    for v in norm:
        idx = int(round(v * 8.0))
        ch = BLOCKS[min(8, max(0, idx))]
        col = "green" if idx > 4 else "cyan" if idx > 2 else "grey37"
        t.append(ch, style=col)
    return t


# =============================================================================
# GAMER HUD MODE: ULTRA-STREAMLINED, HIGH-TECH ESPORTS OVERLAY (TAB TOGGLED)
# =============================================================================
def render_gamer_hud(snap: EngineSnapshot, ctx: LiveContext, width: int, now: float) -> RenderableType:
    """Render compact, esports-optimized high-tech Gamer HUD."""
    uptime_str = format_uptime(ctx.start_time, now)
    ram_mb = get_process_ram_mb()
    tel = snap.telemetry

    # Top Status Bar
    header = Text()
    if snap.muted:
        header.append(" ■ MUTED ", style="bold white on red")
    elif snap.bypass:
        header.append(" ◌ BYPASS ", style="bold black on yellow")
    else:
        header.append(" ● LIVE HUD ", style="bold black on green")

    # Voice Activity Detector Badge (driven directly by CoFi-Lite neural mask VAD)
    gate = tel.get("gate", {})
    dn = tel.get("denoise", {})
    speech_prob = float(dn.get("speech_prob", 0.0))
    is_speech = dn.get("is_speech", 0.0) > 0.5 or speech_prob > 0.45 or gate.get("open", 0.0) > 0.5
    if snap.muted:
        vad = Text(" [🤐 SILENCED] ", style="bold red")
    elif is_speech:
        pct = int(speech_prob * 100)
        vad = Text(f" [🎙️ VOICED {pct}%] ", style="bold green") if pct > 0 else Text(" [🎙️ VOICED SPEECH] ", style="bold green")
    elif dn.get("napping", 0.0) > 0.5:
        vad = Text(" [💤 IDLE NAP] ", style="dim cyan")
    else:
        vad = Text(" [👂 LISTENING] ", style="dim grey50")

    header.append("  ")
    header.append_text(vad)
    header.append("  ")
    header.append("Preset: ", style="grey70")
    header.append(f"[{ctx.preset}]", style="bold cyan")
    header.append("  ")
    header.append(f"⏱ {uptime_str}", style="dim white")
    if ram_mb > 0:
        header.append(f" │ {ram_mb:.0f}MB", style="dim white")

    # Muted Giant Alert Box
    muted_alert = None
    if snap.muted:
        muted_alert = Panel(
            Text("■ MICROPHONE IS CURRENTLY MUTED (TEAMMATES CANNOT HEAR YOU)\nPress 'x' or Global Hotkey to Unmute", style="bold white on red", justify="center"),
            border_style="red",
            box=box.DOUBLE,
        )

    # Audio Specs & Live Compact Meters
    bar_w = max(10, min(24, (width - 44)))
    meter_tbl = Table.grid(padding=(0, 1))
    meter_tbl.add_column(width=4, style="bold")
    meter_tbl.add_column(width=bar_w)
    meter_tbl.add_column(width=10, justify="right")
    meter_tbl.add_column(width=14)
    meter_tbl.add_column(style="grey70")
    meter_tbl.add_column()

    # Line 1: IN
    in_clip = Text("CLIP", style="bold white on red") if snap.in_level.clipping else Text("")
    meter_tbl.add_row(
        "IN",
        level_bar(snap.in_level.peak_db, snap.in_level.hold_db, bar_w),
        Text(f"{fmt_db(snap.in_level.peak_db)} dB"),
        in_clip,
        "│ Latency:", f"~{snap.latency_ms:.0f}ms",
    )

    # Line 2: OUT
    out_db = snap.out_level.peak_db
    out_badge = Text("[TARGET]", style="bold green") if -24.0 <= out_db <= -10.0 else Text("CLIP", style="bold red") if snap.out_level.clipping else Text("")
    meter_tbl.add_row(
        "OUT",
        level_bar(out_db, snap.out_level.hold_db, bar_w),
        Text(f"{fmt_db(out_db)} dB"),
        out_badge,
        "│ DSP Load:", f"{snap.cpu_load * 100:.1f}%",
    )

    # Line 3: REF (AEC speaker reference, only if AEC is enabled)
    aec_active = ("aec" in tel and tel["aec"].get("enabled", 1.0) > 0.5)
    if aec_active and snap.ref_level is not None:
        ref_db = snap.ref_level.peak_db
        if ref_db <= -80.0:
            ref_text = "   silent"
            ref_badge = Text("(Clean/Headphones)", style="dim green")
        elif ref_db > -55.0:
            ref_text = f"{fmt_db(ref_db)} dB"
            ref_badge = Text("(Speaker Active)", style="dim cyan")
        else:
            ref_text = f"{fmt_db(ref_db)} dB"
            ref_badge = Text("(Speaker)", style="dim cyan")

        aec_erle = tel.get("aec", {}).get("erle_db", 0.0)
        meter_tbl.add_row(
            "REF",
            level_bar(ref_db, snap.ref_level.hold_db, bar_w),
            Text(ref_text, style="grey50" if ref_db <= -80.0 else "white"),
            ref_badge,
            "│ AEC Echo:", f"-{aec_erle:.1f} dB" if aec_erle > 0.5 else "Clean",
        )

    # Mini vocal presence spectrum
    presence_wave = _render_mini_presence_spectrum(snap.out_bands, width=min(32, width - 20))

    # Glitch & Pipeline Quickline
    xr = snap.input_overflows + snap.output_underruns
    glitch_col = "green" if xr == 0 else "yellow" if xr < 5 else "red"
    diag_line = Text()
    diag_line.append("Audio Sync: ", style="grey50")
    diag_line.append("PERFECT (0 drops)" if xr == 0 else f"{xr} GLITCHES", style=glitch_col)
    diag_line.append(" │ Noise Floor: ", style="grey50")
    nf = tel.get("denoise", {}).get("noise_floor_db")
    diag_line.append(f"{nf:.1f} dBFS" if nf else "-", style="cyan")
    diag_line.append(" │ De-Verb: ", style="grey50")
    dv = tel.get("deverb", {}).get("reverb_reduction_db", 0.0)
    diag_line.append(f"-{dv:.1f}dB", style="green")
    diag_line.append(" │ Body: ", style="grey50")
    bod = tel.get("proximity", {}).get("boost_db", 0.0)
    diag_line.append(f"+{bod:.1f}dB", style="yellow")

    parts: list[RenderableType] = [header, Text("")]
    if muted_alert:
        parts.append(muted_alert)
    parts.append(meter_tbl)
    parts.append(Text(""))
    parts.append(presence_wave)
    parts.append(diag_line)

    if ctx.message and now < ctx.message_until:
        parts.append(Text(f"\n⚡ {ctx.message}", style="bold cyan"))

    # Quick Gamer Controls Bar
    keys_bar = Text()
    is_narrow = width < 85
    controls = [
        ("Tab", "Studio"), ("1-6", "Preset"), ("x", "Mute"),
        ("b", "Bypass"), ("s", "GUI"), ("?", "Help"), ("q", "Quit"),
    ]
    for i, (k, label) in enumerate(controls):
        keys_bar.append(f" {k} ", style="bold black on cyan")
        keys_bar.append(f" {label}", style="grey70")
        if i < len(controls) - 1:
            keys_bar.append(" │ ", style="grey37")

    parts.append(Text(""))
    parts.append(keys_bar)

    return Panel(
        Group(*parts),
        title=f"[bold cyan]⚡ micforge GAMER HUD[/] [dim]v{__version__}[/]",
        border_style="cyan",
        box=box.ROUNDED,
    )


# =============================================================================
# STUDIO INSPECTOR MODE: FULL MODULAR BROADCAST RACK
# =============================================================================
def render_dashboard(snap: EngineSnapshot, ctx: LiveContext, width: int, now: float) -> RenderableType:
    """Render full Studio Inspector or Gamer HUD depending on mode."""
    if ctx.hud_mode:
        return render_gamer_hud(snap, ctx, width, now)

    header = Text.assemble(
        (f" {ctx.title} ", "bold white on blue"), "  ",
        _status(snap), "  ",
        ("preset ", "grey70"), (f"[{ctx.preset}]", "bold cyan"),
    )

    parts: list[RenderableType] = [
        header,
        _routing(ctx),
        _pipeline_ribbon(snap),
        Text(""),
        _meters(snap, width),
        Text(""),
        Panel(
            _reductions(snap, width),
            title="processing & dsp reductions",
            title_align="left",
            box=box.ROUNDED,
            border_style="grey37",
            padding=(0, 1),
        ),
    ]

    # Responsive layout: only show dual spectrum if width >= 65 columns
    if width >= 65:
        parts.append(Panel(
            _spectrum(snap, ctx, width),
            title="real-time time-frequency spectrum",
            title_align="left",
            box=box.ROUNDED,
            border_style="grey37",
            padding=(0, 1),
        ))

    parts.append(_stats(snap, ctx, now))

    if snap.calibrating is not None:
        pct = int(snap.calibrating * 100)
        parts.append(Text(f"\n  ◉ Calibrating room noise… stay quiet ({pct}%)", style="bold yellow"))
    elif ctx.message and now < ctx.message_until:
        parts.append(Text(f"\n  ⚡ {ctx.message}", style="bold cyan"))
    else:
        parts.append(Text(""))

    # Controls Cheatsheet Footer
    foot = Text()
    is_narrow = width < 85
    controls = [
        ("Tab", "HUD" if is_narrow else "Gamer HUD"),
        ("1-6", "Preset"),
        ("x", "Mute"),
        ("b", "Bypass"),
        ("s", "GUI"),
        ("?", "Help"),
        ("q", "Quit"),
    ]
    for i, (k, label) in enumerate(controls):
        foot.append(f" {k} ", style="bold black on cyan")
        foot.append(f" {label}", style="grey70")
        if i < len(controls) - 1:
            foot.append(" │ ", style="grey37")
    parts.append(foot)

    # Optional Help Overlay
    if ctx.show_help:
        help_text = Text()
        help_text.append("\n[INTERACTIVE HOTKEY CHEATSHEET]\n", style="bold cyan")
        help_text.append("  [Tab]  Toggle Gamer HUD (compact view) vs Studio Inspector\n", style="white")
        help_text.append("  [1-6]  Instant preset jump (1:discord 2:gaming 3:broadcast 4:studio 5:noisy 6:demuffle)\n", style="white")
        help_text.append("  [b]    Toggle A/B bypass (hear raw mic vs processed voice)\n", style="white")
        help_text.append("  [x]    Mute/unmute microphone\n", style="white")
        help_text.append("  [e]    Toggle Acoustic Echo Cancellation (AEC)\n", style="white")
        help_text.append("  [w]    Cycle Voice Proximity Body/Warmth (Off -> 35% -> 55% -> 75%)\n", style="white")
        help_text.append("  [v]    Cycle Room De-Verb (Off -> 30% -> 50% -> 70%)\n", style="white")
        help_text.append("  [g]    Nudge Noise Gate threshold (+2 dB tighter)\n", style="white")
        help_text.append("  [+/-]  AI Denoiser strength up/down (10% steps)\n", style="white")
        help_text.append("  [[ / ]] Input preamp gain up/down (+/- 1 dB)\n", style="white")
        help_text.append("  [s]    Open Native Multi-Tab Control Center GUI\n", style="white")
        help_text.append("  [c]    Calibrate room ambient noise profile (3s silence)\n", style="white")
        help_text.append("  [q]    Quit cleanly\n", style="white")
        parts.append(Panel(help_text, border_style="cyan", box=box.ROUNDED))

    return Panel(
        Group(*parts),
        title=f"[bold cyan]micforge[/] [dim]v{__version__}[/]",
        border_style="cyan",
        box=box.ROUNDED,
    )
