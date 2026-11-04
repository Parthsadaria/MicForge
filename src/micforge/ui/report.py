"""Before/after report used by `test`, `process` and `analyze`."""
from __future__ import annotations

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..analysis.stats import AudioStats, Finding, clarity_label
from .theme import LEVEL_ICON
from .widgets import freq_axis, spectrum_rows
import numpy as np


def _f(v: float | None, fmt: str = "{:.1f}", unit: str = "") -> str:
    return "-" if v is None else fmt.format(v) + unit


def _delta(a: float | None, b: float | None, better_up: bool) -> Text:
    if a is None or b is None:
        return Text("")
    d = b - a
    if abs(d) < 0.5:
        return Text(f"{d:+.1f}", style="grey50")
    good = (d > 0) == better_up
    return Text(f"{d:+.1f}", style="green" if good else "yellow")


def stats_table(before: AudioStats, after: AudioStats | None = None) -> Table:
    t = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", pad_edge=False)
    t.add_column("metric")
    t.add_column("raw", justify="right")
    if after:
        t.add_column("processed", justify="right")
        t.add_column("Δ", justify="right")
    rows = [
        ("peak (dBFS)", before.peak_db, after.peak_db if after else None, None),
        ("voice level (dBFS)", before.active_rms_db, after.active_rms_db if after else None, None),
        ("noise floor (dBFS)", before.noise_floor_db, after.noise_floor_db if after else None, False),
        ("SNR (dB)", before.snr_db, after.snr_db if after else None, True),
        ("presence 2.5-8k vs rest (dB)", before.presence_db, after.presence_db if after else None, True),
        ("spectral centroid (Hz)", before.centroid_hz, after.centroid_hz if after else None, True),
    ]
    for name, a, b, up in rows:
        cells = [name, _f(a, "{:.0f}" if "Hz" in name else "{:.1f}")]
        if after:
            cells.append(_f(b, "{:.0f}" if "Hz" in name else "{:.1f}"))
            cells.append(_delta(a, b, up) if up is not None else Text(""))  # type: ignore[arg-type]
        t.add_row(*cells)
    clar = [clarity_label(before.presence_db)] + ([clarity_label(after.presence_db)] if after else [])
    t.add_row("clarity (heuristic)", *[Text(c, style="bold") for c in clar], *([Text("")] if after else []))
    clips = [str(before.clip_samples)] + ([str(after.clip_samples)] if after else [])
    t.add_row("clipped samples", *clips, *([Text("")] if after else []))
    return t


def findings_panel(findings: list[Finding]) -> Panel:
    lines = Text()
    for f in findings:
        lines.append_text(Text.from_markup(f"{LEVEL_ICON[f.level]} "))
        lines.append(f.text + "\n")
    lines.rstrip()
    return Panel(lines, title="diagnosis", title_align="left", border_style="grey37", box=box.ROUNDED)


def spectrum_compare(before: np.ndarray, after: np.ndarray, f_lo: float = 60.0, f_hi: float = 16000.0) -> Panel:
    ref = float(np.max(before))
    lo, hi = ref - 60.0, ref  # same scale for both so the shapes are comparable
    n = before.shape[0]
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column()
    tbl.add_column()
    tbl.add_row(Text("raw", style="grey70"), Text("processed", style="bold green"))
    for a, b in zip(spectrum_rows(before, 5, 1, lo, hi, "blue"), spectrum_rows(after, 5, 1, lo, hi, "green")):
        tbl.add_row(a, b)
    ax = freq_axis(n, 1, f_lo, f_hi)
    tbl.add_row(ax, ax)
    return Panel(tbl, title="long-term spectrum", title_align="left", border_style="grey37", box=box.ROUNDED)


def full_report(before: AudioStats, after: AudioStats | None, b_bands=None, a_bands=None, findings=None) -> Group:
    parts = [stats_table(before, after)]
    if b_bands is not None and a_bands is not None:
        parts.append(spectrum_compare(b_bands, a_bands))
    if findings:
        parts.append(findings_panel(findings))
    return Group(*parts)
