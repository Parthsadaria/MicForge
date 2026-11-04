"""Pure rendering helpers: meters, spectrum columns, bars. No I/O, trivially testable."""
from __future__ import annotations

import math

import numpy as np
from rich.text import Text

BLOCKS = " ▁▂▃▄▅▆▇█"


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def level_style(db: float) -> str:
    if db >= -6.0:
        return "bold red"
    if db >= -18.0:
        return "yellow"
    return "green"


def level_bar(peak_db: float, hold_db: float, width: int, lo: float = -60.0, hi: float = 0.0) -> Text:
    width = max(8, width)
    filled = int(round(_clamp((peak_db - lo) / (hi - lo)) * width))
    hold_pos = int(round(_clamp((hold_db - lo) / (hi - lo)) * width)) - 1
    t = Text()

    curr_char = ""
    curr_style = ""
    curr_count = 0

    def flush():
        nonlocal curr_char, curr_style, curr_count
        if curr_count > 0:
            t.append(curr_char * curr_count, style=curr_style)
            curr_count = 0

    for i in range(width):
        cell_db = lo + (i + 0.5) / width * (hi - lo)
        if i == hold_pos and hold_db > lo:
            ch = "┃"
            st = level_style(hold_db)
        elif i < filled:
            ch = "█"
            st = level_style(cell_db)
        else:
            ch = "·"
            st = "grey23"

        if ch == curr_char and st == curr_style:
            curr_count += 1
        else:
            flush()
            curr_char = ch
            curr_style = st
            curr_count = 1
    flush()
    return t


def reduction_bar(db: float, width: int, max_db: float = 24.0, style: str = "cyan") -> Text:
    filled = int(round(_clamp(db / max_db) * width))
    t = Text()
    t.append("█" * filled, style=style)
    t.append("·" * (width - filled), style="grey23")
    return t


def load_bar(frac: float, width: int) -> Text:
    frac = _clamp(frac)
    style = "green" if frac < 0.5 else "yellow" if frac < 0.8 else "bold red"
    filled = int(round(frac * width))
    t = Text()
    t.append("█" * filled, style=style)
    t.append("·" * (width - filled), style="grey23")
    return t


def spectrum_rows(levels_db: np.ndarray, height: int, rep: int = 1, lo: float = -95.0, hi: float = -35.0,
                  style: str = "green") -> list[Text]:
    """Multi-row bar spectrum using eighth-block characters. Row 0 is the top."""
    norm = np.clip((np.asarray(levels_db, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    eighths = norm * height * 8.0
    rows = []
    for r in range(height - 1, -1, -1):
        chars = []
        for e in eighths:
            cell = int(round(_clamp(e - r * 8.0, 0.0, 8.0)))
            chars.append(BLOCKS[cell] * rep)
        rows.append(Text("".join(chars), style=style))
    return rows


def freq_axis(n_bands: int, rep: int, f_lo: float, f_hi: float) -> Text:
    width = n_bands * rep
    chars = [" "] * width
    for f, label in ((100, "100"), (1000, "1k"), (10000, "10k")):
        if f_lo < f < f_hi:
            pos = int(n_bands * math.log(f / f_lo) / math.log(f_hi / f_lo) * rep)
            for j, c in enumerate(label):
                if 0 <= pos + j < width:
                    chars[pos + j] = c
    return Text("".join(chars), style="grey50")


def fmt_db(db: float) -> str:
    return " -inf" if db <= -119.0 else f"{db:6.1f}"
