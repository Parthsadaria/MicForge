"""Shared console + palette."""
from __future__ import annotations

import sys

if sys.platform == "win32":
    # Enable UTF-8 encoding and native Virtual Terminal (VT100) processing for Windows CMD / conhost
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass

from rich.console import Console
from rich.theme import Theme

THEME = Theme(
    {
        "accent": "bold cyan",
        "dim": "grey50",
        "ok": "green",
        "warn": "yellow",
        "bad": "bold red",
        "key": "bold black on cyan",
    }
)

console = Console(theme=THEME, highlight=False, legacy_windows=False if sys.platform == "win32" else None)

LEVEL_ICON = {"ok": "[ok]✔[/]", "warn": "[warn]![/]", "bad": "[bad]✖[/]"}
