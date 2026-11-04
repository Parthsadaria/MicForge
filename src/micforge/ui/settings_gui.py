"""Simple, modern native GUI to configure global hotkeys, audio chimes, and startup."""
from __future__ import annotations

import logging
import threading
from typing import Callable

from ..config import AppConfig
from ..util.autostart import is_autostart_enabled, set_autostart

log = logging.getLogger("micforge.settings_gui")


def launch_settings_gui(
    app: AppConfig,
    on_applied: Callable[[AppConfig], None] | None = None,
    standalone: bool = False,
) -> None:
    """Open the Settings & Hotkeys configuration window."""
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError:
        log.error("tkinter is not available in this Python installation.")
        return

    root = tk.Tk() if standalone else tk.Toplevel() if tk._default_root else tk.Tk()
    root.title("Micforge — Settings & Hotkeys")
    root.geometry("480x560")
    root.minsize(440, 520)

    # Dark modern theme palette
    BG = "#18181b"
    PANEL = "#27272a"
    TEXT = "#f4f4f5"
    MUTED = "#a1a1aa"
    ACCENT = "#06b6d4"
    ACCENT_HOVER = "#0891b2"
    BORDER = "#3f3f46"

    root.configure(bg=BG)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("Header.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 12, "bold"))
    style.configure("SubHeader.TLabel", background=PANEL, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
    style.configure("TCheckbutton", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
    style.map("TCheckbutton", background=[("active", PANEL)], foreground=[("active", ACCENT)])

    # Main container with scrolling/padding
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)

    header = ttk.Label(frame, text="⚡ Global Hotkeys & Audio Cues", style="Header.TLabel")
    header.pack(anchor="w", pady=(0, 15))

    # --- Section 1: Global Hotkeys ---
    hotkeys_panel = ttk.Frame(frame, style="Panel.TFrame", padding=15)
    hotkeys_panel.pack(fill=tk.X, pady=(0, 15))

    ttk.Label(hotkeys_panel, text="HOTKEYS (SYSTEM-WIDE)", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))

    # Hotkey string variables
    var_mute = tk.StringVar(value=app.hotkeys.toggle_mute)
    var_ptt = tk.StringVar(value=app.hotkeys.push_to_talk)
    var_ptm = tk.StringVar(value=app.hotkeys.push_to_mute)
    var_bypass = tk.StringVar(value=app.hotkeys.toggle_bypass)

    recording_target: tk.StringVar | None = None
    recording_btn: tk.Button | None = None

    def _start_recording(var: tk.StringVar, btn: tk.Button):
        nonlocal recording_target, recording_btn
        if recording_target is not None and recording_btn is not None:
            recording_btn.configure(text=recording_target.get() or "None", bg=BORDER, fg=TEXT)
        recording_target = var
        recording_btn = btn
        btn.configure(text="[Press keys...]", bg=ACCENT, fg="#000000")
        root.focus_set()

    def _on_key_press(event):
        nonlocal recording_target, recording_btn
        if recording_target is None or recording_btn is None:
            return

        keys = []
        if event.state & 0x0004:  # Ctrl
            keys.append("ctrl")
        if event.state & 0x0001:  # Shift
            keys.append("shift")
        if event.state & 0x20000 or (hasattr(event, "alt") and event.alt):  # Alt
            keys.append("alt")

        sym = event.keysym.lower()
        if sym in ("control_l", "control_r", "shift_l", "shift_r", "alt_l", "alt_r", "win_l", "win_r"):
            return  # Modifier pressed alone, wait for key
        if sym == "escape":
            recording_target.set("")
            recording_btn.configure(text="None", bg=BORDER, fg=TEXT)
            recording_target, recording_btn = None, None
            return

        keys.append(sym)
        combo = "+".join(keys)
        recording_target.set(combo)
        recording_btn.configure(text=combo, bg=BORDER, fg=TEXT)
        recording_target, recording_btn = None, None

    root.bind("<Key>", _on_key_press)

    def _add_hotkey_row(parent, label_text: str, var: tk.StringVar):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=4)
        lbl = ttk.Label(row, text=label_text, width=16, anchor="w", style="Panel.TLabel")
        lbl.pack(side=tk.LEFT)

        btn = tk.Button(
            row,
            text=var.get() or "None",
            bg=BORDER,
            fg=TEXT,
            activebackground=ACCENT_HOVER,
            activeforeground="#000000",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=3,
            cursor="hand2",
        )
        btn.configure(command=lambda v=var, b=btn: _start_recording(v, b))
        btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        btn_clear = tk.Button(
            row,
            text="✕",
            bg=BORDER,
            fg=MUTED,
            activebackground="#ef4444",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2",
            command=lambda v=var, b=btn: (v.set(""), b.configure(text="None")),
        )
        btn_clear.pack(side=tk.RIGHT)

    _add_hotkey_row(hotkeys_panel, "Toggle Mute:", var_mute)
    _add_hotkey_row(hotkeys_panel, "Push-to-Talk:", var_ptt)
    _add_hotkey_row(hotkeys_panel, "Push-to-Mute:", var_ptm)
    _add_hotkey_row(hotkeys_panel, "Toggle Bypass:", var_bypass)

    # --- Section 2: Audio Cues ---
    cues_panel = ttk.Frame(frame, style="Panel.TFrame", padding=15)
    cues_panel.pack(fill=tk.X, pady=(0, 15))

    ttk.Label(cues_panel, text="HEADPHONE AUDIO CHIMES", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))

    var_cues = tk.BooleanVar(value=app.hotkeys.play_cues)
    chk_cues = ttk.Checkbutton(
        cues_panel,
        text="Play gentle audio cues on mute / unmute",
        variable=var_cues,
        style="TCheckbutton",
    )
    chk_cues.pack(anchor="w", pady=(0, 8))

    vol_row = ttk.Frame(cues_panel, style="Panel.TFrame")
    vol_row.pack(fill=tk.X, pady=4)
    ttk.Label(vol_row, text="Chime Volume:", width=14, style="Panel.TLabel").pack(side=tk.LEFT)

    var_vol = tk.DoubleVar(value=app.hotkeys.cue_volume * 100)
    vol_slider = tk.Scale(
        vol_row,
        from_=0,
        to=100,
        orient=tk.HORIZONTAL,
        variable=var_vol,
        bg=PANEL,
        fg=TEXT,
        highlightthickness=0,
        troughcolor=BORDER,
        activebackground=ACCENT,
    )
    vol_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

    def _test_chime():
        from ..audio.cues import CuePlayer

        cp = CuePlayer(sample_rate=app.sample_rate, volume=var_vol.get() / 100.0, enabled=True)
        cp.play_standalone("unmute")

    btn_test = tk.Button(
        vol_row,
        text="♫ Test",
        bg=BORDER,
        fg=TEXT,
        activebackground=ACCENT,
        activeforeground="#000000",
        relief=tk.FLAT,
        padx=10,
        cursor="hand2",
        command=_test_chime,
    )
    btn_test.pack(side=tk.RIGHT)

    # --- Section 3: Startup & Tray ---
    tray_panel = ttk.Frame(frame, style="Panel.TFrame", padding=15)
    tray_panel.pack(fill=tk.X, pady=(0, 15))

    ttk.Label(tray_panel, text="SYSTEM & BACKGROUND", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))

    var_tray = tk.BooleanVar(value=app.hotkeys.start_in_tray)
    chk_tray = ttk.Checkbutton(
        tray_panel,
        text="Start minimized in System Tray",
        variable=var_tray,
        style="TCheckbutton",
    )
    chk_tray.pack(anchor="w", pady=(0, 6))

    var_boot = tk.BooleanVar(value=is_autostart_enabled())
    chk_boot = ttk.Checkbutton(
        tray_panel,
        text="Start Micforge on Windows Boot",
        variable=var_boot,
        style="TCheckbutton",
    )
    chk_boot.pack(anchor="w")

    # --- Action Buttons ---
    def _save():
        app.hotkeys.toggle_mute = var_mute.get().strip()
        app.hotkeys.push_to_talk = var_ptt.get().strip()
        app.hotkeys.push_to_mute = var_ptm.get().strip()
        app.hotkeys.toggle_bypass = var_bypass.get().strip()
        app.hotkeys.play_cues = var_cues.get()
        app.hotkeys.cue_volume = var_vol.get() / 100.0
        app.hotkeys.start_in_tray = var_tray.get()

        # Update registry autostart
        set_autostart(var_boot.get())

        # Persist to config.json
        try:
            app.save()
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not persist config: %s", exc)

        if on_applied is not None:
            on_applied(app)

        if standalone:
            root.destroy()
        else:
            root.destroy()

    btn_row = ttk.Frame(frame)
    btn_row.pack(fill=tk.X, pady=(10, 0))

    btn_save = tk.Button(
        btn_row,
        text="✔ Save & Apply",
        bg=ACCENT,
        fg="#000000",
        activebackground=ACCENT_HOVER,
        relief=tk.FLAT,
        font=("Segoe UI", 10, "bold"),
        pady=6,
        cursor="hand2",
        command=_save,
    )
    btn_save.pack(side=tk.RIGHT, padx=(10, 0))

    btn_cancel = tk.Button(
        btn_row,
        text="Cancel",
        bg=BORDER,
        fg=TEXT,
        relief=tk.FLAT,
        font=("Segoe UI", 10),
        pady=6,
        cursor="hand2",
        command=root.destroy,
    )
    btn_cancel.pack(side=tk.RIGHT)

    if standalone:
        root.mainloop()


def open_settings_window(app: AppConfig, on_applied: Callable[[AppConfig], None] | None = None) -> None:
    """Launch settings GUI in a clean dedicated thread to avoid blocking audio/tray threads."""
    threading.Thread(target=launch_settings_gui, args=(app, on_applied, True), daemon=True).start()

