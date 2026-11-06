"""Modern native multi-tab GUI to configure presets, modular DSP, hotkeys, and devices."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from ..config import AppConfig, ChainConfig
from ..diagnostics import collect_diagnostics, export_diagnostics_zip
from ..errors import ConfigError
from ..presets import (
    BUILTIN,
    delete_preset,
    export_preset,
    get_preset,
    get_preset_description,
    import_preset,
    list_presets,
    save_preset,
)
from ..ui.tuning import MacroKnobs, chain_config_to_macro, macro_to_chain_config
from ..util.autostart import is_autostart_enabled, set_autostart

log = logging.getLogger("micforge.settings_gui")


def launch_settings_gui(
    app: AppConfig,
    on_applied: Callable[[AppConfig], None] | None = None,
    standalone: bool = False,
) -> None:
    """Open the MicForge Control Center & Settings window."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog, ttk
    except ImportError:
        log.error("tkinter is not available in this Python installation.")
        return

    root = tk.Tk() if standalone else tk.Toplevel() if tk._default_root else tk.Tk()
    root.title("Micforge — Control Center & Settings")
    root.geometry("580x680")
    root.minsize(540, 600)

    # Modern dark theme palette
    BG = "#12141a"
    PANEL = "#1a1d26"
    CARD = "#222633"
    TEXT = "#f8fafc"
    MUTED = "#94a3b8"
    ACCENT = "#06b6d4"
    ACCENT_HOVER = "#0891b2"
    BORDER = "#334155"
    GREEN = "#10b981"
    RED = "#ef4444"

    root.configure(bg=BG)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("Card.TFrame", background=CARD)
    style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("Card.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("Header.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 13, "bold"))
    style.configure("SubHeader.TLabel", background=PANEL, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
    style.configure("CardHeader.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10, "bold"))
    style.configure("TCheckbutton", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
    style.map("TCheckbutton", background=[("active", CARD)], foreground=[("active", ACCENT)])

    # Configure Notebook Tabs
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=[14, 8], font=("Segoe UI", 10, "bold"))
    style.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", ACCENT)])

    # Top Header
    header_frame = ttk.Frame(root, padding=(18, 14, 18, 6))
    header_frame.pack(fill=tk.X)
    title_lbl = ttk.Label(header_frame, text="🎙️ MicForge Control Center", style="Header.TLabel")
    title_lbl.pack(side=tk.LEFT)

    # Notebook Tabs Container
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)

    tab_presets = ttk.Frame(notebook)
    tab_dsp = ttk.Frame(notebook)
    tab_hotkeys = ttk.Frame(notebook)
    tab_devices = ttk.Frame(notebook)

    notebook.add(tab_presets, text=" 🎛️ Presets ")
    notebook.add(tab_dsp, text=" ⚙️ Studio DSP ")
    notebook.add(tab_hotkeys, text=" ⚡ Hotkeys & Cues ")
    notebook.add(tab_devices, text=" 🎙️ Routing & System ")

    # State variables
    active_preset_name = tk.StringVar(value=app.preset)
    current_cfg: ChainConfig = get_preset(app.preset)

    # DSP Variables
    var_prox_en = tk.BooleanVar(value=current_cfg.proximity.enabled)
    var_prox_amt = tk.DoubleVar(value=current_cfg.proximity.amount * 100)
    var_prox_warmth = tk.DoubleVar(value=current_cfg.proximity.warmth * 100)
    var_prox_hz = tk.DoubleVar(value=current_cfg.proximity.body_hz)
    var_prox_mud = tk.DoubleVar(value=current_cfg.proximity.mud_cleanup * 100)

    var_verb_en = tk.BooleanVar(value=current_cfg.deverb.enabled)
    var_verb_amt = tk.DoubleVar(value=current_cfg.deverb.amount * 100)
    var_verb_decay = tk.DoubleVar(value=current_cfg.deverb.room_decay)

    var_aec_en = tk.BooleanVar(value=current_cfg.aec.enabled)
    var_aec_amt = tk.DoubleVar(value=current_cfg.aec.amount * 100)

    var_dn_en = tk.BooleanVar(value=current_cfg.denoise.enabled)
    var_dn_str = tk.DoubleVar(value=current_cfg.denoise.strength * 100)

    var_gate_en = tk.BooleanVar(value=current_cfg.gate.enabled)
    var_gate_thresh = tk.DoubleVar(value=current_cfg.gate.threshold_db)
    var_gate_range = tk.DoubleVar(value=current_cfg.gate.range_db)

    var_comp_en = tk.BooleanVar(value=current_cfg.compressor.enabled)
    var_comp_ratio = tk.DoubleVar(value=current_cfg.compressor.ratio)
    var_comp_makeup = tk.DoubleVar(value=current_cfg.compressor.makeup_db)

    # Preamp Quick Gain
    var_input_gain = tk.DoubleVar(value=current_cfg.input_gain_db)
    var_output_gain = tk.DoubleVar(value=0.0)

    # Helper for sliders
    def _add_slider_row(parent, label_text: str, var: tk.DoubleVar, from_: float, to: float, unit: str = "%", res: float = 1.0, on_change: Callable[[float], None] | None = None):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill=tk.X, pady=3)

        lbl = ttk.Label(row, text=label_text, style="Card.TLabel", width=18, anchor="w")
        lbl.pack(side=tk.LEFT)

        val_lbl = ttk.Label(row, text=f"{var.get():.1f}{unit}", style="Card.TLabel", width=8, anchor="e", foreground=ACCENT)

        def _on_val_change(v):
            val_lbl.configure(text=f"{float(v):.1f}{unit}")
            if on_change is not None:
                on_change(float(v))

        scale = tk.Scale(
            row,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=var,
            resolution=res,
            showvalue=False,
            bg=CARD,
            fg=TEXT,
            highlightthickness=0,
            troughcolor=BORDER,
            activebackground=ACCENT,
            command=_on_val_change,
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        val_lbl.pack(side=tk.RIGHT)
        _on_val_change(var.get())

    # =========================================================================
    # TAB 1: PRESETS OVERVIEW & QUICK CONTROLS (UNCLUTTERED)
    # =========================================================================
    p_container = ttk.Frame(tab_presets, padding=14)
    p_container.pack(fill=tk.BOTH, expand=True)

    # Preset Toolbar
    preset_bar = ttk.Frame(p_container, style="Panel.TFrame", padding=12)
    preset_bar.pack(fill=tk.X, pady=(0, 12))

    ttk.Label(preset_bar, text="Active Preset:", style="Panel.TLabel", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 8))

    combo_presets = ttk.Combobox(preset_bar, textvariable=active_preset_name, state="readonly", width=16, font=("Segoe UI", 10))
    combo_presets.pack(side=tk.LEFT, padx=(0, 8))

    lbl_type_badge = ttk.Label(preset_bar, text="[Built-in]", style="Panel.TLabel", foreground=GREEN, font=("Segoe UI", 9, "bold"))
    lbl_type_badge.pack(side=tk.LEFT, padx=(0, 10))

    # Description & Tag Info Box
    desc_card = ttk.Frame(p_container, style="Panel.TFrame", padding=14)
    desc_card.pack(fill=tk.X, pady=(0, 12))

    lbl_desc_title = ttk.Label(desc_card, text="Preset Profile:", style="SubHeader.TLabel")
    lbl_desc_title.pack(anchor="w", pady=(0, 4))

    lbl_desc = ttk.Label(desc_card, text="", style="Panel.TLabel", wraplength=480, foreground=MUTED)
    lbl_desc.pack(anchor="w", pady=(0, 8))

    lbl_tags = ttk.Label(desc_card, text="", style="Panel.TLabel", foreground=ACCENT, font=("Segoe UI", 9, "bold"))
    lbl_tags.pack(anchor="w")

    # Quick Sound Tuning Card
    quick_card = ttk.Frame(p_container, style="Panel.TFrame", padding=14)
    quick_card.pack(fill=tk.X)

    ttk.Label(quick_card, text="QUICK SOUND TUNING", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 8))
    _add_slider_row(quick_card, "Preamp Input Gain:", var_input_gain, -12.0, 18.0, " dB", res=0.5)
    _add_slider_row(quick_card, "AI Denoise Level:", var_dn_str, 0.0, 100.0, "%", res=5.0)

    # =========================================================================
    # TAB 2: STUDIO DSP (SIMPLE 4-KNOB MACROS + ADVANCED COLLAPSIBLE RACK)
    # =========================================================================
    mode_bar = ttk.Frame(tab_dsp, style="Panel.TFrame", padding=(10, 8))
    mode_bar.pack(fill=tk.X, padx=6, pady=(6, 6))

    var_tuning_mode = tk.StringVar(value="simple")

    def _set_mode(mode: str):
        var_tuning_mode.set(mode)
        if mode == "simple":
            btn_simple.configure(bg=ACCENT, fg="#000000")
            btn_adv.configure(bg=BORDER, fg=TEXT)
            frame_simple_mode.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
            container_advanced.pack_forget()
        else:
            btn_simple.configure(bg=BORDER, fg=TEXT)
            btn_adv.configure(bg=ACCENT, fg="#000000")
            frame_simple_mode.pack_forget()
            container_advanced.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    btn_simple = tk.Button(mode_bar, text="💎 Simple Mode (4 Knobs)", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=10, pady=3, cursor="hand2", command=lambda: _set_mode("simple"))
    btn_simple.pack(side=tk.LEFT, padx=3)

    btn_adv = tk.Button(mode_bar, text="⚙️ Advanced Rack", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=10, pady=3, cursor="hand2", command=lambda: _set_mode("advanced"))
    btn_adv.pack(side=tk.LEFT, padx=3)

    def _reset_to_preset():
        name = active_preset_name.get()
        cfg = get_preset(name)
        _load_cfg_to_ui(cfg)

    btn_reset = tk.Button(mode_bar, text="↺ Reset to Preset", bg=BORDER, fg=MUTED, relief=tk.FLAT, font=("Segoe UI", 9), padx=8, pady=3, cursor="hand2", command=_reset_to_preset)
    btn_reset.pack(side=tk.RIGHT, padx=3)

    # 1. Simple Mode Frame
    frame_simple_mode = ttk.Frame(tab_dsp, style="Panel.TFrame", padding=14)

    init_m = chain_config_to_macro(current_cfg)
    var_macro_clarity = tk.DoubleVar(value=init_m.clarity)
    var_macro_warmth = tk.DoubleVar(value=init_m.warmth)
    var_macro_noise = tk.DoubleVar(value=init_m.noise_removal)
    var_macro_room = tk.DoubleVar(value=init_m.room_cleanup)
    _updating_from_macro = False

    def _on_macro_changed(*_):
        nonlocal _updating_from_macro
        if _updating_from_macro:
            return
        _updating_from_macro = True
        try:
            m = MacroKnobs(
                clarity=var_macro_clarity.get(),
                warmth=var_macro_warmth.get(),
                noise_removal=var_macro_noise.get(),
                room_cleanup=var_macro_room.get(),
            )
            mapped_cfg = macro_to_chain_config(m, base_preset_name=active_preset_name.get())
            _load_cfg_to_ui(mapped_cfg, update_macros=False)
        finally:
            _updating_from_macro = False

    ttk.Label(frame_simple_mode, text="STREAMER MACRO SOUND SHAPING", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 4))
    ttk.Label(frame_simple_mode, text="4 intuitive macro controls with broadcast-safe parameter mapping. Instant professional tone without touching complex DSP curves.", style="Panel.TLabel", wraplength=480, foreground=MUTED).pack(anchor="w", pady=(0, 10))

    card_sm = ttk.Frame(frame_simple_mode, style="Card.TFrame", padding=12)
    card_sm.pack(fill=tk.X)

    _add_slider_row(card_sm, "💎 Clarity:", var_macro_clarity, 0.0, 100.0, "%", on_change=_on_macro_changed)
    _add_slider_row(card_sm, "🎙️ Warmth:", var_macro_warmth, 0.0, 100.0, "%", on_change=_on_macro_changed)
    _add_slider_row(card_sm, "🛡️ Noise Removal:", var_macro_noise, 0.0, 100.0, "%", on_change=_on_macro_changed)
    _add_slider_row(card_sm, "🏛️ Room Cleanup:", var_macro_room, 0.0, 100.0, "%", on_change=_on_macro_changed)

    # 2. Advanced Mode Container
    container_advanced = ttk.Frame(tab_dsp)

    canvas_dsp = tk.Canvas(container_advanced, bg=BG, highlightthickness=0)
    scrollbar_dsp = ttk.Scrollbar(container_advanced, orient="vertical", command=canvas_dsp.yview)
    scroll_dsp = ttk.Frame(canvas_dsp)

    scroll_dsp.bind("<Configure>", lambda e: canvas_dsp.configure(scrollregion=canvas_dsp.bbox("all")))
    canvas_dsp_win = canvas_dsp.create_window((0, 0), window=scroll_dsp, anchor="nw")

    def _on_canvas_dsp_cfg(e):
        canvas_dsp.itemconfig(canvas_dsp_win, width=e.width)
    canvas_dsp.bind("<Configure>", _on_canvas_dsp_cfg)

    def _on_mousewheel(e):
        canvas_dsp.yview_scroll(int(-1 * (e.delta / 120)), "units")
    canvas_dsp.bind_all("<MouseWheel>", _on_mousewheel)

    canvas_dsp.configure(yscrollcommand=scrollbar_dsp.set)
    canvas_dsp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
    scrollbar_dsp.pack(side=tk.RIGHT, fill=tk.Y)

    # DSP Card 1: Voice Proximity & Body
    card_prox = ttk.Frame(scroll_dsp, style="Card.TFrame", padding=(12, 10))
    card_prox.pack(fill=tk.X, padx=6, pady=4)
    frame_prox_sliders = ttk.Frame(card_prox, style="Card.TFrame")

    def _toggle_prox():
        if var_prox_en.get():
            frame_prox_sliders.pack(fill=tk.X, pady=(6, 0))
        else:
            frame_prox_sliders.pack_forget()

    chk_prox = ttk.Checkbutton(card_prox, text="🎙️ Voice Proximity & Chest Resonance", variable=var_prox_en, style="TCheckbutton", command=_toggle_prox)
    chk_prox.pack(anchor="w")
    _add_slider_row(frame_prox_sliders, "Intensity:", var_prox_amt, 0.0, 100.0, "%")
    _add_slider_row(frame_prox_sliders, "Warmth (Saturation):", var_prox_warmth, 0.0, 100.0, "%")
    _add_slider_row(frame_prox_sliders, "Chest Pitch (F0):", var_prox_hz, 80.0, 180.0, " Hz", res=1.0)
    _add_slider_row(frame_prox_sliders, "Anti-Mud Cleanup:", var_prox_mud, 0.0, 100.0, "%")

    # DSP Card 2: Room De-Verb
    card_verb = ttk.Frame(scroll_dsp, style="Card.TFrame", padding=(12, 10))
    card_verb.pack(fill=tk.X, padx=6, pady=4)
    frame_verb_sliders = ttk.Frame(card_verb, style="Card.TFrame")

    def _toggle_verb():
        if var_verb_en.get():
            frame_verb_sliders.pack(fill=tk.X, pady=(6, 0))
        else:
            frame_verb_sliders.pack_forget()

    chk_verb = ttk.Checkbutton(card_verb, text="🏛️ Room Dereverberation (De-Verb)", variable=var_verb_en, style="TCheckbutton", command=_toggle_verb)
    chk_verb.pack(anchor="w")
    _add_slider_row(frame_verb_sliders, "De-Verb Intensity:", var_verb_amt, 0.0, 100.0, "%")
    _add_slider_row(frame_verb_sliders, "Room Decay (RT60):", var_verb_decay, 0.15, 1.20, " s", res=0.05)

    # DSP Card 3: Neural AI Denoiser
    card_dn = ttk.Frame(scroll_dsp, style="Card.TFrame", padding=(12, 10))
    card_dn.pack(fill=tk.X, padx=6, pady=4)
    frame_dn_sliders = ttk.Frame(card_dn, style="Card.TFrame")

    def _toggle_dn():
        if var_dn_en.get():
            frame_dn_sliders.pack(fill=tk.X, pady=(6, 0))
        else:
            frame_dn_sliders.pack_forget()

    chk_dn = ttk.Checkbutton(card_dn, text="🛡️ CoFi-Lite Neural AI Denoiser", variable=var_dn_en, style="TCheckbutton", command=_toggle_dn)
    chk_dn.pack(anchor="w")
    _add_slider_row(frame_dn_sliders, "Suppression Strength:", var_dn_str, 0.0, 100.0, "%")

    # DSP Card 4: Acoustic Echo Cancellation
    card_aec = ttk.Frame(scroll_dsp, style="Card.TFrame", padding=(12, 10))
    card_aec.pack(fill=tk.X, padx=6, pady=4)
    frame_aec_sliders = ttk.Frame(card_aec, style="Card.TFrame")

    def _toggle_aec():
        if var_aec_en.get():
            frame_aec_sliders.pack(fill=tk.X, pady=(6, 0))
        else:
            frame_aec_sliders.pack_forget()

    chk_aec = ttk.Checkbutton(card_aec, text="🔊 Acoustic Echo Cancellation (AEC)", variable=var_aec_en, style="TCheckbutton", command=_toggle_aec)
    chk_aec.pack(anchor="w")
    _add_slider_row(frame_aec_sliders, "Cancellation Power:", var_aec_amt, 0.0, 100.0, "%")

    # DSP Card 5: Noise Gate
    card_gate = ttk.Frame(scroll_dsp, style="Card.TFrame", padding=(12, 10))
    card_gate.pack(fill=tk.X, padx=6, pady=4)
    frame_gate_sliders = ttk.Frame(card_gate, style="Card.TFrame")

    def _toggle_gate():
        if var_gate_en.get():
            frame_gate_sliders.pack(fill=tk.X, pady=(6, 0))
        else:
            frame_gate_sliders.pack_forget()

    chk_gate = ttk.Checkbutton(card_gate, text="🚪 Studio Noise Gate / Expander", variable=var_gate_en, style="TCheckbutton", command=_toggle_gate)
    chk_gate.pack(anchor="w")
    _add_slider_row(frame_gate_sliders, "Gate Threshold:", var_gate_thresh, -60.0, -20.0, " dBFS", res=1.0)
    _add_slider_row(frame_gate_sliders, "Range (Depth):", var_gate_range, 10.0, 40.0, " dB", res=1.0)

    # DSP Card 6: Dynamics & Limiter
    card_comp = ttk.Frame(scroll_dsp, style="Card.TFrame", padding=(12, 10))
    card_comp.pack(fill=tk.X, padx=6, pady=4)
    frame_comp_sliders = ttk.Frame(card_comp, style="Card.TFrame")

    def _toggle_comp():
        if var_comp_en.get():
            frame_comp_sliders.pack(fill=tk.X, pady=(6, 0))
        else:
            frame_comp_sliders.pack_forget()

    chk_comp = ttk.Checkbutton(card_comp, text="🎚️ Fast Compressor & Master Limiter", variable=var_comp_en, style="TCheckbutton", command=_toggle_comp)
    chk_comp.pack(anchor="w")
    _add_slider_row(frame_comp_sliders, "Compression Ratio:", var_comp_ratio, 1.5, 6.0, ":1", res=0.5)
    _add_slider_row(frame_comp_sliders, "Makeup Gain:", var_comp_makeup, 0.0, 12.0, " dB", res=0.5)

    def _update_preset_info_and_collapsibles():
        name = active_preset_name.get()
        is_builtin = name in BUILTIN
        lbl_type_badge.configure(
            text="[Built-in]" if is_builtin else "[Custom]",
            foreground=GREEN if is_builtin else ACCENT,
        )
        btn_del.configure(state=tk.DISABLED if is_builtin else tk.NORMAL)

        desc = get_preset_description(name) or f"Custom profile '{name}'."
        lbl_desc.configure(text=desc)

        tags = []
        if var_dn_en.get():
            tags.append(f"AI Denoise {var_dn_str.get():.0f}%")
        if var_prox_en.get():
            tags.append(f"Proximity {var_prox_amt.get():.0f}%")
        if var_verb_en.get():
            tags.append(f"De-Verb {var_verb_amt.get():.0f}%")
        if var_aec_en.get():
            tags.append("AEC")
        if var_gate_en.get():
            tags.append(f"Gate {var_gate_thresh.get():.0f}dB")
        if var_comp_en.get():
            tags.append(f"Comp {var_comp_ratio.get():.1f}:1")
        lbl_tags.configure(text=" • ".join(tags) if tags else "Direct Passthrough (Bypass)")

        # Sync collapsible sections
        _toggle_prox()
        _toggle_verb()
        _toggle_dn()
        _toggle_aec()
        _toggle_gate()
        _toggle_comp()

    def _refresh_preset_dropdown():
        presets = list_presets()
        names = [p.name for p in presets]
        combo_presets["values"] = names
        if active_preset_name.get() not in names and names:
            active_preset_name.set(names[0])
        _update_preset_info_and_collapsibles()

    def _load_cfg_to_ui(cfg: ChainConfig, update_macros: bool = True):
        nonlocal current_cfg
        current_cfg = cfg
        var_prox_en.set(cfg.proximity.enabled)
        var_prox_amt.set(cfg.proximity.amount * 100)
        var_prox_warmth.set(cfg.proximity.warmth * 100)
        var_prox_hz.set(cfg.proximity.body_hz)
        var_prox_mud.set(cfg.proximity.mud_cleanup * 100)

        var_verb_en.set(cfg.deverb.enabled)
        var_verb_amt.set(cfg.deverb.amount * 100)
        var_verb_decay.set(cfg.deverb.room_decay)

        var_aec_en.set(cfg.aec.enabled)
        var_aec_amt.set(cfg.aec.amount * 100)

        var_dn_en.set(cfg.denoise.enabled)
        var_dn_str.set(cfg.denoise.strength * 100)

        var_gate_en.set(cfg.gate.enabled)
        var_gate_thresh.set(cfg.gate.threshold_db)
        var_gate_range.set(cfg.gate.range_db)

        var_comp_en.set(cfg.compressor.enabled)
        var_comp_ratio.set(cfg.compressor.ratio)
        var_comp_makeup.set(cfg.compressor.makeup_db)

        var_input_gain.set(cfg.input_gain_db)

        if update_macros and not _updating_from_macro:
            m = chain_config_to_macro(cfg)
            var_macro_clarity.set(m.clarity)
            var_macro_warmth.set(m.warmth)
            var_macro_noise.set(m.noise_removal)
            var_macro_room.set(m.room_cleanup)

        _update_preset_info_and_collapsibles()

    def _sync_ui_to_cfg() -> ChainConfig:
        cfg = current_cfg
        cfg.input_gain_db = var_input_gain.get()
        cfg.proximity.enabled = var_prox_en.get()
        cfg.proximity.amount = var_prox_amt.get() / 100.0
        cfg.proximity.warmth = var_prox_warmth.get() / 100.0
        cfg.proximity.body_hz = var_prox_hz.get()
        cfg.proximity.mud_cleanup = var_prox_mud.get() / 100.0

        cfg.deverb.enabled = var_verb_en.get()
        cfg.deverb.amount = var_verb_amt.get() / 100.0
        cfg.deverb.room_decay = var_verb_decay.get()

        cfg.aec.enabled = var_aec_en.get()
        cfg.aec.amount = var_aec_amt.get() / 100.0

        cfg.denoise.enabled = var_dn_en.get()
        cfg.denoise.strength = var_dn_str.get() / 100.0

        cfg.gate.enabled = var_gate_en.get()
        cfg.gate.threshold_db = var_gate_thresh.get()
        cfg.gate.range_db = var_gate_range.get()

        cfg.compressor.enabled = var_comp_en.get()
        cfg.compressor.ratio = var_comp_ratio.get()
        cfg.compressor.makeup_db = var_comp_makeup.get()
        return cfg

    def _on_select_preset(event=None):
        name = active_preset_name.get()
        try:
            cfg = get_preset(name)
            app.preset = name
            _load_cfg_to_ui(cfg)
        except Exception as exc:
            messagebox.showerror("Preset Error", str(exc), parent=root)

    combo_presets.bind("<<ComboboxSelected>>", _on_select_preset)

    # Preset Action Functions
    def _action_new_preset():
        name = simpledialog.askstring("New Custom Preset", "Enter name for new preset:", parent=root)
        if not name:
            return
        clean = name.strip()
        try:
            cfg = _sync_ui_to_cfg()
            save_preset(clean, cfg, overwrite=False, description=f"Custom preset based on {active_preset_name.get()}")
            active_preset_name.set(clean)
            app.preset = clean
            _refresh_preset_dropdown()
            messagebox.showinfo("Preset Created", f"Successfully created custom preset '{clean}'!", parent=root)
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=root)

    def _action_save_preset():
        name = active_preset_name.get()
        cfg = _sync_ui_to_cfg()
        if name in BUILTIN:
            new_name = simpledialog.askstring(
                "Save as Custom Preset",
                f"Built-in preset '{name}' cannot be modified directly.\nEnter name for custom preset:",
                initialvalue=f"{name}_custom",
                parent=root,
            )
            if not new_name:
                return
            try:
                save_preset(new_name, cfg, overwrite=True)
                active_preset_name.set(new_name)
                app.preset = new_name
                _refresh_preset_dropdown()
                messagebox.showinfo("Saved", f"Saved custom preset '{new_name}'!", parent=root)
            except Exception as exc:
                messagebox.showerror("Error", str(exc), parent=root)
        else:
            try:
                save_preset(name, cfg, overwrite=True)
                _refresh_preset_dropdown()
                messagebox.showinfo("Saved", f"Preset '{name}' updated successfully!", parent=root)
            except Exception as exc:
                messagebox.showerror("Error", str(exc), parent=root)

    def _action_delete_preset():
        name = active_preset_name.get()
        if name in BUILTIN:
            return
        if messagebox.askyesno("Delete Preset", f"Are you sure you want to delete preset '{name}'?", parent=root):
            try:
                delete_preset(name)
                active_preset_name.set("discord")
                app.preset = "discord"
                _on_select_preset()
            except Exception as exc:
                messagebox.showerror("Error", str(exc), parent=root)

    def _action_export():
        name = active_preset_name.get()
        path = filedialog.asksaveasfilename(
            parent=root,
            initialfile=f"{name}_preset.json",
            defaultextension=".json",
            filetypes=[("Preset JSON", "*.json")],
        )
        if path:
            try:
                export_preset(name, path)
                messagebox.showinfo("Exported", f"Exported preset '{name}' to {Path(path).name}!", parent=root)
            except Exception as exc:
                messagebox.showerror("Export Failed", str(exc), parent=root)

    def _action_import():
        path = filedialog.askopenfilename(
            parent=root,
            filetypes=[("Preset JSON", "*.json")],
        )
        if path:
            try:
                imported_name = import_preset(path, overwrite=True)
                _refresh_preset_dropdown()
                active_preset_name.set(imported_name)
                _on_select_preset()
                messagebox.showinfo("Imported", f"Successfully imported preset '{imported_name}'!", parent=root)
            except Exception as exc:
                messagebox.showerror("Import Failed", str(exc), parent=root)

    # Preset Toolbar Action Buttons
    btn_new = tk.Button(preset_bar, text="➕ New", bg=BORDER, fg=TEXT, relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=8, pady=2, cursor="hand2", command=_action_new_preset)
    btn_new.pack(side=tk.LEFT, padx=3)

    btn_save_p = tk.Button(preset_bar, text="💾 Save", bg=BORDER, fg=TEXT, relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=8, pady=2, cursor="hand2", command=_action_save_preset)
    btn_save_p.pack(side=tk.LEFT, padx=3)

    btn_del = tk.Button(preset_bar, text="🗑 Del", bg=BORDER, fg=TEXT, relief=tk.FLAT, font=("Segoe UI", 9), padx=6, pady=2, cursor="hand2", command=_action_delete_preset)
    btn_del.pack(side=tk.LEFT, padx=3)

    btn_exp = tk.Button(preset_bar, text="📤", bg=BORDER, fg=TEXT, relief=tk.FLAT, padx=6, pady=2, cursor="hand2", command=_action_export)
    btn_exp.pack(side=tk.RIGHT, padx=3)

    btn_imp = tk.Button(preset_bar, text="📥", bg=BORDER, fg=TEXT, relief=tk.FLAT, padx=6, pady=2, cursor="hand2", command=_action_import)
    btn_imp.pack(side=tk.RIGHT, padx=3)

    # Initialize preset info and simple mode
    _refresh_preset_dropdown()
    _set_mode("simple")

    # =========================================================================
    # TAB 3: GLOBAL HOTKEYS & AUDIO CUES
    # =========================================================================
    hotkeys_container = ttk.Frame(tab_hotkeys, padding=16)
    hotkeys_container.pack(fill=tk.BOTH, expand=True)

    hotkeys_panel = ttk.Frame(hotkeys_container, style="Panel.TFrame", padding=14)
    hotkeys_panel.pack(fill=tk.X, pady=(0, 14))

    ttk.Label(hotkeys_panel, text="HOTKEYS (SYSTEM-WIDE)", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))

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
        if event.state & 0x0004:
            keys.append("ctrl")
        if event.state & 0x0001:
            keys.append("shift")
        if event.state & 0x20000 or (hasattr(event, "alt") and event.alt):
            keys.append("alt")

        sym = event.keysym.lower()
        if sym in ("control_l", "control_r", "shift_l", "shift_r", "alt_l", "alt_r", "win_l", "win_r"):
            return
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
            command=lambda v=var, b=None: None,
        )
        btn.configure(command=lambda v=var, b=btn: _start_recording(v, b))
        btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        btn_clear = tk.Button(
            row,
            text="✕",
            bg=BORDER,
            fg=MUTED,
            activebackground=RED,
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

    # Audio Cues Panel
    cues_panel = ttk.Frame(hotkeys_container, style="Panel.TFrame", padding=14)
    cues_panel.pack(fill=tk.X, pady=(0, 14))

    ttk.Label(cues_panel, text="HEADPHONE AUDIO CHIMES", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))

    var_cues = tk.BooleanVar(value=app.hotkeys.play_cues)
    chk_cues = ttk.Checkbutton(cues_panel, text="Play gentle audio cues on mute / unmute", variable=var_cues, style="TCheckbutton")
    chk_cues.pack(anchor="w", pady=(0, 8))

    vol_row = ttk.Frame(cues_panel, style="Panel.TFrame")
    vol_row.pack(fill=tk.X, pady=4)
    ttk.Label(vol_row, text="Chime Volume:", width=14, style="Panel.TLabel").pack(side=tk.LEFT)

    var_vol = tk.DoubleVar(value=app.hotkeys.cue_volume * 100)
    vol_slider = tk.Scale(vol_row, from_=0, to=100, orient=tk.HORIZONTAL, variable=var_vol, bg=PANEL, fg=TEXT, highlightthickness=0, troughcolor=BORDER, activebackground=ACCENT)
    vol_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

    def _test_chime():
        from ..audio.cues import CuePlayer
        cp = CuePlayer(sample_rate=app.sample_rate, volume=var_vol.get() / 100.0, enabled=True)
        cp.play_standalone("unmute")

    btn_test = tk.Button(vol_row, text="♫ Test", bg=BORDER, fg=TEXT, activebackground=ACCENT, activeforeground="#000000", relief=tk.FLAT, padx=10, cursor="hand2", command=_test_chime)
    btn_test.pack(side=tk.RIGHT)

    # =========================================================================
    # TAB 4: AUDIO DEVICES & SYSTEM
    # =========================================================================
    dev_container = ttk.Frame(tab_devices, padding=16)
    dev_container.pack(fill=tk.BOTH, expand=True)

    dev_panel = ttk.Frame(dev_container, style="Panel.TFrame", padding=14)
    dev_panel.pack(fill=tk.X, pady=(0, 14))

    ttk.Label(dev_panel, text="AUDIO ROUTING HARDWARE", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))

    input_devs = []
    output_devs = []
    try:
        from ..audio.devices import list_devices
        all_devs = list_devices()
        input_devs = [f"{d.index}: {d.name}" for d in all_devs if d.max_input > 0]
        output_devs = [f"{d.index}: {d.name}" for d in all_devs if d.max_output > 0]
    except Exception as exc:
        log.debug("Device query in GUI: %s", exc)

    var_dev_mic = tk.StringVar(value=str(app.input_device) if app.input_device is not None else "(Default Microphone)")
    var_dev_cable = tk.StringVar(value=str(app.output_device) if app.output_device is not None else "(Default Output)")
    var_dev_mon = tk.StringVar(value=str(app.monitor_device) if app.monitor_device is not None else "(Disabled)")

    def _add_device_row(parent, label_text: str, var: tk.StringVar, options: list[str]):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=6)
        ttk.Label(row, text=label_text, width=16, anchor="w", style="Panel.TLabel").pack(side=tk.LEFT)
        combo = ttk.Combobox(row, textvariable=var, values=["(System Default)"] + options, state="readonly", font=("Segoe UI", 9))
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

    _add_device_row(dev_panel, "Input Mic:", var_dev_mic, input_devs)
    _add_device_row(dev_panel, "Virtual Cable:", var_dev_cable, output_devs)
    _add_device_row(dev_panel, "Headphones:", var_dev_mon, output_devs)

    # Startup & Background Panel
    sys_panel = ttk.Frame(dev_container, style="Panel.TFrame", padding=14)
    sys_panel.pack(fill=tk.X)
    ttk.Label(sys_panel, text="SYSTEM & BACKGROUND", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))

    var_tray = tk.BooleanVar(value=app.hotkeys.start_in_tray)
    chk_tray = ttk.Checkbutton(sys_panel, text="Start minimized in Windows System Tray", variable=var_tray, style="TCheckbutton")
    chk_tray.pack(anchor="w", pady=(0, 6))

    var_boot = tk.BooleanVar(value=is_autostart_enabled())
    chk_boot = ttk.Checkbutton(sys_panel, text="Start Micforge automatically on Windows Boot", variable=var_boot, style="TCheckbutton")
    chk_boot.pack(anchor="w")

    def _export_diag():
        try:
            from ..diagnostics import collect_diagnostics, export_diagnostics_zip
            rep = collect_diagnostics(app)
            out_p = export_diagnostics_zip(rep)
            messagebox.showinfo(
                "Diagnostics Exported",
                f"Diagnostic report exported successfully!\n\nLocation:\n{out_p}\n\nAll passwords, API tokens, and user home paths have been securely redacted.",
                parent=root,
            )
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc), parent=root)

    btn_diag = tk.Button(sys_panel, text="📦 Export Diagnostics (.zip)", bg=BORDER, fg=TEXT, activebackground=ACCENT, activeforeground="#000000", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=10, pady=4, cursor="hand2", command=_export_diag)
    btn_diag.pack(anchor="w", pady=(10, 0))

    # =========================================================================
    # BOTTOM ACTION BAR: SAVE & APPLY
    # =========================================================================
    btn_row = ttk.Frame(root, padding=(18, 10, 18, 14))
    btn_row.pack(fill=tk.X, side=tk.BOTTOM)

    def _save_and_apply():
        app.preset = active_preset_name.get()
        app.hotkeys.toggle_mute = var_mute.get().strip()
        app.hotkeys.push_to_talk = var_ptt.get().strip()
        app.hotkeys.push_to_mute = var_ptm.get().strip()
        app.hotkeys.toggle_bypass = var_bypass.get().strip()
        app.hotkeys.play_cues = var_cues.get()
        app.hotkeys.cue_volume = var_vol.get() / 100.0
        app.hotkeys.start_in_tray = var_tray.get()

        def _parse_dev(val: str):
            v = val.strip()
            if not v or v.startswith("(") or "Default" in v or "Disabled" in v:
                return None
            if ":" in v:
                idx = v.split(":", 1)[0].strip()
                if idx.isdigit():
                    return int(idx)
            return v

        app.input_device = _parse_dev(var_dev_mic.get())
        app.output_device = _parse_dev(var_dev_cable.get())
        app.monitor_device = _parse_dev(var_dev_mon.get())

        set_autostart(var_boot.get())

        # If custom preset, save preset values
        if active_preset_name.get() not in BUILTIN:
            try:
                save_preset(active_preset_name.get(), _sync_ui_to_cfg(), overwrite=True)
            except Exception:
                pass

        try:
            app.save()
        except Exception as exc:
            log.warning("Could not persist config: %s", exc)

        if on_applied is not None:
            on_applied(app)

        root.destroy()

    btn_save = tk.Button(
        btn_row,
        text="✔ Save & Apply",
        bg=ACCENT,
        fg="#000000",
        activebackground=ACCENT_HOVER,
        relief=tk.FLAT,
        font=("Segoe UI", 10, "bold"),
        padx=16,
        pady=7,
        cursor="hand2",
        command=_save_and_apply,
    )
    btn_save.pack(side=tk.RIGHT, padx=(10, 0))

    btn_cancel = tk.Button(
        btn_row,
        text="Close",
        bg=BORDER,
        fg=TEXT,
        relief=tk.FLAT,
        font=("Segoe UI", 10),
        padx=14,
        pady=7,
        cursor="hand2",
        command=root.destroy,
    )
    btn_cancel.pack(side=tk.RIGHT)

    if standalone:
        root.mainloop()


def open_settings_window(app: AppConfig, on_applied: Callable[[AppConfig], None] | None = None) -> None:
    """Launch settings GUI in a clean dedicated thread to avoid blocking audio/tray threads."""
    threading.Thread(target=launch_settings_gui, args=(app, on_applied, True), daemon=True).start()
