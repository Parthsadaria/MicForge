"""`test`: record -> process -> compare -> play back raw vs processed."""
from __future__ import annotations

import os
import time

import numpy as np
from rich.live import Live
from rich.prompt import Prompt
from rich.text import Text

from ..analysis.stats import analyze, band_levels_db, diagnose
from ..audio.devices import find_device, list_devices
from ..audio.fileio import write_wav
from ..audio.recorder import Recorder, play
from ..config import AppConfig, config_dir
from ..dsp.chain import build_chain, process_offline
from ..errors import MicforgeError
from ..presets import get_preset
from ..profile_store import load_profile
from ..ui.report import full_report
from ..ui.theme import console
from ..ui.widgets import fmt_db, level_bar

SILENCE_S = 2.0


def record_clip(app: AppConfig, mic, seconds: float) -> np.ndarray:
    pool = list_devices(all_apis=True)
    dev = find_device(mic if mic is not None else app.input_device, "input", pool)
    rec = Recorder(dev, app.sample_rate, app.input_channel, app.block_size)
    console.print(f"[dim]Mic: {dev.name}[/]")
    rec.start()
    is_cmd = os.name == "nt" and "WT_SESSION" not in os.environ and "TERM_PROGRAM" not in os.environ
    fps = 10 if is_cmd else 20
    try:
        with Live(console=console, refresh_per_second=fps, transient=True) as live:
            while rec.seconds < seconds:
                lv = rec.level
                phase = ("[yellow]Stay SILENT[/] (measuring room noise)" if rec.seconds < SILENCE_S
                         else "[bold green]Now SPEAK[/] naturally, like on a call")
                t = Text.from_markup(f"{phase}   {rec.seconds:4.1f}/{seconds:.0f}s\n")
                t.append_text(level_bar(lv.peak_db, lv.hold_db, max(20, console.width - 20)))
                t.append(f" {fmt_db(lv.peak_db)} dB")
                live.update(t)
                time.sleep(0.04)
    finally:
        x = rec.stop()
    return x


def run_test(app: AppConfig, *, preset: str | None = None, seconds: float = 8.0, mic=None, out=None) -> None:
    preset = preset or app.preset
    pool = list_devices(all_apis=True)
    speaker = find_device(out if out is not None else app.monitor_device, "output", pool)

    raw = record_clip(app, mic, seconds)
    if raw.shape[0] < app.sample_rate:
        raise MicforgeError("Recording failed (no audio captured).")
    sr = app.sample_rate

    # Learn this room's noise from the silent lead-in, so the test reflects what calibration would give.
    cfg = get_preset(preset)
    chain = build_chain(cfg, sr, model_override=app.denoise_model)
    dn = chain.stage("denoise")
    try:
        dn.learn(raw[: int(SILENCE_S * sr)])  # type: ignore[union-attr]
    except ValueError:
        prof = load_profile(sr)
        if prof:
            dn.set_profile(prof[0])  # type: ignore[union-attr]
    proc = process_offline(chain, raw, app.block_size)

    before, after = analyze(raw, sr), analyze(proc, sr)
    console.print(full_report(before, after, band_levels_db(raw, sr), band_levels_db(proc, sr), diagnose(after)))

    clips = {"1": ("raw", raw), "2": (f"processed ({preset})", proc)}
    console.print("\n[dim]Playback via[/] " + speaker.name)
    while True:
        choice = Prompt.ask(
            "\n[bold cyan]1[/] raw  [bold cyan]2[/] processed  [bold cyan]3[/] A/B  [bold cyan]s[/] save wavs  [bold cyan]q[/] done",
            choices=["1", "2", "3", "s", "q"], default="3", console=console,
        )
        if choice == "q":
            break
        if choice == "s":
            d = config_dir() / "recordings"
            stamp = time.strftime("%Y%m%d-%H%M%S")
            write_wav(d / f"{stamp}-raw.wav", raw, sr)
            write_wav(d / f"{stamp}-{preset}.wav", proc, sr)
            console.print(f"[green]Saved to {d}[/]")
            continue
        order = ["1", "2"] if choice == "3" else [choice]
        for k in order:
            label, data = clips[k]
            console.print(f"  ▶ {label}")
            play(data, sr, speaker)
            time.sleep(0.25)
