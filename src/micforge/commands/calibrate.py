"""`calibrate`: learn the room noise and set the gate threshold."""
from __future__ import annotations

import time

from rich.progress import BarColumn, Progress, TextColumn

from ..audio.devices import find_device, list_devices
from ..audio.recorder import Recorder
from ..config import AppConfig
from ..dsp.denoise import SpectralDenoiser
from ..dsp.util import rms_db
from ..errors import MicforgeError
from ..profile_store import clear_profile, save_profile
from ..ui.theme import console


def run_calibrate(app: AppConfig, *, seconds: float = 4.0, mic=None, reset: bool = False) -> None:
    if reset:
        console.print("[green]Noise profile removed.[/]" if clear_profile(app.sample_rate) else "[dim]No profile saved.[/]")
        return
    dev = find_device(mic if mic is not None else app.input_device, "input", list_devices(all_apis=True))
    console.print(f"[bold]Calibrating[/] on [cyan]{dev.name}[/]. Stay quiet and leave fans/PC running as usual.")
    rec = Recorder(dev, app.sample_rate, app.input_channel, app.block_size)
    rec.start()
    try:
        with Progress(TextColumn("[yellow]listening…"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"), console=console) as p:
            task = p.add_task("", total=seconds)
            while rec.seconds < seconds:
                p.update(task, completed=min(rec.seconds, seconds))
                time.sleep(0.05)
            p.update(task, completed=seconds)
    finally:
        noise = rec.stop()
    if noise.shape[0] < app.sample_rate:
        raise MicforgeError("Not enough audio captured.")
    if rec.level.clipping:
        console.print("[yellow]The mic clipped while calibrating - is something very loud nearby?[/]")

    dn = SpectralDenoiser(app.sample_rate)
    psd = dn.learn(noise)
    noise_db = rms_db(noise)
    gate = noise_db + 9.0
    save_profile(psd, app.sample_rate, noise_db, gate)
    console.print(f"[green]✔ Room noise {noise_db:.1f} dBFS.[/] Gate threshold {gate:.1f} dBFS. Profile saved.")
    if noise_db > -45:
        console.print("[yellow]That is loud for a room. Try moving the mic closer to your mouth or away from fans.[/]")
