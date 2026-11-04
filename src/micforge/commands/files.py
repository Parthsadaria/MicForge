"""`process` and `analyze`: offline work on WAV files."""
from __future__ import annotations

from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from ..analysis.stats import analyze, band_levels_db, diagnose
from ..audio.fileio import read_wav, write_wav
from ..config import AppConfig
from ..dsp.chain import process_offline
from ..errors import MicforgeError
from ..ui.report import full_report
from ..ui.theme import console
from .common import make_chain


def run_process(app: AppConfig, src: str, dst: str, preset: str | None) -> None:
    preset = preset or app.preset
    if not Path(src).is_file():
        raise MicforgeError(f"File not found: {src}")
    x, sr = read_wav(src, target_rate=app.sample_rate)
    chain, _ = make_chain(app, preset)
    dn = chain.stage("denoise")
    if dn is not None and getattr(dn, "profile", None) is None:
        try:  # no saved profile: use the quietest 1 s of the file as the noise reference
            import numpy as np
            fr = int(0.1 * sr)
            n = x.shape[0] // fr
            lv = np.mean(x[: n * fr].reshape(n, fr) ** 2, axis=1)
            i = int(np.argmin(np.convolve(lv, np.ones(10) / 10, mode="valid")))
            dn.learn(x[i * fr : i * fr + 10 * fr])  # type: ignore[attr-defined]
        except (ValueError, IndexError):
            pass
    with Progress(SpinnerColumn(), TextColumn("processing…"), console=console, transient=True):
        y = process_offline(chain, x, app.block_size)
    write_wav(dst, y, sr)
    console.print(f"[green]✔ Wrote {dst}[/]  (preset: {preset})")
    b, a = analyze(x, sr), analyze(y, sr)
    console.print(full_report(b, a, band_levels_db(x, sr), band_levels_db(y, sr), diagnose(a)))


def run_analyze(app: AppConfig, src: str) -> None:
    if not Path(src).is_file():
        raise MicforgeError(f"File not found: {src}")
    x, sr = read_wav(src)
    s = analyze(x, sr)
    console.print(full_report(s, None, findings=diagnose(s)))
