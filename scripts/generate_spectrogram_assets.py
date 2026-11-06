"""Generate audio assets and spectrogram comparisons for README and docs.

Processes test_audios/1.wav through the complete 11-stage Micforge DSP pipeline in real-time
streaming blocks (160 samples = 10 ms), writes out the normal (raw) and processed WAV files to
docs/audio/, and renders high-resolution spectrogram comparison images to docs/images/.
"""
import os
import shutil
import wave
from pathlib import Path

# Ensure Matplotlib uses a non-interactive backend and writable cache
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np

from micforge.presets import get_preset
from micforge.dsp.chain import build_chain

DOCS_DIR = Path("docs")
AUDIO_DIR = DOCS_DIR / "audio"
IMG_DIR = DOCS_DIR / "images"
ARTIFACTS_DIR = Path(r"C:\Users\ASUS\.gemini\antigravity\brain\f3a000a8-8a6b-4e29-b352-31095eb881c9")

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
        if wf.getsampwidth() == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            data = np.frombuffer(raw, dtype=np.float32)
        if ch > 1:
            data = data.reshape(-1, ch)[:, 0]
        return data, sr


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    clipped = np.clip(data, -1.0, 1.0)
    int_data = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_data.tobytes())


def compute_stft(x: np.ndarray, sr: int, n_fft: int = 512, hop: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = (len(x) - n_fft) // hop
    stft = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop
        segment = x[start : start + n_fft] * win
        stft[:, i] = np.fft.rfft(segment)
    mag = np.abs(stft)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-5))
    freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    times = np.arange(n_frames) * (hop / sr)
    return times, freqs, mag_db


def main():
    print("Generating Micforge real-time audio and spectrogram comparison assets...")
    raw_path = Path("test_audios/1.wav")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing {raw_path}")

    x_raw, sr = read_wav(raw_path)
    print(f"Loaded raw audio: {raw_path} ({len(x_raw)} samples @ {sr} Hz, {len(x_raw)/sr:.1f}s)")

    # 1. Process through full Micforge DSP pipeline block-by-block (real-time streaming blocks)
    block_size = 160  # 10 ms @ 16 kHz
    preset_cfg = get_preset("discord")
    chain = build_chain(preset_cfg, sr)

    # Stream block-by-block to simulate real hardware streaming
    lat = chain.latency_samples
    padded_in = np.concatenate((x_raw, np.zeros(lat + block_size, dtype=np.float32)))
    processed_buf = np.empty_like(padded_in)

    for i in range(0, padded_in.shape[0], block_size):
        chunk = padded_in[i : i + block_size]
        out_chunk = chain.process(chunk)
        processed_buf[i : i + len(out_chunk)] = out_chunk

    # Latency compensated output
    x_proc = processed_buf[lat : lat + len(x_raw)]

    # 2. Save WAV files to docs/audio/
    raw_out_path = AUDIO_DIR / "1_raw.wav"
    proc_out_path = AUDIO_DIR / "1_processed.wav"
    write_wav(raw_out_path, x_raw, sr)
    write_wav(proc_out_path, x_proc, sr)
    print(f"Saved: {raw_out_path}")
    print(f"Saved: {proc_out_path}")

    # 3. Compute STFTs
    times_raw, freqs_raw, spec_raw = compute_stft(x_raw, sr)
    times_proc, freqs_proc, spec_proc = compute_stft(x_proc, sr)

    # Dynamic range normalization
    vmin = -65.0
    vmax = -5.0

    # 4. Generate Spectrogram Comparison Figure (Dual-panel Stacked + Waveforms)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 2.5, 1, 2.5]}, constrained_layout=True)

    time_vec_raw = np.linspace(0, len(x_raw) / sr, len(x_raw))
    time_vec_proc = np.linspace(0, len(x_proc) / sr, len(x_proc))

    # A. Raw Waveform
    axes[0].plot(time_vec_raw, x_raw, color="#38bdf8", lw=0.8, alpha=0.9)
    axes[0].set_title("1. RAW MICROPHONE AUDIO (test_audios/1.wav) — Continuous Ambient Noise & Inconsistent Dynamics", fontsize=11, color="#e2e8f0", fontweight="bold", loc="left")
    axes[0].set_xlim(0, len(x_raw) / sr)
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].set_ylabel("Amplitude", fontsize=9, color="#94a3b8")
    axes[0].tick_params(colors="#94a3b8", labelsize=8)
    axes[0].grid(True, alpha=0.15, ls="--")

    # B. Raw Spectrogram
    im0 = axes[1].pcolormesh(times_raw, freqs_raw, spec_raw, cmap="magma", vmin=vmin, vmax=vmax, shading="gouraud")
    axes[1].set_title("RAW SPECTROGRAM — Heavy Background Fan Hiss, Low-End Rumble & Untreated Room Flutter", fontsize=10, color="#f87171", loc="left")
    axes[1].set_xlim(0, len(x_raw) / sr)
    axes[1].set_ylim(0, sr / 2.0)
    axes[1].set_ylabel("Frequency (Hz)", fontsize=9, color="#94a3b8")
    axes[1].tick_params(colors="#94a3b8", labelsize=8)

    # C. Processed Waveform
    axes[2].plot(time_vec_proc, x_proc, color="#4ade80", lw=0.8, alpha=0.9)
    axes[2].set_title("2. FULL MICFORGE PIPELINE OUTPUT — Pristine Silence in Pauses, Consistent Vocal Energy", fontsize=11, color="#e2e8f0", fontweight="bold", loc="left")
    axes[2].set_xlim(0, len(x_proc) / sr)
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].set_ylabel("Amplitude", fontsize=9, color="#94a3b8")
    axes[2].tick_params(colors="#94a3b8", labelsize=8)
    axes[2].grid(True, alpha=0.15, ls="--")

    # D. Processed Spectrogram
    im1 = axes[3].pcolormesh(times_proc, freqs_proc, spec_proc, cmap="magma", vmin=vmin, vmax=vmax, shading="gouraud")
    axes[3].set_title("PROCESSED SPECTROGRAM — Black Inter-Word Noise Floor, Preserved Vocal Harmonics, De-Essed & Limited", fontsize=10, color="#4ade80", loc="left")
    axes[3].set_xlim(0, len(x_proc) / sr)
    axes[3].set_ylim(0, sr / 2.0)
    axes[3].set_xlabel("Time (seconds)", fontsize=10, color="#94a3b8")
    axes[3].set_ylabel("Frequency (Hz)", fontsize=9, color="#94a3b8")
    axes[3].tick_params(colors="#94a3b8", labelsize=8)

    # Shared colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.015, pad=0.01)
    cbar.set_label("Energy (dBFS)", color="#94a3b8", fontsize=9)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)

    comparison_img_path = IMG_DIR / "spectrogram_comparison.png"
    plt.savefig(comparison_img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated comparison image: {comparison_img_path}")

    # 5. Generate Individual Spectrogram Images (Raw & Processed)
    for title, times, freqs, spec, filename, col in [
        ("Raw Microphone Input (1.wav)", times_raw, freqs_raw, spec_raw, "spectrogram_raw.png", "#f87171"),
        ("Full Micforge Pipeline Output", times_proc, freqs_proc, spec_proc, "spectrogram_processed.png", "#4ade80"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        im = ax.pcolormesh(times, freqs, spec, cmap="magma", vmin=vmin, vmax=vmax, shading="gouraud")
        ax.set_title(f"Micforge Audio Analysis: {title}", fontsize=12, color=col, fontweight="bold", loc="left")
        ax.set_xlim(0, times[-1])
        ax.set_ylim(0, sr / 2.0)
        ax.set_xlabel("Time (seconds)", fontsize=10, color="#94a3b8")
        ax.set_ylabel("Frequency (Hz)", fontsize=10, color="#94a3b8")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.015)
        cbar.set_label("dBFS", color="#94a3b8", fontsize=9)
        cbar.ax.tick_params(colors="#94a3b8", labelsize=8)

        out_path = IMG_DIR / filename
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Generated: {out_path}")

    # Copy images to artifacts directory if available
    if ARTIFACTS_DIR.is_dir():
        for f in [comparison_img_path, IMG_DIR / "spectrogram_raw.png", IMG_DIR / "spectrogram_processed.png"]:
            dest = ARTIFACTS_DIR / f.name
            shutil.copy2(f, dest)
            print(f"Copied to artifacts: {dest}")

    print("All assets successfully created!")


if __name__ == "__main__":
    main()

