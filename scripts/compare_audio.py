"""Audio enhancement comparison test tool using the full MicForge architecture.

Processes noisy audio through MicForge's complete 8-stage DSP pipeline:
  1. 80 Hz 4th-order High-Pass Filter (eliminates sub-bass rumble)
  2. CoFi-Lite Neural Speech Denoiser (AI voice cleaning)
  3. Calibrated Noise Gate / Downward Expander (silences speech pauses & hiss)
  4. 3-Band Parametric EQ (de-mud, presence, air)
  5. Split-Band De-Esser at 6.5 kHz (tames sibilance)
  6. Optical RMS Auto-Leveler (evens speech dynamics)
  7. Fast Feed-Forward Compressor (tames loud peaks)
  8. True Peak Lookahead Limiter (prevents clipping)

Generates:
  - 1_noisy.wav (original audio)
  - 1_processed.wav (fully enhanced audio)
  - 1_noisy_spec.png & 1_processed_spec.png (individual spectrograms)
  - comparison_side_by_side.png (side-by-side card comparison matching reference UI)
  - index.html (interactive comparison player)

Usage:
    python scripts/compare_audio.py [path_to_wav] [--preset PRESET] [--strength STRENGTH]
    e.g. python scripts/compare_audio.py test_audios/1.wav
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure repo src/ is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Reconfigure stdout for UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure matplotlib temporary cache directory before importing
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import soundfile as sf

# Import MicForge DSP architecture
from micforge.commands.common import make_chain
from micforge.config import AppConfig
from micforge.dsp.chain import process_offline


def compute_spectrogram(audio: np.ndarray, sample_rate: int, n_fft: int = 512, hop: int = 128):
    """Compute log-magnitude power spectrogram."""
    f, t, sxx = sig.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        mode="magnitude",
    )
    # Convert to dB
    sxx_db = 20.0 * np.log10(np.maximum(sxx, 1e-6))
    vmax = np.max(sxx_db)
    vmin = max(vmax - 80.0, -90.0)
    return f, t, sxx_db, vmin, vmax


def plot_single_spectrogram(f, t, sxx_db, vmin, vmax, out_path: Path):
    """Render a pure spectrogram image with minimal dark framing."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200, facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    ax.pcolormesh(t, f, sxx_db, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    ax.set_ylim([0, f[-1]])
    ax.axis("off")  # Clean borderless spectrogram look
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_comparison_card_image(
    noisy_f, noisy_t, noisy_spec,
    enh_f, enh_t, enh_spec,
    vmin, vmax,
    duration_s: float,
    out_path: Path,
    noisy_title: str = "Noisy speech sample 1",
    enh_title: str = "Enhanced speech by MicForge (CoFi-Lite + DSP)",
):
    """Render side-by-side comparison image matching the reference design."""
    fig = plt.figure(figsize=(13.0, 6.2), dpi=200, facecolor="#f3f4f6")

    # Left Card
    ax_noisy_hdr = fig.add_axes([0.04, 0.86, 0.44, 0.08])
    ax_noisy_hdr.set_facecolor("#3da35d")  # Reference green
    ax_noisy_hdr.text(
        0.5, 0.5, noisy_title,
        color="white", fontsize=13, fontweight="bold",
        ha="center", va="center"
    )
    ax_noisy_hdr.set_xticks([])
    ax_noisy_hdr.set_yticks([])
    for spine in ax_noisy_hdr.spines.values():
        spine.set_color("#3da35d")

    ax_noisy_spec = fig.add_axes([0.04, 0.28, 0.44, 0.56])
    ax_noisy_spec.set_facecolor("#0b0c10")
    ax_noisy_spec.pcolormesh(noisy_t, noisy_f, noisy_spec, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    ax_noisy_spec.set_ylim([0, noisy_f[-1]])
    ax_noisy_spec.set_xticks([])
    ax_noisy_spec.set_yticks([])
    for spine in ax_noisy_spec.spines.values():
        spine.set_color("#2a2b30")
        spine.set_linewidth(1.5)

    # Simulated audio player bar underneath noisy
    ax_noisy_player = fig.add_axes([0.04, 0.12, 0.44, 0.12])
    ax_noisy_player.set_facecolor("#e5e7eb")
    ax_noisy_player.plot(0.08, 0.5, marker=">", markersize=10, color="#374151")
    ax_noisy_player.text(0.14, 0.5, f"0:00 / {int(duration_s // 60):02d}:{int(duration_s % 60):02d}", color="#4b5563", fontsize=10, va="center", family="sans-serif")
    ax_noisy_player.plot([0.38, 0.88], [0.5, 0.5], color="#9ca3af", lw=4, solid_capstyle="round")
    ax_noisy_player.plot([0.38, 0.40], [0.5, 0.5], color="#2563eb", lw=4, solid_capstyle="round")
    ax_noisy_player.plot(0.92, 0.5, marker="s", markersize=6, color="#4b5563")
    ax_noisy_player.set_xlim([0, 1])
    ax_noisy_player.set_ylim([0, 1])
    ax_noisy_player.axis("off")

    # Right Card
    ax_enh_hdr = fig.add_axes([0.52, 0.86, 0.44, 0.08])
    ax_enh_hdr.set_facecolor("#3da35d")  # Reference green
    ax_enh_hdr.text(
        0.5, 0.5, enh_title,
        color="white", fontsize=13, fontweight="bold",
        ha="center", va="center"
    )
    ax_enh_hdr.set_xticks([])
    ax_enh_hdr.set_yticks([])
    for spine in ax_enh_hdr.spines.values():
        spine.set_color("#3da35d")

    ax_enh_spec = fig.add_axes([0.52, 0.28, 0.44, 0.56])
    ax_enh_spec.set_facecolor("#0b0c10")
    ax_enh_spec.pcolormesh(enh_t, enh_f, enh_spec, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    ax_enh_spec.set_ylim([0, enh_f[-1]])
    ax_enh_spec.set_xticks([])
    ax_enh_spec.set_yticks([])
    for spine in ax_enh_spec.spines.values():
        spine.set_color("#2a2b30")
        spine.set_linewidth(1.5)

    # Simulated audio player bar underneath enhanced
    ax_enh_player = fig.add_axes([0.52, 0.12, 0.44, 0.12])
    ax_enh_player.set_facecolor("#e5e7eb")
    ax_enh_player.plot(0.08, 0.5, marker=">", markersize=10, color="#374151")
    ax_enh_player.text(0.14, 0.5, f"0:00 / {int(duration_s // 60):02d}:{int(duration_s % 60):02d}", color="#4b5563", fontsize=10, va="center", family="sans-serif")
    ax_enh_player.plot([0.38, 0.88], [0.5, 0.5], color="#9ca3af", lw=4, solid_capstyle="round")
    ax_enh_player.plot([0.38, 0.40], [0.5, 0.5], color="#2563eb", lw=4, solid_capstyle="round")
    ax_enh_player.plot(0.92, 0.5, marker="s", markersize=6, color="#4b5563")
    ax_enh_player.set_xlim([0, 1])
    ax_enh_player.set_ylim([0, 1])
    ax_enh_player.axis("off")

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def generate_interactive_html(
    noisy_wav_rel: str,
    enh_wav_rel: str,
    noisy_spec_rel: str,
    enh_spec_rel: str,
    duration_s: float,
    html_path: Path,
):
    """Generate self-contained interactive comparison HTML player."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MicForge Audio Enhancement Comparison</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      margin: 0;
      padding: 30px 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .header {{
      text-align: center;
      margin-bottom: 25px;
    }}
    .header h1 {{
      margin: 0 0 8px 0;
      font-size: 26px;
      color: #38bdf8;
    }}
    .header p {{
      margin: 0;
      color: #94a3b8;
      font-size: 14px;
    }}
    .comparison-container {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 24px;
      max-width: 1000px;
      width: 100%;
    }}
    .card {{
      background: #1e293b;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
      border: 1px solid #334155;
      display: flex;
      flex-direction: column;
    }}
    .card-header {{
      background: #3da35d;
      color: #ffffff;
      padding: 12px 18px;
      font-weight: 600;
      font-size: 15px;
      text-align: center;
      letter-spacing: 0.3px;
    }}
    .card-header.processed {{
      background: #2563eb;
    }}
    .spectrogram-container {{
      padding: 12px;
      background: #090d16;
      display: flex;
      justify-content: center;
    }}
    .spectrogram-img {{
      width: 100%;
      height: auto;
      border-radius: 6px;
      display: block;
      border: 1px solid #1e293b;
    }}
    .player-container {{
      padding: 16px;
      background: #1e293b;
      border-top: 1px solid #334155;
    }}
    audio {{
      width: 100%;
      height: 40px;
      border-radius: 6px;
      outline: none;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 600;
      margin-top: 6px;
    }}
    .badge-noisy {{ background: #7f1d1d; color: #fecaca; }}
    .badge-clean {{ background: #064e3b; color: #a7f3d0; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>🎙️ MicForge Speech Enhancement Inspection</h1>
    <p>Complete 8-Stage DSP Architecture with CoFi-Lite Neural Enhancement</p>
  </div>

  <div class="comparison-container">
    <!-- Noisy Speech Card -->
    <div class="card">
      <div class="card-header">
        Noisy speech sample 1
        <div><span class="badge badge-noisy">Raw Audio</span></div>
      </div>
      <div class="spectrogram-container">
        <img class="spectrogram-img" src="{noisy_spec_rel}" alt="Noisy Spectrogram">
      </div>
      <div class="player-container">
        <audio controls src="{noisy_wav_rel}"></audio>
      </div>
    </div>

    <!-- Enhanced Speech Card -->
    <div class="card">
      <div class="card-header processed">
        Enhanced speech by MicForge (CoFi-Lite + DSP)
        <div><span class="badge badge-clean">MicForge Architecture</span></div>
      </div>
      <div class="spectrogram-container">
        <img class="spectrogram-img" src="{enh_spec_rel}" alt="Enhanced Spectrogram">
      </div>
      <div class="player-container">
        <audio controls src="{enh_wav_rel}"></audio>
      </div>
    </div>
  </div>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")


def run_comparison(
    input_file: str | Path,
    output_dir: str | Path,
    preset: str = "discord",
    strength: float = 1.0,
):
    """Run full audio processing using the full MicForge architecture and generate all assets."""
    in_path = Path(input_file).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {in_path}")

    stem = in_path.stem

    # If output_dir was left at default "1&processed" but input is different (e.g. 2.wav), use stem
    out_dir_str = str(output_dir)
    if "1&processed" in out_dir_str and stem != "1":
        out_dir_str = out_dir_str.replace("1&processed", f"{stem}&processed")

    out_dirs = [Path(out_dir_str).resolve()]
    if "test_audio" in out_dir_str and "test_audios" not in out_dir_str:
        out_dirs.append(Path(out_dir_str.replace("test_audio", "test_audios")).resolve())
    elif "test_audios" in out_dir_str:
        out_dirs.append(Path(out_dir_str.replace("test_audios", "test_audio")).resolve())

    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Reading input audio: {in_path.name}")
    audio, sr = sf.read(str(in_path))
    if audio.ndim > 1:
        audio = audio[:, 0]  # Take mono channel
    audio = audio.astype(np.float32)

    duration_s = len(audio) / sr
    print(f"      Duration: {duration_s:.2f}s | Sample Rate: {sr} Hz | Frames: {len(audio)}")

    # 2. Build full MicForge DSP Chain Architecture
    print(f"[2/4] Initializing full MicForge DSP Chain ({preset} preset, strength={strength})...")
    block_size = int(sr * 0.01)  # 10 ms blocks (160 @ 16k, 480 @ 48k)
    app = AppConfig(sample_rate=sr, block_size=block_size, preset=preset)
    chain, _ = make_chain(app, preset)

    # Set denoiser strength
    dn = chain.stage("denoise")
    if dn is not None and hasattr(dn, "set_strength"):
        dn.set_strength(strength)

    # Automatically calibrate noise gate threshold to the audio's quietest ambient frames
    frame_size = int(sr * 0.1)
    frames_rms = [np.sqrt(np.mean(audio[i : i + frame_size] ** 2)) for i in range(0, len(audio), frame_size)]
    min_rms = np.min(frames_rms) if frames_rms else 1e-4
    noise_floor_db = 20.0 * np.log10(max(float(min_rms), 1e-6))

    gate = chain.stage("gate")
    if gate is not None:
        # Set gate threshold 6 dB above measured noise floor so ambient hiss & pressure cooker sound are completely silenced during pauses
        gate.threshold_db = min(noise_floor_db + 6.0, -32.0)
        gate.range_db = 28.0
        print(f"      Calibrated Noise Floor: {noise_floor_db:.1f} dBFS -> Noise Gate Threshold: {gate.threshold_db:.1f} dBFS (Range: {gate.range_db} dB)")

    print("      Processing block-by-block through full 8-stage DSP chain...")
    enhanced = process_offline(chain, audio, block=block_size)

    # Compute energy levels
    in_rms = float(np.sqrt(np.mean(audio**2)))
    out_rms = float(np.sqrt(np.mean(enhanced**2)))
    snr_gain = 20.0 * np.log10(max(in_rms, 1e-6) / max(out_rms, 1e-6))
    print(f"      Full-Chain Dynamics: Input RMS {in_rms:.4f} -> Output RMS {out_rms:.4f} (Effective Attenuation: {snr_gain:+.1f} dB)")

    # 3. Compute Spectrograms
    print("[3/4] Computing time-frequency spectrograms...")
    noisy_f, noisy_t, noisy_spec, vmin1, vmax1 = compute_spectrogram(audio, sr)
    enh_f, enh_t, enh_spec, vmin2, vmax2 = compute_spectrogram(enhanced, sr)

    common_vmin = min(vmin1, vmin2)
    common_vmax = max(vmax1, vmax2)

    # 4. Save Audio and Images to Output Directory
    print("[4/4] Writing audio files, spectrogram images, and interactive player...")

    primary = out_dirs[0]
    noisy_wav = primary / f"{stem}_noisy.wav"
    enh_wav = primary / f"{stem}_processed.wav"
    noisy_spec_png = primary / f"{stem}_noisy_spec.png"
    enh_spec_png = primary / f"{stem}_processed_spec.png"
    side_by_side_png = primary / "comparison_side_by_side.png"
    html_file = primary / "index.html"

    # Write WAV files
    sf.write(str(noisy_wav), audio, sr)
    sf.write(str(enh_wav), enhanced, sr)

    # Render Spectrogram Images
    plot_single_spectrogram(noisy_f, noisy_t, noisy_spec, common_vmin, common_vmax, noisy_spec_png)
    plot_single_spectrogram(enh_f, enh_t, enh_spec, common_vmin, common_vmax, enh_spec_png)

    # Render Side-by-Side Comparison Card Image (matching user reference design)
    render_comparison_card_image(
        noisy_f, noisy_t, noisy_spec,
        enh_f, enh_t, enh_spec,
        common_vmin, common_vmax,
        duration_s,
        side_by_side_png,
        noisy_title=f"Noisy speech sample {stem}",
        enh_title="Enhanced speech by MicForge (CoFi-Lite + DSP)",
    )

    # Generate interactive HTML comparison
    generate_interactive_html(
        noisy_wav_rel=noisy_wav.name,
        enh_wav_rel=enh_wav.name,
        noisy_spec_rel=noisy_spec_png.name,
        enh_spec_rel=enh_spec_png.name,
        duration_s=duration_s,
        html_path=html_file,
    )

    # Mirror to alternate directory
    for od in out_dirs[1:]:
        for f in [noisy_wav, enh_wav, noisy_spec_png, enh_spec_png, side_by_side_png, html_file]:
            shutil.copy2(f, od / f.name)

    print("\n[SUCCESS] Successfully generated all assets:")
    for od in out_dirs:
        print(f"   Directory: {od}")
        print(f"      - {noisy_wav.name} (Original raw recording)")
        print(f"      - {enh_wav.name} (MicForge enhanced speech)")
        print(f"      - {noisy_spec_png.name} (Noisy spectrogram)")
        print(f"      - {enh_spec_png.name} (Enhanced spectrogram)")
        print(f"      - {side_by_side_png.name} (Side-by-side comparison card image)")
        print(f"      - {html_file.name} (Interactive web player)")

    return {
        "audio_noisy": noisy_wav,
        "audio_enhanced": enh_wav,
        "spec_side_by_side": side_by_side_png,
        "html_player": html_file,
    }


def main():
    parser = argparse.ArgumentParser(description="MicForge audio comparison test tool")
    parser.add_argument("input", nargs="?", default="test_audios/1.wav", help="Path to input .wav file")
    parser.add_argument("--output", "-o", default="test_audio/outputs/1&processed", help="Output directory")
    parser.add_argument("--preset", "-p", default="discord", help="MicForge preset (discord, noisy_room, etc.)")
    parser.add_argument("--strength", "-s", type=float, default=1.0, help="Neural denoiser strength (0.0 to 1.0)")
    args = parser.parse_args()

    run_comparison(args.input, args.output, preset=args.preset, strength=args.strength)


if __name__ == "__main__":
    main()
