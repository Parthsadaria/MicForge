"""Audio comparison test tool for MicForge v1.2.0.

Supports:
1. Comparing an audio file against its enhanced MicForge output (defaults to sample 2: test_audios/2.wav).
2. Comparing two existing audio files side-by-side (e.g. file1.wav vs file2.wav, or raw vs competitor/GTCRN).
3. Comparing two different MicForge presets/models on the same audio (e.g. --preset1 discord --preset2 gaming).

Generates:
  - Spectrograms for each audio stream
  - comparison_side_by_side.png (side-by-side card comparison matching reference UI)
  - index.html (interactive comparison player with synchronized playback & A/B toggle)
  - Audio files placed in output directory

Usage:
  # 1. Process and compare sample 2 (default):
  python scripts/compare_two.py

  # 2. Process and compare a specific file:
  python scripts/compare_two.py test_audios/2.wav

  # 3. Compare two existing audio files side-by-side:
  python scripts/compare_two.py file1.wav file2.wav --label1 "Sample 1" --label2 "Sample 2"

  # 4. Compare two presets on sample 2:
  python scripts/compare_two.py test_audios/2.wav --preset1 discord --preset2 gaming
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
    sxx_db = 20.0 * np.log10(np.maximum(sxx, 1e-6))
    vmax = float(np.max(sxx_db))
    vmin = max(vmax - 80.0, -90.0)
    return f, t, sxx_db, vmin, vmax


def compute_audio_metrics(audio: np.ndarray, sample_rate: int) -> dict:
    """Compute audio metrics: RMS, peak, crest factor, and estimated noise floor."""
    rms = float(np.sqrt(np.mean(audio**2)))
    rms_db = 20.0 * np.log10(max(rms, 1e-6))
    peak = float(np.max(np.abs(audio)))
    peak_db = 20.0 * np.log10(max(peak, 1e-6))
    crest_factor_db = peak_db - rms_db

    # Estimated noise floor using 100ms quietest frame
    frame_size = int(sample_rate * 0.1)
    if len(audio) >= frame_size:
        frames_rms = [
            float(np.sqrt(np.mean(audio[i : i + frame_size] ** 2)))
            for i in range(0, len(audio) - frame_size + 1, frame_size // 2)
        ]
        min_rms = min(frames_rms) if frames_rms else 1e-4
    else:
        min_rms = rms
    noise_floor_db = 20.0 * np.log10(max(float(min_rms), 1e-6))

    return {
        "rms": rms,
        "rms_db": rms_db,
        "peak": peak,
        "peak_db": peak_db,
        "crest_factor_db": crest_factor_db,
        "noise_floor_db": noise_floor_db,
    }


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
    left_f, left_t, left_spec,
    right_f, right_t, right_spec,
    vmin, vmax,
    duration_s: float,
    out_path: Path,
    left_title: str = "Noisy speech sample 2",
    right_title: str = "Enhanced speech by MicForge (CoFi-Lite + DSP)",
    left_header_color: str = "#3da35d",
    right_header_color: str = "#3da35d",
):
    """Render side-by-side comparison image matching the reference design."""
    fig = plt.figure(figsize=(13.0, 6.2), dpi=200, facecolor="#f3f4f6")

    # Left Card
    ax_left_hdr = fig.add_axes([0.04, 0.86, 0.44, 0.08])
    ax_left_hdr.set_facecolor(left_header_color)
    ax_left_hdr.text(
        0.5, 0.5, left_title,
        color="white", fontsize=13, fontweight="bold",
        ha="center", va="center"
    )
    ax_left_hdr.set_xticks([])
    ax_left_hdr.set_yticks([])
    for spine in ax_left_hdr.spines.values():
        spine.set_color(left_header_color)

    ax_left_spec = fig.add_axes([0.04, 0.28, 0.44, 0.56])
    ax_left_spec.set_facecolor("#0b0c10")
    ax_left_spec.pcolormesh(left_t, left_f, left_spec, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    ax_left_spec.set_ylim([0, left_f[-1]])
    ax_left_spec.set_xticks([])
    ax_left_spec.set_yticks([])
    for spine in ax_left_spec.spines.values():
        spine.set_color("#2a2b30")
        spine.set_linewidth(1.5)

    # Simulated audio player bar underneath left
    ax_left_player = fig.add_axes([0.04, 0.12, 0.44, 0.12])
    ax_left_player.set_facecolor("#e5e7eb")
    ax_left_player.plot(0.08, 0.5, marker=">", markersize=10, color="#374151")
    ax_left_player.text(0.14, 0.5, f"0:00 / {int(duration_s // 60):02d}:{int(duration_s % 60):02d}", color="#4b5563", fontsize=10, va="center", family="sans-serif")
    ax_left_player.plot([0.38, 0.88], [0.5, 0.5], color="#9ca3af", lw=4, solid_capstyle="round")
    ax_left_player.plot([0.38, 0.40], [0.5, 0.5], color="#2563eb", lw=4, solid_capstyle="round")
    ax_left_player.plot(0.92, 0.5, marker="s", markersize=6, color="#4b5563")
    ax_left_player.set_xlim([0, 1])
    ax_left_player.set_ylim([0, 1])
    ax_left_player.axis("off")

    # Right Card
    ax_right_hdr = fig.add_axes([0.52, 0.86, 0.44, 0.08])
    ax_right_hdr.set_facecolor(right_header_color)
    ax_right_hdr.text(
        0.5, 0.5, right_title,
        color="white", fontsize=13, fontweight="bold",
        ha="center", va="center"
    )
    ax_right_hdr.set_xticks([])
    ax_right_hdr.set_yticks([])
    for spine in ax_right_hdr.spines.values():
        spine.set_color(right_header_color)

    ax_right_spec = fig.add_axes([0.52, 0.28, 0.44, 0.56])
    ax_right_spec.set_facecolor("#0b0c10")
    ax_right_spec.pcolormesh(right_t, right_f, right_spec, cmap="inferno", vmin=vmin, vmax=vmax, shading="auto")
    ax_right_spec.set_ylim([0, right_f[-1]])
    ax_right_spec.set_xticks([])
    ax_right_spec.set_yticks([])
    for spine in ax_right_spec.spines.values():
        spine.set_color("#2a2b30")
        spine.set_linewidth(1.5)

    # Simulated audio player bar underneath right
    ax_right_player = fig.add_axes([0.52, 0.12, 0.44, 0.12])
    ax_right_player.set_facecolor("#e5e7eb")
    ax_right_player.plot(0.08, 0.5, marker=">", markersize=10, color="#374151")
    ax_right_player.text(0.14, 0.5, f"0:00 / {int(duration_s // 60):02d}:{int(duration_s % 60):02d}", color="#4b5563", fontsize=10, va="center", family="sans-serif")
    ax_right_player.plot([0.38, 0.88], [0.5, 0.5], color="#9ca3af", lw=4, solid_capstyle="round")
    ax_right_player.plot([0.38, 0.40], [0.5, 0.5], color="#2563eb", lw=4, solid_capstyle="round")
    ax_right_player.plot(0.92, 0.5, marker="s", markersize=6, color="#4b5563")
    ax_right_player.set_xlim([0, 1])
    ax_right_player.set_ylim([0, 1])
    ax_right_player.axis("off")

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def generate_interactive_html(
    left_wav_rel: str,
    right_wav_rel: str,
    left_spec_rel: str,
    right_spec_rel: str,
    duration_s: float,
    html_path: Path,
    left_title: str = "Audio Stream 1",
    right_title: str = "Audio Stream 2",
    left_metrics: dict | None = None,
    right_metrics: dict | None = None,
):
    """Generate interactive comparison HTML player with synchronized playback & A/B switching."""
    left_noise_str = f"{left_metrics['noise_floor_db']:.1f} dBFS" if left_metrics else "N/A"
    right_noise_str = f"{right_metrics['noise_floor_db']:.1f} dBFS" if right_metrics else "N/A"
    left_rms_str = f"{left_metrics['rms_db']:.1f} dBFS" if left_metrics else "N/A"
    right_rms_str = f"{right_metrics['rms_db']:.1f} dBFS" if right_metrics else "N/A"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MicForge Audio Comparison</title>
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
    .controls-bar {{
      margin-bottom: 20px;
      display: flex;
      gap: 12px;
    }}
    .btn {{
      background: #3b82f6;
      color: white;
      border: none;
      border-radius: 8px;
      padding: 10px 18px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }}
    .btn:hover {{
      background: #2563eb;
    }}
    .btn-secondary {{
      background: #334155;
    }}
    .btn-secondary:hover {{
      background: #475569;
    }}
    .comparison-container {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 24px;
      max-width: 1080px;
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
      padding: 14px 18px;
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
    .metrics {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 10px;
      font-size: 12px;
      color: #94a3b8;
    }}
    .metric-item {{
      background: #0f172a;
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid #334155;
    }}
    .metric-val {{
      font-weight: bold;
      color: #f1f5f9;
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
    <h1>🎙️ MicForge Audio Comparison</h1>
    <p>Time-Frequency Spectrogram & Audio Inspection</p>
  </div>

  <div class="controls-bar">
    <button class="btn" id="playSyncBtn" onclick="toggleSyncPlay()">▶ Play Both Synced</button>
    <button class="btn btn-secondary" onclick="toggleAB()">⇄ Toggle A / B Focus</button>
  </div>

  <div class="comparison-container">
    <!-- Left Card -->
    <div class="card">
      <div class="card-header">
        {left_title}
        <div><span class="badge badge-noisy">Stream A</span></div>
      </div>
      <div class="spectrogram-container">
        <img class="spectrogram-img" src="{left_spec_rel}" alt="Stream A Spectrogram">
      </div>
      <div class="player-container">
        <audio id="audioA" controls src="{left_wav_rel}"></audio>
        <div class="metrics">
          <div class="metric-item">RMS: <span class="metric-val">{left_rms_str}</span></div>
          <div class="metric-item">Noise Floor: <span class="metric-val">{left_noise_str}</span></div>
        </div>
      </div>
    </div>

    <!-- Right Card -->
    <div class="card">
      <div class="card-header processed">
        {right_title}
        <div><span class="badge badge-clean">Stream B</span></div>
      </div>
      <div class="spectrogram-container">
        <img class="spectrogram-img" src="{right_spec_rel}" alt="Stream B Spectrogram">
      </div>
      <div class="player-container">
        <audio id="audioB" controls src="{right_wav_rel}"></audio>
        <div class="metrics">
          <div class="metric-item">RMS: <span class="metric-val">{right_rms_str}</span></div>
          <div class="metric-item">Noise Floor: <span class="metric-val">{right_noise_str}</span></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const a = document.getElementById('audioA');
    const b = document.getElementById('audioB');
    let activeFocus = 'A';

    function toggleSyncPlay() {{
      const btn = document.getElementById('playSyncBtn');
      if (a.paused && b.paused) {{
        b.currentTime = a.currentTime;
        a.play();
        b.play();
        btn.textContent = '⏸ Pause Both';
      }} else {{
        a.pause();
        b.pause();
        btn.textContent = '▶ Play Both Synced';
      }}
    }}

    function toggleAB() {{
      const curTime = a.currentTime || b.currentTime;
      if (activeFocus === 'A') {{
        activeFocus = 'B';
        a.muted = true;
        b.muted = false;
        b.currentTime = curTime;
        if (!a.paused) b.play();
      }} else {{
        activeFocus = 'A';
        b.muted = true;
        a.muted = false;
        a.currentTime = curTime;
        if (!b.paused) a.play();
      }}
    }}
  </script>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")


def run_enhancement_comparison(
    input_file: str | Path,
    output_dir: str | Path | None = None,
    preset: str = "discord",
    strength: float = 1.0,
):
    """Process an audio file through MicForge architecture and generate side-by-side comparison."""
    in_path = Path(input_file).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {in_path}")

    stem = in_path.stem  # e.g. "2"

    if output_dir is None:
        out_dirs = [
            _REPO_ROOT / "test_audios" / "outputs" / f"{stem}&processed",
            _REPO_ROOT / "test_audio" / "outputs" / f"{stem}&processed",
        ]
    else:
        out_dirs = [Path(output_dir).resolve()]

    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Reading input audio: {in_path.name}")
    audio, sr = sf.read(str(in_path))
    if audio.ndim > 1:
        audio = audio[:, 0]  # Mono channel
    audio = audio.astype(np.float32)

    duration_s = len(audio) / sr
    print(f"      Duration: {duration_s:.2f}s | Sample Rate: {sr} Hz | Frames: {len(audio)}")

    # 2. Build full MicForge DSP Chain Architecture
    print(f"[2/4] Initializing full MicForge DSP Chain ({preset} preset, strength={strength})...")
    block_size = int(sr * 0.01)  # 10 ms blocks
    app = AppConfig(sample_rate=sr, block_size=block_size, preset=preset)
    chain, _ = make_chain(app, preset)

    # Set denoiser strength
    dn = chain.stage("denoise")
    if dn is not None and hasattr(dn, "set_strength"):
        dn.set_strength(strength)

    # Calibrate noise gate threshold
    frame_size = int(sr * 0.1)
    frames_rms = [np.sqrt(np.mean(audio[i : i + frame_size] ** 2)) for i in range(0, len(audio), frame_size)]
    min_rms = np.min(frames_rms) if frames_rms else 1e-4
    noise_floor_db = 20.0 * np.log10(max(float(min_rms), 1e-6))

    gate = chain.stage("gate")
    if gate is not None:
        gate.threshold_db = min(noise_floor_db + 6.0, -32.0)
        gate.range_db = 28.0
        print(f"      Calibrated Noise Floor: {noise_floor_db:.1f} dBFS -> Noise Gate Threshold: {gate.threshold_db:.1f} dBFS")

    print("      Processing block-by-block through full 11-stage DSP chain...")
    enhanced = process_offline(chain, audio, block=block_size)

    # Compute metrics
    m_noisy = compute_audio_metrics(audio, sr)
    m_enh = compute_audio_metrics(enhanced, sr)
    snr_gain = m_noisy["rms_db"] - m_enh["rms_db"]
    print(f"      Input RMS: {m_noisy['rms_db']:.1f} dBFS -> Output RMS: {m_enh['rms_db']:.1f} dBFS (Delta: {snr_gain:+.1f} dB)")

    # 3. Compute Spectrograms
    print("[3/4] Computing time-frequency spectrograms...")
    noisy_f, noisy_t, noisy_spec, vmin1, vmax1 = compute_spectrogram(audio, sr)
    enh_f, enh_t, enh_spec, vmin2, vmax2 = compute_spectrogram(enhanced, sr)
    common_vmin = min(vmin1, vmin2)
    common_vmax = max(vmax1, vmax2)

    # 4. Save Audio and Images
    print("[4/4] Writing output assets and comparison images...")
    primary = out_dirs[0]
    noisy_wav = primary / f"{stem}_noisy.wav"
    enh_wav = primary / f"{stem}_processed.wav"
    noisy_spec_png = primary / f"{stem}_noisy_spec.png"
    enh_spec_png = primary / f"{stem}_processed_spec.png"
    side_by_side_png = primary / "comparison_side_by_side.png"
    html_file = primary / "index.html"

    sf.write(str(noisy_wav), audio, sr)
    sf.write(str(enh_wav), enhanced, sr)

    plot_single_spectrogram(noisy_f, noisy_t, noisy_spec, common_vmin, common_vmax, noisy_spec_png)
    plot_single_spectrogram(enh_f, enh_t, enh_spec, common_vmin, common_vmax, enh_spec_png)

    render_comparison_card_image(
        noisy_f, noisy_t, noisy_spec,
        enh_f, enh_t, enh_spec,
        common_vmin, common_vmax,
        duration_s,
        side_by_side_png,
        left_title=f"Noisy speech sample {stem}",
        right_title="Enhanced speech by MicForge (CoFi-Lite + DSP)",
    )

    generate_interactive_html(
        left_wav_rel=noisy_wav.name,
        right_wav_rel=enh_wav.name,
        left_spec_rel=noisy_spec_png.name,
        right_spec_rel=enh_spec_png.name,
        duration_s=duration_s,
        html_path=html_file,
        left_title=f"Noisy speech sample {stem}",
        right_title="Enhanced speech by MicForge (CoFi-Lite + DSP)",
        left_metrics=m_noisy,
        right_metrics=m_enh,
    )

    # Mirror to secondary directories
    for od in out_dirs[1:]:
        od.mkdir(parents=True, exist_ok=True)
        for f in [noisy_wav, enh_wav, noisy_spec_png, enh_spec_png, side_by_side_png, html_file]:
            shutil.copy2(f, od / f.name)

    print("\n[SUCCESS] Generated comparison assets in:")
    for od in out_dirs:
        print(f"   📂 {od}")
        print(f"      • {noisy_wav.name}")
        print(f"      • {enh_wav.name}")
        print(f"      • {noisy_spec_png.name}")
        print(f"      • {enh_spec_png.name}")
        print(f"      • {side_by_side_png.name}")
        print(f"      • {html_file.name}")

    return {
        "noisy_wav": noisy_wav,
        "enhanced_wav": enh_wav,
        "side_by_side": side_by_side_png,
        "html": html_file,
    }


def run_direct_two_file_comparison(
    file1: str | Path,
    file2: str | Path,
    output_dir: str | Path,
    label1: str = "Audio File 1",
    label2: str = "Audio File 2",
):
    """Compare two arbitrary audio files side-by-side."""
    f1_path = Path(file1).resolve()
    f2_path = Path(file2).resolve()
    if not f1_path.is_file():
        raise FileNotFoundError(f"File 1 not found: {f1_path}")
    if not f2_path.is_file():
        raise FileNotFoundError(f"File 2 not found: {f2_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading files:\n      A: {f1_path.name}\n      B: {f2_path.name}")
    a1, sr1 = sf.read(str(f1_path))
    a2, sr2 = sf.read(str(f2_path))

    if a1.ndim > 1:
        a1 = a1[:, 0]
    if a2.ndim > 1:
        a2 = a2[:, 0]

    a1 = a1.astype(np.float32)
    a2 = a2.astype(np.float32)

    dur = max(len(a1) / sr1, len(a2) / sr2)

    m1 = compute_audio_metrics(a1, sr1)
    m2 = compute_audio_metrics(a2, sr2)

    print("[2/3] Computing spectrograms...")
    f1, t1, s1, vmin1, vmax1 = compute_spectrogram(a1, sr1)
    f2, t2, s2, vmin2, vmax2 = compute_spectrogram(a2, sr2)
    common_vmin = min(vmin1, vmin2)
    common_vmax = max(vmax1, vmax2)

    print("[3/3] Generating comparison cards and HTML player...")
    out_a = out_dir / f"stream_A_{f1_path.name}"
    out_b = out_dir / f"stream_B_{f2_path.name}"
    spec_a = out_dir / f"{f1_path.stem}_spec.png"
    spec_b = out_dir / f"{f2_path.stem}_spec.png"
    side_by_side = out_dir / "comparison_side_by_side.png"
    html_file = out_dir / "index.html"

    shutil.copy2(f1_path, out_a)
    shutil.copy2(f2_path, out_b)

    plot_single_spectrogram(f1, t1, s1, common_vmin, common_vmax, spec_a)
    plot_single_spectrogram(f2, t2, s2, common_vmin, common_vmax, spec_b)

    render_comparison_card_image(
        f1, t1, s1,
        f2, t2, s2,
        common_vmin, common_vmax,
        dur,
        side_by_side,
        left_title=label1,
        right_title=label2,
        left_header_color="#3da35d",
        right_header_color="#2563eb",
    )

    generate_interactive_html(
        left_wav_rel=out_a.name,
        right_wav_rel=out_b.name,
        left_spec_rel=spec_a.name,
        right_spec_rel=spec_b.name,
        duration_s=dur,
        html_path=html_file,
        left_title=label1,
        right_title=label2,
        left_metrics=m1,
        right_metrics=m2,
    )

    print(f"\n[SUCCESS] Generated two-audio comparison in {out_dir}:")
    print(f"   • {side_by_side.name}")
    print(f"   • {html_file.name}")
    return {"side_by_side": side_by_side, "html": html_file}


def main():
    parser = argparse.ArgumentParser(description="MicForge comparison tool (two audios or enhancement)")
    parser.add_argument("files", nargs="*", default=[], help="Audio file(s) to compare. Pass 0 for default 2.wav, 1 to enhance, 2 to compare directly.")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--preset", "-p", default="discord", help="Preset when enhancing a file (discord, gaming, broadcast, studio, noisy_room)")
    parser.add_argument("--strength", "-s", type=float, default=1.0, help="Denoiser strength (0.0 to 1.0)")
    parser.add_argument("--label1", default=None, help="Label for stream 1")
    parser.add_argument("--label2", default=None, help="Label for stream 2")
    args = parser.parse_args()

    files = args.files
    if len(files) == 0:
        # Default behavior: compare sample 2 against MicForge enhancement
        default_2 = _REPO_ROOT / "test_audios" / "2.wav"
        if not default_2.is_file():
            default_2 = _REPO_ROOT / "test_audios" / "1.wav"
        run_enhancement_comparison(default_2, output_dir=args.output, preset=args.preset, strength=args.strength)
    elif len(files) == 1:
        run_enhancement_comparison(files[0], output_dir=args.output, preset=args.preset, strength=args.strength)
    elif len(files) >= 2:
        out = args.output or (_REPO_ROOT / "test_audios" / "outputs" / "comparison_two")
        label1 = args.label1 or Path(files[0]).stem
        label2 = args.label2 or Path(files[1]).stem
        run_direct_two_file_comparison(files[0], files[1], output_dir=out, label1=label1, label2=label2)


if __name__ == "__main__":
    main()

