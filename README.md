# Micforge

[![Version](https://img.shields.io/badge/version-1.1.1-0284c7.svg)](version%20notes/v1.1.1.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-slate.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-emerald.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-74%2F74%20passing-teal.svg)](tests/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-neutral.svg)]()

> **Real-time, low-latency speech enhancement and DSP engine for voice chat, streaming and gaming.**  
> Built and maintained by **[Parth Sadaria](https://github.com/Parthsadaria)**.

---

## Overview

**Micforge** is an open-source, local-first real-time audio processor that makes entry-level and untreated microphones sound cleaner and clearer. It combines a classical multi-stage dynamics/EQ chain with a **small neural speech-enhancement model** based on the **[CoFi-Lite](https://arxiv.org/abs/2607.10142)** architecture, and falls back to a classical spectral Wiener denoiser when the neural model is unavailable.

Everything runs on the local CPU with **no cloud dependencies**. On a Ryzen 7 the whole process uses **about 100 MB of RAM** and mic-to-virtual-cable latency is **about 68 ms\***. Processed audio is routed into Discord, OBS Studio, Zoom and games through a standard virtual audio cable.

> \* Measured on an AMD Ryzen 7 machine (Windows, 48 kHz). See [Notes on measurements](#notes-on-measurements).

```
                  ┌────────────────────── 8-Stage DSP Signal Chain ──────────────────────┐
                  │                                                                      │
Microphone Input ─┼─▶ High-Pass Filter (4th-order, 90 Hz default)                        │
                  │   └─▶ Neural Denoiser (or Spectral Wiener fallback)                  │
                  │       └─▶ Downward Expander / Noise Gate                             │
                  │           └─▶ Parametric EQ (De-mud / Presence / Air)                │
                  │               └─▶ Split-Band De-Esser                                │
                  │                   └─▶ Slow RMS Auto-Leveler                          │
                  │                       └─▶ Feed-Forward Compressor                    │
                  │                           └─▶ Peak Limiter (hard-clip safety)        │
                  └───────────────────────────────────┬──────────────────────────────────┘
                                                      │
                                                      ├──▶ Virtual Cable Output ──▶ Discord / OBS / Games
                                                      └──▶ Headphone Monitor (with isolated audio cues)
```

---

## Key Capabilities

- **Hybrid noise suppression:** Suppresses steady and non-stationary background noise (fans, keyboard clatter, street noise) with a lightweight ONNX neural model, with a one-key hot-swap to classical spectral Wiener denoising.
- **Low latency:** About **67.6 ms\*** mic-to-virtual-cable latency with the neural model and **47.6 ms\*** with classical filtering.
- **Glitch-resistant audio pipeline:** Startup sink gating and a moving-average clock-drift compensator reduce startup underruns and sample drift between mismatched hardware clocks.
- **Windows system tray daemon (`--tray`):** Runs in the background with a taskbar icon reflecting mic state (Active, Muted, Bypassed).
- **Global hotkeys:** OS-level keyboard hooks (`Ctrl+Shift+M` mute, `Ctrl+Shift+B` bypass, push-to-talk) work across most applications. Exclusive-fullscreen games and anti-cheat–protected titles may block keyboard hooks; use borderless/windowed mode if hotkeys don't register.
- **Isolated feedback chimes:** Procedural, click-free cues (*blip* on mute, *chirp* on unmute) are synthesized in memory and sent only to the headphone monitor, never to the virtual microphone.
- **Settings GUI:** A dark-themed interface (`micforge settings`) for hotkeys, chime volume and Windows autostart.

> **Realistic expectations:** Micforge cleans up noise, levels and tone. It cannot restore high frequencies a microphone never captured. Mic placement (about a fist from your mouth) still matters more than any filter.

---

## Quick Start

### 1. Prerequisites
- **Python 3.10** or higher
- A virtual audio cable driver to route processed output into your voice software:
  - **Windows:** [VB-CABLE Driver](https://vb-audio.com/Cable/) *(recommended, free)*
  - **macOS:** [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) (`brew install blackhole-2ch`)
  - **Linux:** PulseAudio/PipeWire null sink (`pactl load-module module-null-sink sink_name=micforge`)

### 2. Installation
```bash
git clone https://github.com/Parthsadaria/MicForge.git
cd MicForge
pip install -e .
```
*(Debian/Ubuntu: `sudo apt install libportaudio2` · macOS: `brew install portaudio`)*

### 3. Setup & Calibration
```bash
micforge doctor      # verify environment and audio drivers
micforge setup       # pick physical mic, virtual cable and headphones
micforge calibrate   # sample room noise for a few seconds (sets the gate threshold)
micforge run         # start the terminal dashboard
```

> **Windows tip:** use the **WASAPI** entries for your mic and for `CABLE Input` in `micforge setup`. The legacy MME entries add latency and cause clock drift.

### 4. Application Configuration
In Discord, OBS Studio, Steam or Zoom:
1. Set **Input Device** to `CABLE Output (VB-Audio Virtual Cable)`.
2. Turn off downstream **Noise Suppression** and **Automatic Gain Control** to avoid double processing.
3. **Echo Cancellation:** leave it **on** if you use speakers; it is safe to turn off with headphones.

To run Micforge in the background:
```bash
micforge --tray
```

---

## Live Terminal Dashboard

`micforge run` opens a full-screen Rich dashboard (illustrative layout):

```
╭─ MIC ENHANCER ───────── PRESET: discord ────────── MODEL: neural ─────────────╮
│ IN   [-42.1 dBFS]  █████████████································              │
│ OUT  [-18.4 dBFS]  ████████████████████████████················  PEAK HOLD    │
│ GR   [ -8.2 dB  ]  ████████····································  GATE: OPEN   │
├───────────────────────────────────────────────────────────────────────────────┤
│ SPECTRUM (raw vs processed)                                                   │
│   100 Hz       500 Hz        1 kHz         4 kHz        10 kHz       16 kHz   │
├───────────────────────────────────────────────────────────────────────────────┤
│ latency   ~ 67.6 ms*       noise floor  -64.2 dBFS                            │
│ dsp load  ███·······  18%  input gain   +2.0 dB                               │
│ glitches  0                clock fixes  0                                     │
╰───────────────────────────────────────────────────────────────────────────────╯
```

### Interactive Controls
| Key | Action | Key | Action |
|:---:|:---|:---:|:---|
| `b` | Toggle A/B bypass (raw vs processed) | `c` | Calibrate room noise (3 s) |
| `x` | Mute / unmute microphone | `+` / `-` | Adjust denoiser strength |
| `l` | Toggle headphone monitoring | `[` / `]` | Input gain (±1 dB) |
| `p` | Cycle presets | `m` | Hot-swap denoiser (neural ⟷ Wiener) |
| `q` | Quit | | |

---

## Engineering Notes

### 1. Latency
Real-time Python audio pipelines often end up well above 100 ms because of driver double-buffering, long STFT windows and conservative jitter buffers.

Micforge v1.1.1 reaches about **67.6 ms\*** (neural) and **47.6 ms\*** (spectral Wiener) by:
- **Reducing causal STFT delay:** algorithmic group delay of the neural path reduced from 2,335 samples (48.6 ms) to 1,470 samples (30.6 ms) at 48 kHz.
- **Smaller jitter cushion:** default output buffer of **1.5 blocks (15 ms)**, configurable via `jitter_blocks`.
- **Measuring real buffer occupancy** instead of driver-reported theoretical latency.

### 2. Startup Gating & Clock-Drift Compensation
- **Sink gating:** output streams stay idle until the input stream delivers its first frame, which suppresses the burst of startup underruns seen when WASAPI outputs start before the input is ready.
- **Drift compensation:** independent hardware clocks drift apart slowly. Micforge keeps a moving average of output-buffer occupancy and applies small sample-level corrections to keep it centred.
- **Allocation-light callbacks:** DSP state is preallocated; the remaining per-block NumPy temporaries are small and short-lived.

> [!TIP]
> **Crackles or rising glitch counter?** Real-time audio is sensitive to CPU load, USB bandwidth and Windows DPC latency. Make sure mic and cable use **WASAPI**, close heavy background apps, then raise `"jitter_blocks"` in `%APPDATA%\micforge\config.json` (e.g. from `1.5` to `2.5` or `3.0`).

### 3. Neural Model
- **Architecture:** CoFi-Lite (per the paper: ~83 k parameters, 12.87 M MACs/s), running through ONNX Runtime on one CPU thread.
- **Resource use:** the whole Micforge process uses about **100 MB RAM\*** (Python, NumPy/SciPy and ONNX Runtime dominate; the model itself is under 1 MB).
- **Processing rate:** the model runs at 16 kHz internally.
- **Tooling:**
  ```bash
  micforge model info                               # inspect the loaded ONNX model
  micforge model export checkpoint.tar model.onnx  # export a PyTorch checkpoint to ONNX
  micforge process input.wav output.wav -p studio  # offline batch processing
  ```

#### Model provenance
The neural denoiser uses the **CoFi-Lite** architecture. The upstream authors released their source code and a checkpoint trained on a simulated DNS3 dataset on **2026-10-19** ([github.com/Acceleration123/CoFi-Lite](https://github.com/Acceleration123/CoFi-Lite)). `models/cofilite.onnx` is exported from that official checkpoint (see `micforge model export`). To use a different or newer checkpoint, export it to ONNX and pass it with `--model`.

- **Training data:** the released checkpoint was trained on simulated DNS3 data, so results on your real microphone and room may differ from the paper's benchmarks. Use `micforge test` and the `m` key (neural vs Wiener) to compare on your own voice.
- **License:** the bundled model is derived from the upstream checkpoint and is subject to **its** license terms, not necessarily this project's MIT license. Check the upstream repository's license before redistributing the model file or using it commercially.

Reference: L. Yang, D. Wang, X. Rong, J. Zhao, J. Lu, *CoFi-Lite: Pushing the Limits of Ultra-Lightweight Speech Enhancement*, IEEE Signal Processing Letters, 2026 ([arXiv:2607.10142](https://arxiv.org/abs/2607.10142)).

### 4. Isolated Feedback Chimes
Cues are synthesized on demand with logarithmic sine sweeps and half-Hann envelopes to avoid edge clicks:
- **Mute:** 460 Hz → 280 Hz descending sweep.
- **Unmute:** 280 Hz → 540 Hz ascending sweep.
- **Routing:** chimes go only to the monitor sink; the virtual cable never receives them.

---

## DSP Specification

The engine processes 32-bit float audio through eight stages (default `discord` preset shown; all values are configurable):

| Stage | Algorithm | Function | Defaults |
|:---|:---|:---|:---|
| **1. High-pass** | 4th-order Butterworth (two cascaded biquads) | Removes rumble and desk thumps | 90 Hz |
| **2. Denoise** | Neural model, or STFT Wiener with minimum-statistics noise tracking | Noise suppression | 16 kHz internal (neural) |
| **3. Gate** | Soft downward expander with hold and hysteresis | Attenuates residual noise between phrases | Threshold set by `calibrate` (default −52 dBFS), 20 dB range |
| **4. Parametric EQ** | Biquad cascade | De-mud, presence, air | 300 Hz −2.5 dB · 3.2 kHz +3 dB · 9 kHz high shelf +3 dB |
| **5. De-esser** | Split-band dynamic attenuator | Tames harsh sibilance | ~7 kHz band |
| **6. Leveler** | Slow RMS auto-gain | Evens out quiet vs loud speech | Attack ~0.25 s, release ~1.5 s, up to +9 dB |
| **7. Compressor** | Feed-forward, soft knee | Transient control, consistency | 3:1, 6 ms attack, 90 ms release |
| **8. Limiter** | Peak limiter with hard-clip safety | Prevents digital overs | Ceiling −1.5 dBFS |

---

## Notes on measurements

Figures marked with **\*** were measured on a single machine and will vary:

| Metric | Value | Conditions |
|:---|:---|:---|
| End-to-end latency (mic → virtual cable) | **67.6 ms\*** neural · **47.6 ms\*** spectral | AMD Ryzen 7, Windows, WASAPI mic and cable, 48 kHz |
| Process RAM | **~100 MB\*** | Same machine, neural denoiser active |

Latency depends heavily on your audio hardware, drivers, host API (WASAPI vs MME), `block_size` / `jitter_blocks`, and system load. It excludes the extra buffering added by the receiving app (Discord, OBS, …) and by the virtual cable driver. Run `micforge doctor` to see DSP speed and the latency reported on your machine.

---

## Repository Structure

```
micforge/
├── src/micforge/
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── stats.py             # Audio analysis, SNR calculation, diagnostic heuristics
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── backend.py           # sounddevice / PortAudio interface
│   │   ├── cues.py              # Procedural sine-sweep feedback chimes
│   │   ├── devices.py           # Device discovery, host-API query, virtual-cable detection
│   │   ├── engine.py            # Stream lifecycle, sink gating, drift compensation
│   │   ├── fileio.py            # WAV read/write
│   │   ├── metrics.py           # RMS, peak hold, clipping, spectrum
│   │   ├── recorder.py          # Recording and playback helpers
│   │   └── ring.py              # Ring buffer between input and output streams
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── calibrate.py         # Room-noise sampling and gate threshold
│   │   ├── common.py            # Shared CLI helpers
│   │   ├── devices_cmd.py       # Device listing and setup
│   │   ├── doctor.py            # Driver, device and performance diagnostics
│   │   ├── files.py             # Offline WAV processing and analysis
│   │   ├── live.py              # Real-time dashboard and tray runner
│   │   ├── test_mic.py          # Record, compare and A/B playback
│   │   └── wizard.py            # Interactive menu / first-run setup
│   ├── dsp/
│   │   ├── __init__.py
│   │   ├── base.py              # Processor base class
│   │   ├── biquad.py            # Biquad filters (streaming SOS)
│   │   ├── chain.py             # DSP chain assembly
│   │   ├── cofilite.py          # Neural denoiser: feature extraction + ONNX inference
│   │   ├── compressor.py        # Compressor and peak limiter
│   │   ├── deesser.py           # Split-band de-esser
│   │   ├── denoise.py           # Classical spectral Wiener denoiser
│   │   ├── export_cofilite.py   # Checkpoint -> ONNX converter
│   │   ├── gate.py              # Soft noise gate / downward expander
│   │   ├── leveler.py           # Slow RMS auto-leveler
│   │   └── util.py              # dB conversion and DSP helpers
│   ├── models/
│   │   └── cofilite.onnx        # Bundled model (see "Model provenance")
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── dashboard.py         # Rich terminal dashboard
│   │   ├── hotkeys.py           # Global keyboard hook manager
│   │   ├── keys.py              # Cross-platform console key reader
│   │   ├── report.py            # Before/after report formatting
│   │   ├── settings_gui.py      # Tkinter settings window
│   │   ├── theme.py             # Colors and styles
│   │   ├── tray.py              # System tray icon and menu
│   │   └── widgets.py           # Meters, spectrum and bar widgets
│   ├── util/
│   │   ├── __init__.py
│   │   └── autostart.py         # Windows boot autostart (HKCU Run key)
│   ├── __init__.py  __main__.py  cli.py  config.py  errors.py
│   └── logging_setup.py  presets.py  profile_store.py
├── tests/                       # pytest suite with a fake PortAudio backend (no hardware needed)
│   ├── conftest.py  fake_sd.py
│   └── test_cofilite.py  test_cues.py  test_dsp.py  test_engine.py  test_hotkeys.py  test_misc.py
├── version notes/
│   └── v1.1.1.md                # Release notes and changelog
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## CLI Reference

```
usage: micforge [-h] [-v] [--debug] [--model MODEL] [--tray] <command> ...

Commands:
  run             Live pipeline (mic -> DSP -> virtual cable)
  setup           Select mic, virtual cable and headphones
  calibrate       Measure room noise; sets the gate threshold
  doctor          Check drivers, devices and DSP performance
  settings        Graphical settings: hotkeys, chimes, startup
  loopback        Hear yourself live through headphones (A/B raw vs processed)
  test            Record a short sample, show before/after metrics, play both back
  presets         List presets or print one as JSON
  process         Process a WAV file offline
  analyze         Analyze a WAV file (level, noise, clarity)
  model           Inspect or export the neural ONNX model
  devices         List audio devices
  version         Print version and environment info
```

---

## Testing

Audio drivers are mocked, so the suite runs on headless machines without sound hardware:

```bash
pip install -e .[dev]
pytest -v
```

---

## Built With

| Component | Project | Role |
|:---|:---|:---|
| Neural model | [CoFi-Lite](https://github.com/Acceleration123/CoFi-Lite) ([paper](https://arxiv.org/abs/2607.10142)) | Lightweight causal speech-enhancement architecture and released checkpoint |
| Inference | [ONNX Runtime](https://github.com/microsoft/onnxruntime) | CPU execution of the neural model |
| Audio I/O | [sounddevice](https://github.com/spatialaudio/python-sounddevice) · [PortAudio](http://www.portaudio.com/) | Real-time streams (WASAPI, ALSA, CoreAudio) |
| Numerics / DSP | [NumPy](https://github.com/numpy/numpy) · [SciPy](https://github.com/scipy/scipy) | Filters, FFTs, buffers, synthesis |
| Terminal UI | [Rich](https://github.com/Textualize/rich) | Dashboard, meters, spectrum |
| System tray | [pystray](https://github.com/moses-palmer/pystray) · [Pillow](https://github.com/python-pillow/Pillow) | Tray icon and menu |
| Global hotkeys | [pynput](https://github.com/moses-palmer/pynput) | OS-level keyboard hooks |
| Settings GUI | [Tkinter](https://docs.python.org/3/library/tkinter.html) | Settings window |
| Virtual audio | [VB-CABLE](https://vb-audio.com/Cable/) · [BlackHole](https://github.com/ExistentialAudio/BlackHole) | Routing into Discord, OBS, VoIP |
| Tests | [pytest](https://github.com/pytest-dev/pytest) | Test runner |

---

## Author & Credits

- **Lead developer & maintainer:** [Parth Sadaria](https://github.com/Parthsadaria)
- **Neural model:** CoFi-Lite by Leyan Yang, Dahan Wang, Xiaobin Rong, Jiadong Zhao and Jing Lu. Micforge is not affiliated with the authors.
- **Feedback & contributions:** issues and pull requests are welcome on [GitHub](https://github.com/Parthsadaria/MicForge).

---

## License

Micforge is released under the **MIT License**; see `LICENSE`. Third-party components are governed by their own licenses: this includes the bundled `cofilite.onnx` (derived from the upstream CoFi-Lite checkpoint), VB-CABLE and BlackHole.