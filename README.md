# Micforge

[![Version](https://img.shields.io/badge/version-1.1.0-0284c7.svg)](version%20notes/v1.1.0.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-slate.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-emerald.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-65%2F65%20passing-teal.svg)](tests/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-neutral.svg)]()

> **Real-time, ultra-low latency speech enhancement and digital signal processing engine for voice communication and broadcasting.**  
> Built and maintained by **[Parth Sadaria](https://github.com/Parthsadaria)**.

---

## Overview

**Micforge** is an open-source, local-first real-time audio processor designed to transform entry-level and untreated microphones into studio-grade broadcast signals. It combines classical multi-stage dynamic DSP with **[CoFi-Lite](https://github.com/Acceleration123/CoFi-Lite)**, an ultra-lightweight causal neural speech enhancement model.

The engine executes completely on the local CPU with **zero cloud dependencies**, maintains a memory footprint **under 100 MB RAM**, delivers **sub-70 ms roundtrip latency**, and routes processed audio directly into Discord, OBS Studio, Zoom, and games via standard virtual audio cables.

```
                  ┌────────────────────── 8-Stage DSP Signal Chain ──────────────────────┐
                  │                                                                      │
Microphone Input ─┼─▶ High-Pass Filter (80 Hz)                                           │
                  │   └─▶ CoFi-Lite Neural Denoiser (or Spectral Wiener)                 │
                  │       └─▶ Downward Expander / Noise Gate                             │
                  │           └─▶ 3-Band Parametric EQ (De-mud / Presence / Air)         │
                  │               └─▶ Split-Band De-Esser (6.5 kHz)                      │
                  │                   └─▶ Optical RMS Auto-Leveler                       │
                  │                       └─▶ Feed-Forward Fast Compressor               │
                  │                           └─▶ True Peak Lookahead Limiter (-1 dBFS)  │
                  └───────────────────────────────────┬──────────────────────────────────┘
                                                      │
                                                      ├──▶ Virtual Cable Output ──▶ Discord / OBS / Games
                                                      └──▶ Headphone Monitor (with isolated audio cues)
```

---

## Key Capabilities

- **Hybrid Noise Suppression:** Erases non-stationary acoustic noise (mechanical key switches, fan turbulence, street noise, room reverberation) using an optimized **CoFi-Lite** ONNX neural network, with instantaneous on-the-fly fallback to classical spectral subtraction.
- **Ultra-Low Latency:** Engineered for real-time interactive voice chat and competitive gaming, achieving **67.6 ms** roundtrip pipeline latency with the neural model and **47.6 ms** with classical filtering.
- **Zero-Glitch Audio Pipeline:** Active sink gating and moving-average clock drift compensation eliminate startup buffer underruns and sample synchronization errors across mismatched hardware clocks.
- **Windows System Tray Daemon (`--tray`):** Operates silently in the background with a dynamic taskbar icon reflecting microphone state (Active, Muted, Bypassed).
- **System-Wide Global Hotkeys:** Low-level OS keyboard hooks capture commands (`Ctrl+Shift+M` mute toggle, `Ctrl+Shift+B` bypass toggle, Push-to-Talk) across all windows, including fullscreen 3D applications.
- **Isolated Feedback Chimes:** Procedural, click-free audio cues (*blip* on mute, *chirp* on unmute) synthesized in-memory and routed exclusively to the headphone monitor—never broadcast to the virtual microphone stream.
- **Settings & Hotkeys GUI:** Built-in dark-themed graphical interface (`micforge settings`) for visual hotkey assignment, chime volume adjustment, and Windows boot autostart configuration.

---

## Quick Start

### 1. Prerequisites
- **Python 3.10** or higher
- A virtual audio cable driver to route processed output to your voice software:
  - **Windows:** [VB-CABLE Driver](https://vb-audio.com/Cable/) *(Recommended, Free)*
  - **macOS:** [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) (`brew install blackhole-2ch`)
  - **Linux:** Native ALSA / PulseAudio null-sink (`pactl load-module module-null-sink sink_name=micforge`)

### 2. Installation
```bash
git clone https://github.com/Parthsadaria/MicForge.git
cd MicForge
pip install -e .
```
*(On Debian/Ubuntu: `sudo apt install libportaudio2` | On macOS: `brew install portaudio`)*

### 3. Setup & Calibration
```bash
# Verify environment and audio driver health
micforge doctor

# Select your physical microphone, virtual cable, and optional headphones
micforge setup

# Sample room ambient profile for 3 seconds (calibrates denoiser and gate)
micforge calibrate

# Start the interactive terminal dashboard
micforge run
```

### 4. Application Configuration
In Discord, OBS Studio, Steam, or Zoom:
1. Set **Input Device** to `CABLE Output (VB-Audio Virtual Cable)`.
2. Disable downstream noise suppression (e.g. Discord Krisp / Standard), Echo Cancellation, and Automatic Gain Control to avoid redundant double-processing.

To run Micforge permanently in the background:
```bash
micforge --tray
```

---

## Live Terminal Dashboard

Running `micforge run` launches a full-screen, low-overhead terminal dashboard rendered via Rich:

```
╭─ MIC ENHANCER ───────── PRESET: discord ────────── MODEL: CoFi-Lite (Neural) ─╮
│ IN   [-42.1 dBFS]  █████████████································              │
│ OUT  [-18.4 dBFS]  ████████████████████████████················  PEAK HOLD    │
│ GR   [ -8.2 dB  ]  ████████····································  GATE: OPEN   │
├───────────────────────────────────────────────────────────────────────────────┤
│ SPECTRUM (raw vs processed)                                                   │
│   100 Hz       500 Hz        1 kHz         4 kHz        10 kHz       16 kHz   │
├───────────────────────────────────────────────────────────────────────────────┤
│ latency   ~ 67.6 ms        noise floor  -64.2 dBFS                            │
│ dsp load  ███·······  18%  input gain   +2.0 dB                               │
│ glitches  0                clock fixes  0                                     │
╰───────────────────────────────────────────────────────────────────────────────╯
```

### Interactive Controls
| Key | Action | Key | Action |
|:---:|:---|:---:|:---|
| `b` | Toggle A/B Bypass (raw mic vs processed) | `c` | Calibrate ambient room noise (3 sec) |
| `x` | Mute / Unmute microphone | `+` / `-` | Adjust denoiser reduction strength |
| `l` | Toggle real-time headphone monitoring | `[` / `]` | Adjust input gain (±1 dB) |
| `p` | Cycle voice presets (*discord*, *clean*, *podcast*) | `m` | Hot-swap denoiser (CoFi-Lite ⟷ Wiener) |
| `q` | Exit application | | |

---

## Engineering & Architecture Deep Dive

### 1. Latency Optimization Pipeline
Standard real-time Python audio pipelines frequently incur 120 ms+ latency due to driver double-buffering, high STFT lookahead windows, and conservative circular buffers.

Micforge v1.1.0 reduces end-to-end latency to **67.6 ms** (CoFi-Lite) and **47.6 ms** (Spectral Wiener) through:
- **Causal STFT Delay Minimization:** Restructured the causal lookahead window inside the neural feature extractor, lowering algorithmic group delay from 2,335 samples (48.6 ms) to 1,470 samples (30.6 ms) at 48 kHz.
- **Jitter Cushion Optimization:** Reduced the PortAudio ring-buffer safety cushion from 3.0 blocks to **1.5 blocks (15 ms)** while preserving zero-underrun stability.
- **Hardware-Accurate Profiling:** Latency calculation reflects actual hardware FIFO queue occupancy rather than driver theoretical allowances.

### 2. Deterministic Startup & Clock Drift Compensation
- **Active Sink Gating:** Output streams are held in a gated idle state until the input stream captures and delivers its initial valid audio frame. This prevents initial WASAPI shared-mode underruns and eliminates the startup glitches common on Windows audio hosts.
- **Moving-Average Drift Synchronization:** Audio hardware clocks inherently experience subtle frequency variances (e.g., a 48,000.2 Hz USB microphone paired with a 47,999.7 Hz onboard DAC). Micforge employs a linear-algebra moving-average estimator that dynamically applies micro-resampling without audible pitch or phase distortion.
- **Zero Heap Allocations:** DSP execution loops operate strictly on pre-allocated NumPy contiguous arrays within PortAudio real-time callbacks.

### 3. Neural Architecture: CoFi-Lite
- **Parameters:** ~83,000 parameters (~600 KB compiled ONNX binary).
- **Inference Complexity:** 12.87M MACs/sec running via float32 SIMD ONNX Runtime.
- **Memory Consumption:** ~65–75 MB resident RAM, allowing permanent background execution without impacting gaming performance.
- **Model Tooling:**
  ```bash
  micforge model info                               # Inspect loaded ONNX architecture
  micforge model export checkpoint.tar model.onnx  # Export PyTorch weights to standalone ONNX
  micforge process input.wav output.wav -p studio  # Offline batch processing
  ```

### 4. Isolated In-Memory Audio Feedback
Feedback chimes are procedurally synthesized on demand using logarithmic sine sweeps with smooth half-Hanning window envelopes to eliminate edge clicks:
- **Mute Cue:** $460\text{ Hz} \to 280\text{ Hz}$ descending sweep.
- **Unmute Cue:** $280\text{ Hz} \to 540\text{ Hz}$ ascending sweep.
- **Architectural Routing:** Chimes are injected strictly into the monitor audio sink downstream of the virtual cable split. The virtual cable sink never receives chime samples, ensuring zero spillover into voice calls.

---

## DSP Signal Processing Specification

The audio engine processes signals in 32-bit floating point through 8 discrete stages:

| Stage | Algorithm | Function | Key Parameters |
|:---|:---|:---|:---|
| **1. High-Pass** | 2nd-order Butterworth Biquad | Rejection of mechanical sub-rumble and proximity thumps | $f_c = 80\text{ Hz}$, $Q = 0.707$ |
| **2. Denoise** | CoFi-Lite Neural / STFT Wiener | Suppression of stationary and non-stationary acoustic noise | $16\text{ kHz}$ internal, causal overlap-add |
| **3. Gate** | Downward Expander with Hysteresis | Attenuation of room reverberation and breath intake | Soft-knee, dynamic floor calibration |
| **4. Parametric EQ** | 3-Band Biquad Cascade | Tonal correction (de-mud, vocal presence, top-end air) | 300 Hz ($-3\text{ dB}$), 3.2 kHz ($+2.5\text{ dB}$), 10 kHz ($+2\text{ dB}$) |
| **5. De-Esser** | Split-Band Dynamic Attenuator | Transparent suppression of harsh vocal sibilance | $f_c = 6.5\text{ kHz}$, fast RMS envelope detection |
| **6. Leveler** | Slow Optical-Style RMS Leveler | Normalization of conversational dynamics (whispers vs shouts) | $\tau = 1.2\text{ s}$ time constant |
| **7. Compressor** | Feed-Forward VCA Emulation | Transient control and vocal consistency | $20\text{ ms}$ attack, $100\text{ ms}$ release, soft knee |
| **8. Limiter** | True Peak Lookahead Brickwall | Guaranteed prevention of digital overs and DAC clipping | Threshold: $-1.0\text{ dBFS}$ |

---

## Repository Structure

```
micforge/
├── src/micforge/
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── stats.py             # Audio analysis, SNR calculation, and diagnostic heuristics
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── backend.py           # Sounddevice / PortAudio interface bindings
│   │   ├── cues.py              # Procedural sine-sweep feedback chimes (NumPy)
│   │   ├── devices.py           # Audio device discovery, host API query, and pairing
│   │   ├── engine.py            # Stream lifecycle, active sink gating, and drift sync
│   │   ├── fileio.py            # High-throughput 32-bit WAV serialization
│   │   ├── metrics.py           # Thread-safe RMS, peak hold, clipping, and telemetry
│   │   ├── recorder.py          # Device calibration clip recording
│   │   └── ring.py              # Lock-free circular jitter buffer implementation
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── calibrate.py         # 3-second noise floor sampling and threshold calculation
│   │   ├── common.py            # Shared CLI helpers and formatting utilities
│   │   ├── devices_cmd.py       # Audio device listing and inspection command
│   │   ├── doctor.py            # System configuration and driver diagnostic checks
│   │   ├── files.py             # Offline WAV processing and audio analysis commands
│   │   ├── live.py              # Real-time dashboard and background tray runner
│   │   ├── test_mic.py          # Interactive loopback and mic comparison testing
│   │   └── wizard.py            # First-run setup and configuration wizard
│   ├── dsp/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract Processor base class
│   │   ├── biquad.py            # Direct Form II Transposed biquadratic filters
│   │   ├── chain.py             # Composite DSP rack pipeline dispatcher
│   │   ├── cofilite.py          # CoFi-Lite neural feature extractor and ONNX inference
│   │   ├── compressor.py        # Feed-forward compressor and lookahead peak limiter
│   │   ├── deesser.py           # Split-band dynamic sibilance attenuator
│   │   ├── denoise.py           # Classical Spectral Wiener denoiser with minimum statistics
│   │   ├── export_cofilite.py   # Checkpoint converter (PyTorch -> standalone ONNX)
│   │   ├── gate.py              # Downward expander / noise gate with hysteresis
│   │   ├── leveler.py           # Optical RMS automatic leveling amplifier
│   │   └── util.py              # DSP mathematical utilities (dB conversion, curves)
│   ├── models/
│   │   └── cofilite.onnx        # Standalone pre-compiled neural model binary (<600 KB)
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── dashboard.py         # Full-screen Rich terminal studio dashboard
│   │   ├── hotkeys.py           # Low-level Windows global keyboard hook manager
│   │   ├── keys.py              # Cross-platform non-blocking console key reader
│   │   ├── report.py            # Visual diagnostic report formatting
│   │   ├── settings_gui.py      # Dark-themed Tkinter configuration interface
│   │   ├── theme.py             # Design tokens and color palettes
│   │   ├── tray.py              # Win32 system tray daemon and procedural icon renderer
│   │   └── widgets.py           # Terminal meters, spectrum graphs, and sparklines
│   ├── util/
│   │   ├── __init__.py
│   │   └── autostart.py         # Windows Registry HKCU boot startup integration
│   ├── __init__.py              # Package entry and version definition
│   ├── __main__.py              # Package execution wrapper
│   ├── cli.py                   # Argument parsing and command routing
│   ├── config.py                # Schema, defaults, and configuration persistence
│   ├── errors.py                # Typed exception hierarchy
│   ├── logging_setup.py         # Diagnostic logging configuration
│   ├── presets.py               # Built-in DSP preset profiles
│   └── profile_store.py         # Calibrated ambient noise profile persistence
├── tests/
│   ├── conftest.py              # Pytest fixtures and mock environments
│   ├── fake_sd.py               # Mock PortAudio sounddevice driver for CI testing
│   ├── test_cofilite.py         # Neural model and feature extraction test suite
│   ├── test_cues.py             # Audio feedback synthesis tests
│   ├── test_dsp.py              # DSP biquad, filter, and dynamic processor unit tests
│   ├── test_engine.py           # Ring buffer, audio engine, and gating tests
│   ├── test_hotkeys.py          # Global hotkey and hook integration tests
│   └── test_misc.py             # CLI, configuration, and preset validation tests
├── version notes/
│   └── v1.1.0.md                # Comprehensive release notes and changelog
├── pyproject.toml               # PEP 517/518 build metadata and dependency specification
├── requirements.txt             # Direct development dependency requirements
└── README.md
```

---

## CLI Reference

```
usage: micforge [-h] [-v] [--debug] [--model MODEL] [--tray] <command> ...

Commands:
  run             Execute live processing pipeline (mic -> DSP -> virtual cable)
  setup           Select input mic, virtual cable, and monitor headphones
  calibrate       Analyze ambient room noise to establish gate and denoiser thresholds
  doctor          Run diagnostic validation on audio drivers and host configuration
  settings        Open graphical settings for hotkeys, audio chimes, and startup
  loopback        Direct headphone monitor mode for real-time acoustic tuning
  test            Record an 8-second speech sample and inspect before/after metrics
  presets         List available voice presets or output profile JSON
  process         Process an offline WAV file through the DSP pipeline
  analyze         Analyze WAV file characteristics (RMS, noise floor, clarity)
  model           Inspect or export CoFi-Lite neural ONNX checkpoints
  devices         Display enumerated audio input and output devices
  version         Print version and environment diagnostic information
```

---

## Verification & Testing

Micforge maintains a rigorous automated test suite. Audio drivers are mocked in software, allowing complete validation on headless systems without physical sound hardware:

```bash
pip install -e .[dev]
pytest -v
```

```
============================== test session starts ==============================
collected 65 items

tests/test_cofilite.py ......                                            [  9%]
tests/test_cues.py ...                                                   [ 13%]
tests/test_dsp.py ....................................                   [ 69%]
tests/test_engine.py ...                                                 [ 73%]
tests/test_hotkeys.py ....                                               [ 80%]
tests/test_misc.py .............                                         [100%]

============================== 65 passed in ~23s ==============================
```

---

## Tech Stack & Upstream Projects

Micforge stands on the shoulders of fantastic open-source libraries, neural architectures, and audio infrastructure:

| Component | Project / Repository | Role in Micforge |
|:---|:---|:---|
| **Neural Speech Model** | [CoFi-Lite](https://github.com/Acceleration123/CoFi-Lite) | Core causal neural speech enhancement architecture and PyTorch checkpoints |
| **Inference Engine** | [ONNX Runtime](https://github.com/microsoft/onnxruntime) & [ONNX](https://github.com/onnx/onnx) | Ultra-fast SIMD float32 CPU execution of the neural speech model |
| **Audio I/O & Backend** | [sounddevice](https://github.com/spatialaudio/python-sounddevice) & [PortAudio](http://www.portaudio.com/) | Real-time, non-blocking audio stream bindings for WASAPI, ALSA, and CoreAudio |
| **Numerical & DSP Math** | [NumPy](https://github.com/numpy/numpy) & [SciPy](https://github.com/scipy/scipy) | Vectorized DSP filter calculations, ring buffer operations, and procedural audio synthesis |
| **Terminal Studio UI** | [Rich](https://github.com/Textualize/rich) | Studio ASCII meters, spectrum analyzer visualizer, and live dashboard rendering |
| **System Tray Daemon** | [pystray](https://github.com/moses-palmer/pystray) | Windows system tray taskbar icon and context menu lifecycle |
| **Dynamic Icon Drawing** | [Pillow (PIL)](https://github.com/python-pillow/Pillow) | In-memory procedural generation of dynamic 64×64 tray status icons |
| **Global Input Hooks** | [pynput](https://github.com/moses-palmer/pynput) | Low-level OS keyboard event hooks for in-game and fullscreen hotkey capture |
| **Graphical Interface** | [Tkinter](https://docs.python.org/3/library/tkinter.html) | Native Python desktop GUI toolkit powering the dark-mode settings window |
| **Virtual Audio Drivers** | [VB-Audio Cable](https://vb-audio.com/Cable/) & [BlackHole](https://github.com/ExistentialAudio/BlackHole) | Loopback routing of processed audio into Discord, OBS, and VoIP clients |
| **Test Framework** | [pytest](https://github.com/pytest-dev/pytest) | Automated test suite execution and fixture mocking |

---

## Author & Community

- **Lead Developer & Maintainer:** **[Parth Sadaria](https://github.com/Parthsadaria)**
- **Neural Network Research:** Upstream model architecture design by the [CoFi-Lite Authors](https://github.com/Acceleration123/CoFi-Lite).
- **Contributions & Feedback:** Bug reports, feature requests, and pull requests are warmly welcomed via the [GitHub Repository](https://github.com/Parthsadaria/MicForge).

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
