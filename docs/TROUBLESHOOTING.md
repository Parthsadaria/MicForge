# Micforge Troubleshooting & Diagnostic Guide

Micforge is designed to be self-healing, low-latency, and safe. However, operating real-time virtual audio devices across different operating systems, sample rates, and applications can sometimes run into configuration snags.

Use this guide to diagnose and resolve issues in seconds.

---

## ⚡ Quick Diagnostics

Before tweaking individual settings, run these commands to inspect your pipeline:

```bash
# 1. Run the system and performance doctor
micforge doctor

# 2. Measure your actual end-to-end latency with acoustic breakdown
micforge latency

# 3. Export a complete, redacted diagnostic bundle (.zip) for bug reports
micforge doctor --export
```

You can also click **📦 Export Diagnostics (.zip)** anytime from the **🎙️ Routing & System** tab in the Settings GUI (`micforge settings`).

---

## 1. No Audio in Discord / OBS / Zoom (Virtual Cable Silent)

### Symptoms
- Micforge dashboard shows active input levels (`IN: -24 dB`), but nobody in Discord can hear you.
- Virtual cable output stays at `-inf dB` or Discord voice status shows "No Route" / silent circle.

### Root Cause
Discord or OBS is either listening to the wrong virtual cable endpoint, or the operating system has muted or de-prioritized the virtual device.

### Fix by OS

#### Windows
1. **Discord Input Device:** In Discord → *User Settings* → *Voice & Video* → *Input Device*, select **CABLE Output (VB-Audio Virtual Cable)**.
   > **Note:** The names can be counter-intuitive! Micforge outputs audio to `CABLE Input`, and Discord listens on `CABLE Output`.
2. **Windows Sound Levels:** Open Windows Sound Settings (`mmsys.cpl`):
   - Under the **Recording** tab, find *CABLE Output*.
   - Right-click → **Properties** → **Levels** tab: ensure the volume is set to **100%** and not muted.
3. **Listen to this Device Test:**
   - In *CABLE Output* Properties → **Listen** tab: check "Listen to this device" and choose your headphones temporarily to verify sound is passing through. Uncheck when done.
4. **App Volume & Device Preferences:** Ensure Windows hasn't set Discord's input volume to 0 in Windows 11 *Volume Mixer*.

#### macOS
1. **BlackHole Routing:** Ensure Micforge is routing to `BlackHole 2ch` and Discord has `BlackHole 2ch` selected as its Input Device.
2. **Microphone Permissions:** Go to *System Settings* → *Privacy & Security* → *Microphone*. Ensure your terminal / Python / Micforge is granted microphone access. If permissions were changed, restart your terminal.

#### Linux (PipeWire / PulseAudio)
1. **Null Sink Setup:** Ensure the virtual null sink exists:
   ```bash
   pactl load-module module-null-sink sink_name=micforge sink_properties=device.description=Micforge_Virtual_Mic
   ```
2. **Discord Selection:** In Discord, choose `Monitor of Micforge_Virtual_Mic` as your recording input device.
3. **Pavucontrol Check:** Open `pavucontrol` → *Recording* tab. Ensure the stream from Micforge is mapped to the `micforge` sink.

---

## 2. Robotic, Metallic, or "Underwater" Voice

### Symptoms
- Voice sounds phasey, unnatural, or heavily muffled.
- Sounds like two AI noise suppressors are fighting each other.

### Fix
1. **Disable Discord's Built-in DSP (Critical!):**
   Micforge applies studio-grade AI neural speech enhancement (`CoFi-Lite`), linear-phase high-pass filtering, de-essing, and downward expansion. If Discord also runs Krisp or Standard noise cancellation, the two engines will double-process your audio:
   - In Discord: *Settings* → *Voice & Video*:
     - **Noise Suppression:** Set to **None** (turn off Krisp and Standard).
     - **Echo Cancellation:** **OFF**.
     - **Noise Reduction:** **OFF**.
     - **Automatic Gain Control (AGC):** **OFF**.
2. **Sample Rate Mismatch:**
   - Check `micforge doctor`. If your physical microphone is set to 44,100 Hz and the virtual cable is set to 48,000 Hz, Windows does software resampling which introduces phase smearing and comb filtering.
   - Open Windows Sound Control Panel (`mmsys.cpl`), navigate to both your microphone and `CABLE Input`, open **Properties** → **Advanced**, and set both to **2 channel, 24-bit, 48000 Hz (Studio Quality)** (or 16-bit 48000 Hz).
3. **Noise Removal Sensitivity:**
   - In `micforge settings`, switch to **Simple Mode** and decrease the **Noise Removal** knob from `80` to `40–60`.
   - In environments with fan or AC noise, run:
     ```bash
     micforge profile switch fan_on
     ```

---

## 3. Echo or Game Sound Bleed

### Symptoms
- Teammates hear their own voices echoing back through your microphone.
- Loud game sounds or music leak into the mic stream.

### Fix
1. **Headphones vs. Open Speakers:**
   - Acoustic echo cancellation (AEC) works best with headphones. If you are using desktop speakers near a desk mic, acoustic reflections can overwhelm basic filters.
2. **Enable Far-End AEC Reference:**
   - Micforge includes an adaptive normalized LMS acoustic echo canceller.
   - Start Micforge with monitor loopback or press `[e]` in the live HUD to toggle AEC.
   - In `micforge settings`, configure your system output / headphones as the far-end reference stream.
3. **Gate Threshold / Room Cleanup:**
   - In `micforge settings`, increase **Room Cleanup** (which increases De-Verb suppression and high-pass rumble cutoff).
   - Lower the noise gate sensitivity so background game audio below your voice level stays silenced.

---

## 4. High Latency / Audio Delay

### Symptoms
- Noticeable lag between speaking and hearing sidetone.
- Latency test reports >80 ms total delay.

### Diagnostic Command
Run the built-in end-to-end latency test:
```bash
micforge latency
```
This sends a 60 ms band-limited chirp through the hardware loopback and measures exact cross-correlation roundtrip delay, giving an itemized breakdown.

### Tuning Actions
1. **Reduce Block Size:**
   - Default block size is 256 samples (~5.3 ms at 48 kHz).
   - If your CPU is fast, launch with `--block 128`:
     ```bash
     micforge run --block 128
     ```
2. **WASAPI Shared vs Exclusive Mode (Windows):**
   - In Windows, ensure the mic properties have exclusive mode allowed, or run with lower host buffer requests.
3. **Jitter Cushion:**
   - Micforge uses a calibrated ring buffer cushion (default 3 blocks = ~16 ms) to prevent buffer underruns.
   - If your system has zero DPC latency spikes, you can run with `--jitter 2` to shave another 5.3 ms off your pipeline.

---

## 5. Glitches, Clicks, or Audio Dropouts After PC Sleep / Wake

### Symptoms
- Audio sounds clean initially, but develops crackles or clicks after the computer resumes from sleep or hibernate.
- The dashboard shows rapidly incrementing `glitches` or `clock fixes`.

### What Micforge Does Automatically
- **Phase-Calibrated Clock Alignment:** Micforge v1.3.0 automatically measures clock drift between unsynchronized input/output DACs. It locks target phase after 100 blocks and applies sub-millisecond spline micro-resampling with deadband protection (cooldown $\ge 150$ blocks).
- **Device Watcher & Auto-Recovery:** If a USB microphone or headset disconnects, `DeviceWatcher` detects the topology change and attempts up to 5 exponential backoff reconnections. If the device cannot be recovered, it falls back to a safe bypass mode so you are never left silently muted.

### External Fixes
1. **Disable USB Selective Suspend (Windows):**
   - Windows Power Plan → *Change advanced power settings* → *USB settings* → *USB selective suspend setting* → **Disabled**.
   - In Device Manager → *Universal Serial Bus controllers* → USB Root Hub → *Power Management* → Uncheck *"Allow the computer to turn off this device to save power"*.
2. **Exclusive Mode Conflicts:**
   - In Windows Sound Properties for your mic, uncheck *"Allow applications to take exclusive control of this device"*. This prevents games or VoIP apps from hijacking the mic handle upon waking up.

---

## 6. Noise Floor Drift (Fans, AC, Mechanical Keyboard)

### Symptoms
- Micforge gated properly during initial calibration, but after a gaming session with fans spinning up, noise cuts in and out.

### Fix
- **Continuous Adaptive Learning:** Micforge v1.3.0 features voice-aware continuous background learning. When you are silent, it slowly updates its noise spectrum estimate ($\le 1.0$ dB/s) without ever learning during speech.
- **Select an Environment Profile:**
  ```bash
  micforge profile list
  micforge profile switch fan_on
  ```
- **Quick Re-Learn:**
  - Press the **Relearn Noise** button in `micforge settings`, or trigger hotkey / CLI:
    ```bash
    micforge profile relearn
    ```

---

## 7. Exporting Diagnostics for Support

If an issue persists:
1. Run:
   ```bash
   micforge doctor --export
   ```
2. A `.zip` archive will be created in your current directory (e.g. `micforge_diag_YYYYMMDD_HHMMSS.zip`).
3. The archive contains:
   - `report.json` – Detailed system and audio diagnostics.
   - `doctor_summary.txt` – Plain-English health status and root-cause hints.
   - `devices.txt` – Full PortAudio device topology.
   - `config_redacted.json` – Active configuration with secrets and home paths completely redacted.
   - `recent_logs.txt` – Recent operational and error traces.
4. Attach this `.zip` file when opening an issue on GitHub!

