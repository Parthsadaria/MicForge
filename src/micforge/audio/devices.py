"""Device discovery and selection, with host-API de-duplication and virtual-cable detection."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..errors import DeviceError
from .backend import get_sd

HOSTAPI_PRIORITY = [
    "Windows WASAPI",
    "Core Audio",
    "JACK Audio Connection Kit",
    "ALSA",
    "Windows DirectSound",
    "MME",
    "OSS",
    "Windows WDM-KS",
]

# Playback-side names of virtual cables, best first. Discord then picks the matching *recording* end.
VIRTUAL_OUTPUT_PATTERNS = [
    r"cable input",
    r"vb-audio",
    r"voicemeeter (aux )?input",
    r"blackhole",
    r"soundflower",
    r"loopback",
    r"virtual",
    r"null",
]


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    max_input: int
    max_output: int
    default_rate: float
    is_default_input: bool
    is_default_output: bool

    @property
    def is_virtual(self) -> bool:
        low = self.name.lower()
        return any(re.search(p, low) for p in VIRTUAL_OUTPUT_PATTERNS)

    @property
    def is_wasapi(self) -> bool:
        return "WASAPI" in self.hostapi


def _priority(api: str) -> int:
    return HOSTAPI_PRIORITY.index(api) if api in HOSTAPI_PRIORITY else len(HOSTAPI_PRIORITY)


def list_devices(all_apis: bool = False) -> list[DeviceInfo]:
    """All devices; by default only those on the best host API available (hides MME/DirectSound dupes)."""
    sd = get_sd()
    apis = sd.query_hostapis()
    default_in, default_out = sd.default.device
    infos = []
    for i, d in enumerate(sd.query_devices()):
        infos.append(
            DeviceInfo(
                index=i,
                name=str(d["name"]),
                hostapi=str(apis[d["hostapi"]]["name"]),
                max_input=int(d["max_input_channels"]),
                max_output=int(d["max_output_channels"]),
                default_rate=float(d["default_samplerate"]),
                is_default_input=(i == default_in),
                is_default_output=(i == default_out),
            )
        )
    if all_apis or not infos:
        return infos
    best = min((_priority(d.hostapi) for d in infos), default=0)
    # keep the best API, but never hide the system default device
    return [d for d in infos if _priority(d.hostapi) == best or d.is_default_input or d.is_default_output]


def _usable(d: DeviceInfo, kind: str) -> bool:
    return (d.max_input if kind == "input" else d.max_output) > 0


def find_device(spec: Any, kind: str, devices: list[DeviceInfo] | None = None) -> DeviceInfo:
    """Resolve an index / name substring / None (=default) to a device."""
    if kind not in ("input", "output"):
        raise ValueError(kind)
    pool = devices if devices is not None else list_devices()
    candidates = [d for d in pool if _usable(d, kind)]
    if not candidates:
        raise DeviceError(f"No {kind} devices found.", hint="Plug one in, or run `micforge doctor`.")

    if spec in (None, ""):
        default = [d for d in candidates if (d.is_default_input if kind == "input" else d.is_default_output)]
        return (default or candidates)[0]

    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        idx = int(spec)
        for d in get_all(devices):
            if d.index == idx and _usable(d, kind):
                return d
        raise DeviceError(f"Device index {idx} is not a valid {kind} device.", hint="See `micforge devices`.")

    needle = str(spec).lower()
    matches = [d for d in get_all(devices) if needle in d.name.lower() and _usable(d, kind)]
    if not matches:
        raise DeviceError(f"No {kind} device matches {spec!r}.", hint="See `micforge devices`.")
    matches.sort(key=lambda d: (_priority(d.hostapi), d.index))
    return matches[0]


def get_all(devices: list[DeviceInfo] | None) -> list[DeviceInfo]:
    return devices if devices is not None else list_devices(all_apis=True)


def detect_virtual_output(devices: list[DeviceInfo] | None = None) -> DeviceInfo | None:
    pool = [d for d in get_all(devices) if d.max_output > 0]
    for pat in VIRTUAL_OUTPUT_PATTERNS:
        hits = [d for d in pool if re.search(pat, d.name.lower())]
        if hits:
            hits.sort(key=lambda d: (_priority(d.hostapi), d.index))
            return hits[0]
    return None


def extra_settings(device: DeviceInfo, kind: str):
    """WASAPI shared-mode auto-conversion lets us run at 48 kHz regardless of the mixer format."""
    sd = get_sd()
    if device.is_wasapi and hasattr(sd, "WasapiSettings"):
        try:
            return sd.WasapiSettings(auto_convert=True)
        except TypeError:
            return None
    return None
