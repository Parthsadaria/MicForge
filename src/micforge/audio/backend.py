"""Lazy access to sounddevice so `--help`, tests and offline commands work without PortAudio."""
from __future__ import annotations

from types import ModuleType

from ..errors import AudioBackendError

_sd: ModuleType | None = None


def get_sd() -> ModuleType:
    global _sd
    if _sd is None:
        try:
            import sounddevice as sd  # noqa: PLC0415
        except ImportError as exc:
            raise AudioBackendError(
                "The 'sounddevice' package is not installed.", hint="pip install sounddevice"
            ) from exc
        except OSError as exc:
            raise AudioBackendError(
                f"PortAudio library not found ({exc}).",
                hint="Linux: sudo apt install libportaudio2   |   macOS: brew install portaudio",
            ) from exc
        _sd = sd
    return _sd


def set_backend(module: ModuleType | None) -> None:
    """Inject a backend (used by tests)."""
    global _sd
    _sd = module
