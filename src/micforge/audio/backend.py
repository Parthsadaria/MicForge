"""Lazy access to sounddevice so `--help`, tests and offline commands work without PortAudio."""
from __future__ import annotations

from types import ModuleType

from ..errors import AudioBackendError

_sd: ModuleType | None = None


def get_sd() -> ModuleType:
    global _sd
    if _sd is None:
        import os
        if os.environ.get("MICFORGE_TEST_FAKE_SD") == "1":
            try:
                import sys
                from pathlib import Path
                tests_dir = str(Path(__file__).resolve().parents[3] / "tests")
                if tests_dir not in sys.path:
                    sys.path.insert(0, tests_dir)
                import importlib
                fake_mod = importlib.import_module("fake_sd")
                _sd = getattr(fake_mod, "make_fake_sd")()
                return _sd
            except Exception:
                pass
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
