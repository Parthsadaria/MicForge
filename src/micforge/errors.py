"""Exception hierarchy. The CLI catches MicforgeError and prints it cleanly."""
from __future__ import annotations


class MicforgeError(Exception):
    """Base class for all expected, user-facing errors."""

    hint: str | None = None

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        if hint:
            self.hint = hint


class ConfigError(MicforgeError):
    """Invalid or unreadable configuration."""


class DeviceError(MicforgeError):
    """Audio device missing, unsupported, or ambiguous."""


class AudioBackendError(MicforgeError):
    """PortAudio / sounddevice unavailable or failed."""
