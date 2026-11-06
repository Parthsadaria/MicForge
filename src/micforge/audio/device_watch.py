"""Hot-plug detection and auto-recovery for audio devices."""
from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .devices import DeviceInfo, list_devices

log = logging.getLogger("micforge.device_watch")


class DeviceEventType(str, enum.Enum):
    DEVICE_ADDED = "device_added"
    DEVICE_REMOVED = "device_removed"
    DEFAULT_CHANGED = "default_changed"
    STREAM_ERROR = "stream_error"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_SUCCEEDED = "recovery_succeeded"
    RECOVERY_FAILED = "recovery_failed"


@dataclass
class DeviceWatchEvent:
    event_type: DeviceEventType
    message: str
    device_name: str | None = None
    device_index: int | None = None
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class DeviceWatcher:
    """Monitors audio device topology changes and initiates auto-recovery on disconnects."""

    def __init__(
        self,
        poll_interval_s: float = 1.5,
        max_retries: int = 5,
        backoff_base_s: float = 0.5,
        on_recovery_attempt: Callable[[str, int | str | None], bool] | None = None,
        on_fallback: Callable[[], None] | None = None,
    ) -> None:
        self.poll_interval = poll_interval_s
        self.max_retries = max_retries
        self.backoff_base = backoff_base_s
        self.on_recovery_attempt = on_recovery_attempt
        self.on_fallback = on_fallback

        self._listeners: list[Callable[[DeviceWatchEvent], None]] = []
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

        # Cached device state
        self._cached_inputs: set[str] = set()
        self._cached_outputs: set[str] = set()
        self._cached_default_in: int | None = None
        self._cached_default_out: int | None = None
        self._recovering = False

        self._refresh_cached_state()

    @property
    def is_recovering(self) -> bool:
        with self._lock:
            return self._recovering

    def add_listener(self, cb: Callable[[DeviceWatchEvent], None]) -> None:
        with self._lock:
            if cb not in self._listeners:
                self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[DeviceWatchEvent], None]) -> None:
        with self._lock:
            if cb in self._listeners:
                self._listeners.remove(cb)

    def _emit(self, event_type: DeviceEventType, message: str, dev_name: str | None = None, dev_idx: int | None = None) -> DeviceWatchEvent:
        evt = DeviceWatchEvent(
            event_type=event_type,
            message=message,
            device_name=dev_name,
            device_index=dev_idx,
        )
        log.info("DeviceWatch event [%s]: %s (dev=%s)", event_type.value, message, dev_name or "none")
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(evt)
            except Exception as exc:
                log.debug("Listener callback exception: %s", exc)
        return evt

    def _refresh_cached_state(self) -> list[DeviceInfo]:
        try:
            devs = list_devices()
            self._cached_inputs = {d.name for d in devs if d.max_input > 0}
            self._cached_outputs = {d.name for d in devs if d.max_output > 0}
            def_in = next((d.index for d in devs if d.is_default_input), None)
            def_out = next((d.index for d in devs if d.is_default_output), None)
            self._cached_default_in = def_in
            self._cached_default_out = def_out
            return devs
        except Exception as exc:
            log.debug("Device query failure in watcher: %s", exc)
            return []

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="micforge-dev-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def trigger_stream_error(self, stream_kind: str, device_name: str | None = None) -> None:
        """Called by audio engine if a stream callback or write raises a device error."""
        self._emit(
            DeviceEventType.STREAM_ERROR,
            f"Audio stream error on {stream_kind} stream ({device_name or 'unknown device'})",
            dev_name=device_name,
        )
        self.schedule_recovery(stream_kind, device_name)

    def schedule_recovery(self, kind: str, target_device: int | str | None = None) -> None:
        """Asynchronously initiate bounded retry recovery with backoff."""
        if self._recovering:
            return
        threading.Thread(target=self._perform_recovery, args=(kind, target_device), daemon=True).start()

    def _perform_recovery(self, kind: str, target_device: int | str | None) -> bool:
        with self._lock:
            if self._recovering:
                return False
            self._recovering = True

        self._emit(DeviceEventType.RECOVERY_STARTED, f"Starting auto-recovery for {kind}...", dev_name=str(target_device))

        recovered = False
        for attempt in range(1, self.max_retries + 1):
            delay = self.backoff_base * (1.6 ** (attempt - 1))
            log.info("Recovery attempt %d/%d in %.2fs...", attempt, self.max_retries, delay)
            time.sleep(delay)

            try:
                # Refresh hardware list
                self._refresh_cached_state()
                if self.on_recovery_attempt is not None:
                    success = self.on_recovery_attempt(kind, target_device)
                    if success:
                        self._emit(
                            DeviceEventType.RECOVERY_SUCCEEDED,
                            f"Successfully recovered {kind} stream on attempt {attempt}!",
                            dev_name=str(target_device),
                        )
                        recovered = True
                        break
            except Exception as exc:
                log.warning("Recovery attempt %d failed: %s", attempt, exc)

        if not recovered:
            self._emit(
                DeviceEventType.RECOVERY_FAILED,
                f"Auto-recovery failed after {self.max_retries} attempts. Engaging safe passthrough fallback.",
                dev_name=str(target_device),
            )
            if self.on_fallback is not None:
                try:
                    self.on_fallback()
                except Exception as exc:
                    log.error("Fallback handler error: %s", exc)

        with self._lock:
            self._recovering = False
        return recovered

    def check_devices(self, devs: list[DeviceInfo] | None = None) -> list[DeviceWatchEvent]:
        """Check given (or current) device list against cached state, emitting events on changes."""
        if devs is None:
            devs = list_devices()
        curr_inputs = {d.name for d in devs if d.max_input > 0}
        curr_outputs = {d.name for d in devs if d.max_output > 0}
        curr_def_in = next((d.index for d in devs if d.is_default_input), None)
        curr_def_out = next((d.index for d in devs if d.is_default_output), None)

        events: list[DeviceWatchEvent] = []
        removed_in = self._cached_inputs - curr_inputs
        for name in removed_in:
            events.append(self._emit(DeviceEventType.DEVICE_REMOVED, f"Input device disconnected: {name}", dev_name=name))

        removed_out = self._cached_outputs - curr_outputs
        for name in removed_out:
            events.append(self._emit(DeviceEventType.DEVICE_REMOVED, f"Output device disconnected: {name}", dev_name=name))

        added_in = curr_inputs - self._cached_inputs
        for name in added_in:
            events.append(self._emit(DeviceEventType.DEVICE_ADDED, f"New input device detected: {name}", dev_name=name))

        added_out = curr_outputs - self._cached_outputs
        for name in added_out:
            events.append(self._emit(DeviceEventType.DEVICE_ADDED, f"New output device detected: {name}", dev_name=name))

        if curr_def_in != self._cached_default_in or curr_def_out != self._cached_default_out:
            events.append(self._emit(
                DeviceEventType.DEFAULT_CHANGED,
                f"Default audio endpoint changed (in: {curr_def_in}, out: {curr_def_out})",
            ))

        self._cached_inputs = curr_inputs
        self._cached_outputs = curr_outputs
        self._cached_default_in = curr_def_in
        self._cached_default_out = curr_def_out
        return events

    def _run_loop(self) -> None:
        while self._running:
            time.sleep(self.poll_interval)
            if not self._running or self._recovering:
                continue

            try:
                self.check_devices()
            except Exception as exc:
                log.debug("Device watcher check error: %s", exc)
