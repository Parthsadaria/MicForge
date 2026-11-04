"""System-wide global hotkeys using non-intrusive pynput hooks.

Works while in fullscreen games, Discord, or desktop apps without requiring admin rights.
Supports Toggle Mute, Push-to-Mute, Push-to-Talk, and Toggle Bypass.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from ..config import HotkeyConfig

log = logging.getLogger("micforge.hotkeys")


def normalize_hotkey_for_pynput(s: str) -> str:
    """Convert human-friendly hotkey string (e.g. 'ctrl+shift+m') to pynput format ('<ctrl>+<shift>+m')."""
    if not s or not s.strip():
        return ""
    tokens = [t.strip().lower() for t in s.replace(" ", "").split("+") if t.strip()]
    formatted = []
    for t in tokens:
        if t in ("ctrl", "control", "lctrl", "rctrl"):
            formatted.append("<ctrl>")
        elif t in ("shift", "lshift", "rshift"):
            formatted.append("<shift>")
        elif t in ("alt", "lalt", "ralt"):
            formatted.append("<alt>")
        elif t in ("cmd", "win", "windows", "super"):
            formatted.append("<cmd>")
        elif len(t) == 1:
            formatted.append(t)
        else:
            # Special keys like f1..f12, space, etc.
            formatted.append(f"<{t}>")
    return "+".join(formatted)


class GlobalHotkeyManager:
    """Manages system-wide hotkeys and push-to-talk/push-to-mute listeners."""

    def __init__(
        self,
        config: HotkeyConfig,
        *,
        on_toggle_mute: Callable[[], None] | None = None,
        on_set_mute: Callable[[bool], None] | None = None,
        on_toggle_bypass: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.on_toggle_mute = on_toggle_mute
        self.on_set_mute = on_set_mute
        self.on_toggle_bypass = on_toggle_bypass

        self._hotkey_listener = None
        self._ptt_listener = None
        self._running = False
        self._lock = threading.Lock()
        self._ptt_down = False
        self._ptm_down = False

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._start_listeners()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_listeners()

    def update_config(self, new_config: HotkeyConfig) -> None:
        """Update hotkey configuration and rebind listeners dynamically."""
        with self._lock:
            self.config = new_config
            if self._running:
                self._stop_listeners()
                self._start_listeners()

    def _start_listeners(self) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            log.warning("pynput is not installed; global hotkeys disabled.")
            return

        # 1. GlobalHotKeys mapping for combinations (Toggle Mute, Toggle Bypass)
        hotkey_map: dict[str, Callable[[], None]] = {}
        if self.config.toggle_mute and self.on_toggle_mute:
            key_str = normalize_hotkey_for_pynput(self.config.toggle_mute)
            if key_str:
                hotkey_map[key_str] = self.on_toggle_mute

        if self.config.toggle_bypass and self.on_toggle_bypass:
            key_str = normalize_hotkey_for_pynput(self.config.toggle_bypass)
            if key_str:
                hotkey_map[key_str] = self.on_toggle_bypass

        if hotkey_map:
            try:
                self._hotkey_listener = keyboard.GlobalHotKeys(hotkey_map)
                self._hotkey_listener.daemon = True
                self._hotkey_listener.start()
                log.info("Registered global hotkeys: %s", list(hotkey_map.keys()))
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to register GlobalHotKeys: %s", exc)

        # 2. Key press/release listener for Push-to-Talk (PTT) and Push-to-Mute (PTM)
        has_ptt = bool(self.config.push_to_talk and self.config.push_to_talk.strip())
        has_ptm = bool(self.config.push_to_mute and self.config.push_to_mute.strip())

        if (has_ptt or has_ptm) and self.on_set_mute:
            ptt_target = self.config.push_to_talk.strip().lower()
            ptm_target = self.config.push_to_mute.strip().lower()

            def _get_key_name(key) -> str:
                try:
                    if hasattr(key, "char") and key.char:
                        return key.char.lower()
                    if hasattr(key, "name") and key.name:
                        return key.name.lower()
                except Exception:  # noqa: BLE001
                    pass
                return str(key).lower().replace("key.", "")

            def _on_press(key):
                kname = _get_key_name(key)
                if has_ptt and kname == ptt_target and not self._ptt_down:
                    self._ptt_down = True
                    self.on_set_mute(False)  # Unmute while pressed
                elif has_ptm and kname == ptm_target and not self._ptm_down:
                    self._ptm_down = True
                    self.on_set_mute(True)  # Mute while pressed

            def _on_release(key):
                kname = _get_key_name(key)
                if has_ptt and kname == ptt_target and self._ptt_down:
                    self._ptt_down = False
                    self.on_set_mute(True)  # Mute when released
                elif has_ptm and kname == ptm_target and self._ptm_down:
                    self._ptm_down = False
                    self.on_set_mute(False)  # Unmute when released

            try:
                self._ptt_listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
                self._ptt_listener.daemon = True
                self._ptt_listener.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to register PTT/PTM listener: %s", exc)

    def _stop_listeners(self) -> None:
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:  # noqa: BLE001
                pass
            self._hotkey_listener = None
        if self._ptt_listener is not None:
            try:
                self._ptt_listener.stop()
            except Exception:  # noqa: BLE001
                pass
            self._ptt_listener = None

