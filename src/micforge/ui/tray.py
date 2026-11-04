"""System Tray Daemon for Micforge.

Provides a persistent icon in the Windows taskbar with:
* Live dynamic icon (cyan when active, gray with red slash when muted, amber when bypassed)
* Instant mute / bypass controls
* Presets and CoFi-Lite / Spectral model switcher
* Quick access to Settings & Hotkeys GUI
* Windows autostart toggle
* Hide / Restore console window
"""
from __future__ import annotations

import ctypes
import logging
import os
import threading
from typing import Any, Callable

from PIL import Image, ImageDraw

from .. import __version__
from ..audio.engine import AudioEngine
from ..config import AppConfig
from ..presets import list_presets
from ..util.autostart import is_autostart_enabled, set_autostart
from .settings_gui import open_settings_window

log = logging.getLogger("micforge.tray")


def _safe_patch_pystray_win32() -> None:
    """Patch pystray's ChangeWindowMessageFilterEx to gracefully handle Windows UIPI security restrictions."""
    if os.name != "nt":
        return
    try:
        import pystray._util.win32 as w32util

        orig = getattr(w32util, "ChangeWindowMessageFilterEx", None)
        if orig is not None and not getattr(orig, "_micforge_patched", False):

            def _safe_filter(*args, **kwargs):
                try:
                    return orig(*args, **kwargs)
                except Exception:
                    return True

            _safe_filter._micforge_patched = True  # type: ignore[attr-defined]
            w32util.ChangeWindowMessageFilterEx = _safe_filter
    except Exception as exc:  # noqa: BLE001
        log.debug("pystray win32 patch failed: %s", exc)


def generate_tray_icon(state: str = "active") -> Image.Image:
    """Procedurally draw a crisp 64x64 RGBA microphone icon."""
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    if state == "muted":
        color = (150, 150, 155)
    elif state == "bypass":
        color = (250, 204, 21)  # Amber
    else:
        color = (6, 182, 212)  # Cyan

    # 1. Microphone capsule
    d.rounded_rectangle([24, 12, 40, 36], radius=7, fill=color)

    # 2. Cradle arc
    d.arc([16, 20, 48, 44], start=0, end=180, fill=color, width=3)

    # 3. Base stem and foot
    d.line([32, 44, 32, 54], fill=color, width=3)
    d.line([20, 54, 44, 54], fill=color, width=3)

    # 4. If muted, add prominent red diagonal slash and indicator dot
    if state == "muted":
        d.line([14, 14, 50, 50], fill=(239, 68, 68), width=5)
        d.ellipse([44, 10, 56, 22], fill=(239, 68, 68))

    return im


class SystemTrayDaemon:
    """Manages the Windows system tray lifecycle, icon updates, and menu actions."""

    def __init__(
        self,
        app: AppConfig,
        engine: AudioEngine,
        on_exit: Callable[[], None],
        on_switch_preset: Callable[[str], None] | None = None,
        on_switch_model: Callable[[str], None] | None = None,
    ) -> None:
        self.app = app
        self.engine = engine
        self.on_exit = on_exit
        self.on_switch_preset = on_switch_preset
        self.on_switch_model = on_switch_model

        self._icon: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._console_visible = True

    @property
    def current_state(self) -> str:
        if self.engine.muted:
            return "muted"
        if self.engine.bypass:
            return "bypass"
        return "active"

    def update_icon(self) -> None:
        """Refresh tray icon image to reflect current mute/bypass state."""
        if self._icon is not None:
            try:
                self._icon.icon = generate_tray_icon(self.current_state)
                self._icon.title = f"Micforge v{__version__} ({self.current_state.capitalize()})"
            except Exception as exc:  # noqa: BLE001
                log.debug("Tray icon refresh failed: %s", exc)

    def start(self) -> None:
        """Start the system tray icon in a dedicated daemon thread."""
        try:
            import pystray
        except ImportError:
            log.warning("pystray not installed; running without system tray.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_tray, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:  # noqa: BLE001
                pass
            self._icon = None

    def toggle_console(self) -> None:
        """Toggle console window visibility on Windows."""
        if os.name != "nt":
            return
        try:
            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32
            hwnd = kernel32.GetConsoleWindow()
            if hwnd:
                self._console_visible = not self._console_visible
                # SW_HIDE = 0, SW_SHOW = 5
                user32.ShowWindow(hwnd, 5 if self._console_visible else 0)
        except Exception as exc:  # noqa: BLE001
            log.debug("Console toggle failed: %s", exc)

    def _build_menu(self):
        import pystray

        # Preset submenu
        preset_items = []
        for p in list_presets():
            p_name = p.name

            def _make_preset_cb(name: str):
                return lambda item: self._on_preset_click(name)

            def _make_preset_checked(name: str):
                return lambda item: self.app.preset == name

            preset_items.append(
                pystray.MenuItem(
                    p_name,
                    _make_preset_cb(p_name),
                    checked=_make_preset_checked(p_name),
                    radio=True,
                )
            )

        # Model submenu
        def _set_cofilite(item):
            if self.on_switch_model:
                self.on_switch_model("cofilite")

        def _set_spectral(item):
            if self.on_switch_model:
                self.on_switch_model("spectral")

        def _is_cofilite(item):
            dn = self.engine.chain.stage("denoise")
            return getattr(dn, "name", "") == "denoise" and "cofilite" in getattr(dn, "model_name", "").lower()

        def _is_spectral(item):
            dn = self.engine.chain.stage("denoise")
            return getattr(dn, "name", "") == "denoise" and "cofilite" not in getattr(dn, "model_name", "").lower()

        model_items = [
            pystray.MenuItem("CoFi-Lite Neural (High Quality, <70ms)", _set_cofilite, checked=_is_cofilite, radio=True),
            pystray.MenuItem("Spectral Wiener (Ultra-Low Latency, <48ms)", _set_spectral, checked=_is_spectral, radio=True),
        ]

        # Top level menu
        menu = pystray.Menu(
            pystray.MenuItem(
                lambda item: "✔ Muted" if self.engine.muted else "Mute Mic",
                self._on_toggle_mute,
                checked=lambda item: self.engine.muted,
            ),
            pystray.MenuItem(
                lambda item: "✔ Bypass On" if self.engine.bypass else "Bypass DSP",
                self._on_toggle_bypass,
                checked=lambda item: self.engine.bypass,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Presets", pystray.Menu(*preset_items)),
            pystray.MenuItem("Noise Suppression Model", pystray.Menu(*model_items)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings & Hotkeys...", lambda item: open_settings_window(self.app, self._on_settings_applied)),
            pystray.MenuItem(
                "Start on Windows Boot",
                self._on_toggle_autostart,
                checked=lambda item: is_autostart_enabled(),
            ),
            pystray.MenuItem(
                lambda item: "Hide Dashboard" if self._console_visible else "Show Dashboard",
                lambda item: self.toggle_console(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit Micforge", lambda item: self.on_exit()),
        )
        return menu

    def _run_tray(self) -> None:
        _safe_patch_pystray_win32()
        import pystray

        initial_image = generate_tray_icon(self.current_state)
        self._icon = pystray.Icon(
            "micforge",
            initial_image,
            title=f"Micforge v{__version__} ({self.current_state.capitalize()})",
            menu=self._build_menu(),
        )
        try:
            self._icon.run()
        except Exception as exc:  # noqa: BLE001
            log.warning("System tray run exited: %s", exc)

    def _on_toggle_mute(self, item=None) -> None:
        self.engine.toggle_mute()
        self.update_icon()

    def _on_toggle_bypass(self, item=None) -> None:
        self.engine.toggle_bypass()
        self.update_icon()

    def _on_preset_click(self, preset_name: str) -> None:
        self.app.preset = preset_name
        try:
            self.app.save()
        except Exception:  # noqa: BLE001
            pass
        if self.on_switch_preset is not None:
            self.on_switch_preset(preset_name)

    def _on_toggle_autostart(self, item=None) -> None:
        current = is_autostart_enabled()
        set_autostart(not current)

    def _on_settings_applied(self, updated_app: AppConfig) -> None:
        log.info("Settings updated via GUI")
        self.app = updated_app
        self.engine.cues.set_enabled(updated_app.hotkeys.play_cues)
        self.engine.cues.set_volume(updated_app.hotkeys.cue_volume)

