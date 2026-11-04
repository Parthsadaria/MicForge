from micforge.config import AppConfig, HotkeyConfig
from micforge.ui.hotkeys import normalize_hotkey_for_pynput
from micforge.ui.tray import generate_tray_icon
from micforge.util.autostart import is_autostart_enabled


def test_normalize_hotkey():
    assert normalize_hotkey_for_pynput("ctrl+shift+m") == "<ctrl>+<shift>+m"
    assert normalize_hotkey_for_pynput("ctrl + alt + b") == "<ctrl>+<alt>+b"
    assert normalize_hotkey_for_pynput("control+shift+x") == "<ctrl>+<shift>+x"
    assert normalize_hotkey_for_pynput("f12") == "<f12>"
    assert normalize_hotkey_for_pynput("") == ""


def test_hotkey_config_in_app_config():
    app = AppConfig()
    assert app.hotkeys.toggle_mute == "ctrl+shift+m"
    assert app.hotkeys.toggle_bypass == "ctrl+shift+b"
    assert app.hotkeys.play_cues is True
    assert app.hotkeys.cue_volume == 0.5


def test_tray_icon_rendering():
    im_active = generate_tray_icon("active")
    im_muted = generate_tray_icon("muted")
    im_bypass = generate_tray_icon("bypass")

    assert im_active.size == (64, 64)
    assert im_muted.size == (64, 64)
    assert im_bypass.size == (64, 64)


def test_autostart_safe_call():
    # Must not raise an uncaught exception
    res = is_autostart_enabled()
    assert isinstance(res, bool)

