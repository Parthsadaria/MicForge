"""Windows Registry Run key helper for starting Micforge on boot."""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("micforge.autostart")

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_NAME = "Micforge"


def is_autostart_enabled() -> bool:
    """Return True if Micforge is registered in HKCU Run key."""
    if os.name != "nt":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_READ) as key:
            val, _ = winreg.QueryValueEx(key, APP_NAME)
            return bool(val)
    except OSError:
        return False


def set_autostart(enabled: bool) -> bool:
    """Enable or disable Micforge auto-start on Windows user login."""
    if os.name != "nt":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            if enabled:
                # Use pythonw if running via python interpreter to avoid opening a blank console
                py_exe = sys.executable
                if "python.exe" in py_exe.lower():
                    pyw = py_exe.lower().replace("python.exe", "pythonw.exe")
                    if os.path.exists(pyw):
                        py_exe = pyw
                cmd = f'"{py_exe}" -m micforge.cli run --tray'
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)
                log.info("Registered Micforge autostart command: %s", cmd)
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                    log.info("Removed Micforge autostart command")
                except FileNotFoundError:
                    pass
        return True
    except OSError as exc:
        log.warning("Could not update Windows registry autostart key: %s", exc)
        return False

