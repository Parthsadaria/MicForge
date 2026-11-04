"""Non-blocking single-key input for the live dashboard (POSIX + Windows)."""
from __future__ import annotations

import os
import sys
import time


class KeyReader:
    def __init__(self) -> None:
        self.enabled = False
        self._fd: int | None = None
        self._old = None

    def __enter__(self) -> "KeyReader":
        if not sys.stdin or not sys.stdin.isatty():
            return self
        if os.name == "nt":
            self.enabled = True
            return self
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # keeps Ctrl+C working
        self.enabled = True
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None and self._old is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self, timeout: float = 0.0) -> str | None:
        if not self.enabled:
            time.sleep(timeout)
            return None
        if os.name == "nt":
            import msvcrt

            end = time.monotonic() + timeout
            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):  # arrow / function key prefix
                        msvcrt.getwch()
                        return None
                    return ch
                if time.monotonic() >= end:
                    return None
                time.sleep(0.01)
        import select

        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        ch = os.read(self._fd, 1).decode(errors="ignore")  # type: ignore[arg-type]
        if ch == "\x1b":  # swallow escape sequences
            while select.select([sys.stdin], [], [], 0)[0]:
                os.read(self._fd, 1)  # type: ignore[arg-type]
            return None
        return ch
