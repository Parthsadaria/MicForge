"""File-only logging. Console logging would corrupt the live dashboard."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import APP_NAME, config_dir


def setup_logging(debug: bool = False) -> None:
    root = logging.getLogger(APP_NAME)
    if root.handlers:
        return
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    try:
        d = config_dir()
        d.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            d / f"{APP_NAME}.log", maxBytes=512_000, backupCount=2, encoding="utf-8"
        )
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.propagate = False
