import pytest

from micforge.audio import backend
from micforge.config import AppConfig
import micforge.config as cfgmod

import sys
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tests.fake_sd import make_fake_sd


@pytest.fixture()
def fake_audio(monkeypatch):
    monkeypatch.setenv("MICFORGE_TEST_FAKE_SD", "1")
    sd = make_fake_sd()
    backend.set_backend(sd)
    yield sd
    backend.set_backend(None)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    yield
