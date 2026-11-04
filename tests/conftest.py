import pytest

from micforge.audio import backend
from micforge.config import AppConfig
import micforge.config as cfgmod

from fake_sd import make_fake_sd


@pytest.fixture()
def fake_audio(monkeypatch):
    sd = make_fake_sd()
    backend.set_backend(sd)
    yield sd
    backend.set_backend(None)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    yield
