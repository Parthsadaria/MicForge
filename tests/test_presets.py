"""Tests for MicForge custom preset engine, CRUD operations, export/import, and CLI suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from micforge.cli import main
from micforge.config import AppConfig, ChainConfig
from micforge.errors import ConfigError
from micforge.presets import (
    BUILTIN,
    clone_preset,
    delete_preset,
    export_preset,
    get_preset,
    import_preset,
    list_presets,
    save_preset,
    user_preset_dir,
)


@pytest.fixture(autouse=True)
def clean_custom_presets(tmp_path, monkeypatch):
    """Isolate user preset directory to a temporary folder for tests."""
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("micforge.presets.user_preset_dir", lambda: preset_dir)
    yield preset_dir


def test_list_builtin_presets():
    presets = list_presets()
    names = [p.name for p in presets]
    assert "discord" in names
    assert "gaming" in names
    assert "broadcast" in names
    assert "studio" in names
    assert "noisy_room" in names
    assert "bypass" in names

    # Verify all built-ins are marked builtin=True
    for p in presets:
        if p.name in BUILTIN:
            assert p.builtin is True
            assert len(p.description) > 0


def test_get_broadcast_preset_features():
    cfg = get_preset("broadcast")
    assert cfg.proximity.enabled is True
    assert cfg.proximity.amount >= 0.7
    assert cfg.deverb.enabled is True
    assert cfg.denoise.enabled is True


def test_save_and_load_custom_preset():
    cfg = get_preset("discord")
    cfg.proximity.amount = 0.88
    cfg.proximity.warmth = 0.65

    path = save_preset("my_custom_voice", cfg, description="Custom testing preset")
    assert path.is_file()

    # Verify appears in list_presets()
    presets = list_presets()
    custom_entry = next((p for p in presets if p.name == "my_custom_voice"), None)
    assert custom_entry is not None
    assert custom_entry.builtin is False
    assert custom_entry.description == "Custom testing preset"

    # Verify get_preset loads the custom values
    loaded = get_preset("my_custom_voice")
    assert loaded.proximity.amount == pytest.approx(0.88)
    assert loaded.proximity.warmth == pytest.approx(0.65)


def test_save_cannot_overwrite_builtin():
    cfg = get_preset("discord")
    with pytest.raises(ConfigError, match="Cannot overwrite built-in preset"):
        save_preset("gaming", cfg)


def test_save_requires_overwrite_flag_for_existing():
    cfg = get_preset("discord")
    save_preset("test_dup", cfg)
    with pytest.raises(ConfigError, match="already exists"):
        save_preset("test_dup", cfg, overwrite=False)

    # With overwrite=True, succeeds
    cfg.proximity.amount = 0.95
    save_preset("test_dup", cfg, overwrite=True)
    assert get_preset("test_dup").proximity.amount == pytest.approx(0.95)


def test_clone_preset():
    cloned = clone_preset("broadcast", "my_broadcast_clone", description="Clone of broadcast")
    assert cloned.proximity.amount == get_preset("broadcast").proximity.amount

    loaded = get_preset("my_broadcast_clone")
    assert loaded.proximity.body_hz == 115.0


def test_delete_preset():
    cfg = get_preset("discord")
    save_preset("to_delete", cfg)
    assert "to_delete" in [p.name for p in list_presets()]

    delete_preset("to_delete")
    assert "to_delete" not in [p.name for p in list_presets()]

    # Cannot delete built-in
    with pytest.raises(ConfigError, match="Cannot delete built-in preset"):
        delete_preset("discord")


def test_export_and_import_preset(tmp_path):
    cfg = get_preset("broadcast")
    cfg.proximity.amount = 0.92
    save_preset("for_export", cfg)

    export_file = tmp_path / "shared_broadcast.json"
    exported_path = export_preset("for_export", export_file)
    assert exported_path.is_file()

    # Import with a new name
    imported_name = import_preset(export_file, name="imported_broadcast")
    assert imported_name == "imported_broadcast"

    loaded = get_preset("imported_broadcast")
    assert loaded.proximity.amount == pytest.approx(0.92)


def test_cli_preset_commands(capsys):
    # Test preset list
    res = main(["preset", "list"])
    assert res == 0

    # Test preset show
    res = main(["preset", "show", "broadcast"])
    assert res == 0

    # Test preset save
    res = main(["preset", "save", "cli_test_preset", "--from", "gaming", "--desc", "CLI test"])
    assert res == 0
    assert "cli_test_preset" in [p.name for p in list_presets()]

    # Test preset copy
    res = main(["preset", "copy", "cli_test_preset", "cli_test_copy", "--force"])
    assert res == 0
    assert "cli_test_copy" in [p.name for p in list_presets()]

    # Test preset delete
    res = main(["preset", "delete", "cli_test_preset", "--yes"])
    assert res == 0
    assert "cli_test_preset" not in [p.name for p in list_presets()]

