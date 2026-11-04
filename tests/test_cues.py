import numpy as np
from micforge.audio.cues import CuePlayer


def test_cue_generation():
    player = CuePlayer(sample_rate=48000, volume=0.5, enabled=True)
    assert "mute" in player._cache
    assert "unmute" in player._cache
    assert "toggle" in player._cache

    blip = player._cache["mute"]
    chirp = player._cache["unmute"]
    toggle = player._cache["toggle"]

    assert len(blip) > 0
    assert len(chirp) > 0
    assert len(toggle) > 0
    assert np.max(np.abs(blip)) > 0.1
    assert np.max(np.abs(chirp)) > 0.1


def test_cue_mixing():
    player = CuePlayer(sample_rate=48000, volume=0.5, enabled=True)
    block = np.zeros(480, dtype=np.float32)

    # Before playing, mixing returns clean block
    out = player.mix_into(block)
    assert np.all(out == 0)

    # After queuing a chime, mixing introduces audio
    player.play("unmute")
    out = player.mix_into(block)
    assert np.any(out != 0)


def test_cue_disabled_or_zero_volume():
    player = CuePlayer(sample_rate=48000, volume=0.0, enabled=True)
    player.play("mute")
    block = np.zeros(480, dtype=np.float32)
    assert np.all(player.mix_into(block) == 0)

    player2 = CuePlayer(sample_rate=48000, volume=0.5, enabled=False)
    player2.play("mute")
    assert np.all(player2.mix_into(block) == 0)

