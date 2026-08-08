"""Verify UDP payload contract matches Unity expectations."""

from vision_server.udp import default_payload


def test_default_payload_has_unity_keys():
    data = default_payload()
    expected_keys = {
        "head_yaw",
        "head_pitch",
        "head_pitch_raw",
        "pitch_cal_status",
        "pitch_cal_neutral",
        "pitch_calibrated",
        "tilt_left",
        "tilt_right",
        "leftFist",
        "leftOpenPalm",
        "leftIndexUp",
        "leftPeace",
        "rightFist",
        "rightOpenPalm",
        "rightIndexUp",
        "rightPeace",
        "watchTap",
        "watchTapDistance",
        "fistRotX",
        "fistRotY",
        "fistRotZ",
        "palmX",
        "palmY",
        "indexTipX",
        "indexTipY",
        "hands",
        "lstm_gesture",
        "player_locked",
        "lock_id",
        "lock_status",
        "puzzle_active",
        "puzzle_gate_source",
    }
    assert set(data.keys()) == expected_keys


def test_default_payload_types():
    data = default_payload()
    assert isinstance(data["leftFist"], bool)
    assert isinstance(data["head_yaw"], float)
    assert isinstance(data["tilt_left"], bool)
    assert isinstance(data["tilt_right"], bool)
    assert isinstance(data["hands"], list)
    assert isinstance(data["lstm_gesture"], str)
    assert isinstance(data["indexTipX"], float)
    assert isinstance(data["indexTipY"], float)
    assert data["indexTipX"] == -1.0
    assert data["indexTipY"] == -1.0
    assert data["player_locked"] is False
    assert data["lock_id"] == 0
    assert data["lock_status"] == "unlocked"
    # Gate ships closed so Unity, not the payload default, opens it per puzzle.
    assert data["puzzle_active"] is False
    assert data["puzzle_gate_source"] == "default"


def test_v1_single_hand_fields_are_gone():
    """Superseded by the left/right split; Unity reads palmX/palmY instead."""
    data = default_payload()
    for key in ("hand_x", "hand_y", "hand_up", "openPalm", "isFist", "landmarks"):
        assert key not in data
