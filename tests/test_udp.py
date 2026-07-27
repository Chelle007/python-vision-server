"""Verify UDP payload contract matches Unity expectations."""

from vision_server.udp import default_payload


def test_default_payload_has_unity_keys():
    data = default_payload()
    expected_keys = {
        "hand_x",
        "hand_y",
        "hand_up",
        "head_yaw",
        "head_pitch",
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
        "landmarks",
        "openPalm",
        "isFist",
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
