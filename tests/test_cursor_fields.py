from vision_server.gestures.hand.cursor_fields import (
    apply_right_hand_cursor_fields,
    reset_last_point,
)
from vision_server.udp import default_payload

from conftest import make_landmarks


def _pointing_gestures():
    return {"fist": False, "open_palm": False, "index_up": True, "peace": False}


def _fist_gestures():
    return {"fist": True, "open_palm": False, "index_up": False, "peace": False}


def test_apply_right_hand_cursor_fields_when_pointing():
    data = default_payload()
    last_point = [-1.0, -1.0]
    landmarks = make_landmarks(
        {
            8: {"x": 0.6, "y": 0.3},
            0: {"x": 0.5, "y": 0.5},
            5: {"x": 0.5, "y": 0.5},
            9: {"x": 0.5, "y": 0.5},
            13: {"x": 0.5, "y": 0.5},
            17: {"x": 0.5, "y": 0.5},
        }
    )

    apply_right_hand_cursor_fields(data, landmarks, _pointing_gestures(), last_point)

    assert data["palmX"] == 0.5
    assert data["palmY"] == 0.5
    assert data["indexTipX"] == 0.6
    assert data["indexTipY"] == 0.3
    assert last_point == [0.5, 0.5]


def test_apply_right_hand_cursor_fields_latches_palm_on_fist():
    data = default_payload()
    last_point = [0.4, 0.6]
    landmarks = make_landmarks(
        {
            8: {"x": 0.9, "y": 0.9},
            0: {"x": 0.8, "y": 0.8},
            5: {"x": 0.8, "y": 0.8},
            9: {"x": 0.8, "y": 0.8},
            13: {"x": 0.8, "y": 0.8},
            17: {"x": 0.8, "y": 0.8},
        }
    )

    apply_right_hand_cursor_fields(data, landmarks, _fist_gestures(), last_point)

    assert data["palmX"] == 0.4
    assert data["palmY"] == 0.6
    assert data["indexTipX"] == -1.0
    assert data["indexTipY"] == -1.0


def test_reset_last_point():
    last_point = [0.4, 0.6]
    reset_last_point(last_point)
    assert last_point == [-1.0, -1.0]
