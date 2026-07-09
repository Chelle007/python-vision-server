from vision_server.gestures.hand.watch_tap import apply_watch_tap_fields, is_watch_tap
from vision_server.udp import default_payload

from conftest import make_landmarks


def _left_wrist(x=0.3, y=0.5):
    return make_landmarks({0: {"x": x, "y": y}})


def _right_index(x=0.7, y=0.5):
    return make_landmarks({8: {"x": x, "y": y}})


def test_is_watch_tap_requires_both_hands():
    assert is_watch_tap(None, _right_index()) is False
    assert is_watch_tap(_left_wrist(), None) is False


def test_is_watch_tap_true_when_touching():
    left = _left_wrist(x=0.3, y=0.5)
    assert is_watch_tap(left, _right_index(x=0.8, y=0.5)) is False
    assert is_watch_tap(left, _right_index(x=0.31, y=0.5)) is True


def test_is_watch_tap_stays_true_until_finger_moves_away():
    left = _left_wrist(x=0.3, y=0.5)
    assert is_watch_tap(left, _right_index(x=0.31, y=0.5)) is True
    assert is_watch_tap(left, _right_index(x=0.31, y=0.5)) is True
    assert is_watch_tap(left, _right_index(x=0.5, y=0.5)) is False


def test_apply_watch_tap_fields_clears_solo_gestures():
    data = default_payload()
    data["leftFist"] = True
    data["rightFist"] = True
    data["leftIndexUp"] = True
    data["lstm_gesture"] = "Turn_Key"

    left = _left_wrist(x=0.3, y=0.5)
    right = _right_index(x=0.31, y=0.5)

    assert apply_watch_tap_fields(data, left, right) is True
    assert data["watchTap"] is True
    assert data["watchTapDistance"] <= 0.09
    assert data["leftFist"] is False
    assert data["rightFist"] is False
    assert data["leftIndexUp"] is False
    assert data["lstm_gesture"] == "Idle"
