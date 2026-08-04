"""Unit tests for LSTM one-shot action hold / hand-miss buffer without TF."""

from collections import deque

from vision_server.gestures.dynamic.lstm import GestureLSTM


def _lstm_without_model(*, hold_s=1.0, miss_clear_s=1.0) -> GestureLSTM:
    lstm = GestureLSTM.__new__(GestureLSTM)
    lstm.model = None
    lstm.frame_buffer = deque(maxlen=30)
    lstm.buffer_size = 30
    lstm.confidence_threshold = 0.8
    lstm.hand_miss_clear_s = miss_clear_s
    lstm.action_max_hold_s = hold_s
    lstm._hand_miss_started_at = None
    lstm.last_output = "Idle"
    lstm.classes = ["Idle", "Turn_Key"]
    lstm._action_class = None
    lstm._action_started_at = 0.0
    lstm._action_expired = False
    return lstm


def test_action_hold_reports_then_expires(monkeypatch):
    lstm = _lstm_without_model(hold_s=1.0)
    t = {"now": 100.0}
    monkeypatch.setattr(
        "vision_server.gestures.dynamic.lstm.time.monotonic",
        lambda: t["now"],
    )

    assert lstm._apply_action_hold("Turn_Key") == "Turn_Key"
    t["now"] = 100.5
    assert lstm._apply_action_hold("Turn_Key") == "Turn_Key"
    t["now"] = 101.0
    assert lstm._apply_action_hold("Turn_Key") == "Idle"
    t["now"] = 101.5
    assert lstm._apply_action_hold("Turn_Key") == "Idle"


def test_action_hold_rearms_after_idle(monkeypatch):
    lstm = _lstm_without_model(hold_s=1.0)
    t = {"now": 200.0}
    monkeypatch.setattr(
        "vision_server.gestures.dynamic.lstm.time.monotonic",
        lambda: t["now"],
    )

    assert lstm._apply_action_hold("Turn_Key") == "Turn_Key"
    t["now"] = 201.1
    assert lstm._apply_action_hold("Turn_Key") == "Idle"
    assert lstm._apply_action_hold("Idle") == "Idle"
    t["now"] = 201.2
    assert lstm._apply_action_hold("Turn_Key") == "Turn_Key"


def test_action_hold_switches_class_restarts_timer(monkeypatch):
    lstm = _lstm_without_model(hold_s=1.0)
    t = {"now": 300.0}
    monkeypatch.setattr(
        "vision_server.gestures.dynamic.lstm.time.monotonic",
        lambda: t["now"],
    )

    assert lstm._apply_action_hold("Turn_Key") == "Turn_Key"
    t["now"] = 300.8
    assert lstm._apply_action_hold("Pull_Lever") == "Pull_Lever"
    t["now"] = 301.3
    assert lstm._apply_action_hold("Pull_Lever") == "Pull_Lever"


def test_brief_hand_miss_keeps_buffer(monkeypatch):
    lstm = _lstm_without_model(miss_clear_s=1.0)
    t = {"now": 50.0}
    monkeypatch.setattr(
        "vision_server.gestures.dynamic.lstm.time.monotonic",
        lambda: t["now"],
    )

    for _ in range(30):
        lstm.frame_buffer.append([0.0] * 63)
    lstm.last_output = "Idle"

    lstm.register_hand_lost()
    t["now"] = 50.4
    lstm.register_hand_lost()
    assert len(lstm.frame_buffer) == 30

    t["now"] = 51.1
    lstm.register_hand_lost()
    assert len(lstm.frame_buffer) == 0
    assert lstm.last_output == "Idle"
