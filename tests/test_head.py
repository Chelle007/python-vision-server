from vision_server.gestures.head import HEAD_RULES
from vision_server.gestures.head.tilt import detect_head_tilt

from conftest import make_face_landmarks


def test_head_rules_registry_contains_expected_gestures():
    names = {name for name, _ in HEAD_RULES}
    assert names == {"orientation", "tilt"}


def test_detect_head_tilt_neutral():
    result = detect_head_tilt(make_face_landmarks())
    assert result == {"tilt_left": False, "tilt_right": False}


def test_detect_head_tilt_left():
    result = detect_head_tilt(
        make_face_landmarks(
            {
                234: {"x": 0.35, "y": 0.65},
                454: {"x": 0.65, "y": 0.35},
            }
        )
    )
    assert result["tilt_left"] is True
    assert result["tilt_right"] is False


def test_detect_head_tilt_right():
    result = detect_head_tilt(
        make_face_landmarks(
            {
                234: {"x": 0.35, "y": 0.35},
                454: {"x": 0.65, "y": 0.65},
            }
        )
    )
    assert result["tilt_left"] is False
    assert result["tilt_right"] is True
