from vision_server.gestures.hand import HAND_RULES
from vision_server.gestures.hand.fist import is_fist
from vision_server.gestures.hand.index_up import is_index_up
from vision_server.gestures.hand.open_palm import is_open_palm
from vision_server.gestures.hand.peace import is_peace_sign
from conftest import make_landmarks


def _folded_fist_landmarks():
    """Tips below bases on y-axis -> folded fingers."""
    return make_landmarks(
        {
            8: {"y": 0.8},
            6: {"y": 0.5},
            12: {"y": 0.8},
            10: {"y": 0.5},
            16: {"y": 0.8},
            14: {"y": 0.5},
            20: {"y": 0.8},
            18: {"y": 0.5},
            5: {"y": 0.4},
            9: {"y": 0.4},
            13: {"y": 0.4},
            17: {"y": 0.4},
        }
    )


def test_hand_rules_registry_contains_expected_gestures():
    names = {name for name, _ in HAND_RULES}
    assert names == {"fist", "open_palm", "peace", "index_up"}


def test_is_fist_detects_folded_fingers():
    assert is_fist(_folded_fist_landmarks()) is True


def test_is_fist_rejects_open_hand():
    landmarks = make_landmarks(
        {
            8: {"y": 0.1},
            6: {"y": 0.5},
            12: {"y": 0.1},
            10: {"y": 0.5},
            16: {"y": 0.1},
            14: {"y": 0.5},
            20: {"y": 0.1},
            18: {"y": 0.5},
            5: {"y": 0.6},
            9: {"y": 0.6},
            13: {"y": 0.6},
            17: {"y": 0.6},
        }
    )
    assert is_fist(landmarks) is False


def test_is_fist_rejects_index_up():
    landmarks = make_landmarks(
        {
            8: {"y": 0.1},
            6: {"y": 0.5},
            12: {"y": 0.8},
            10: {"y": 0.5},
            16: {"y": 0.8},
            14: {"y": 0.5},
            20: {"y": 0.8},
            18: {"y": 0.5},
        }
    )
    assert is_fist(landmarks) is False


def test_is_index_up():
    landmarks = make_landmarks(
        {
            8: {"y": 0.1},
            6: {"y": 0.5},
            12: {"y": 0.8},
            10: {"y": 0.5},
            16: {"y": 0.8},
            14: {"y": 0.5},
            20: {"y": 0.8},
            18: {"y": 0.5},
        }
    )
    assert is_index_up(landmarks) is True


def test_is_peace_sign():
    landmarks = make_landmarks(
        {
            8: {"y": 0.1},
            6: {"y": 0.5},
            12: {"y": 0.1},
            10: {"y": 0.5},
            16: {"y": 0.8},
            14: {"y": 0.5},
            20: {"y": 0.8},
            18: {"y": 0.5},
        }
    )
    assert is_peace_sign(landmarks) is True


def test_is_open_palm_with_all_fingers_extended():
    landmarks = make_landmarks(
        {
            0: {"x": 0.5, "y": 0.5},
            5: {"x": 0.5, "y": 0.5},
            9: {"x": 0.5, "y": 0.5},
            13: {"x": 0.5, "y": 0.5},
            17: {"x": 0.5, "y": 0.5},
            4: {"x": 0.3, "y": 0.3},
            8: {"x": 0.5, "y": 0.2},
            6: {"y": 0.5},
            12: {"x": 0.5, "y": 0.2},
            10: {"y": 0.5},
            16: {"x": 0.5, "y": 0.2},
            14: {"y": 0.5},
            20: {"x": 0.5, "y": 0.2},
            18: {"y": 0.5},
        }
    )
    assert is_open_palm(landmarks) is True


def test_is_open_palm_rejects_partial_extension():
    landmarks = make_landmarks(
        {
            0: {"x": 0.5, "y": 0.5},
            5: {"x": 0.5, "y": 0.5},
            9: {"x": 0.5, "y": 0.5},
            13: {"x": 0.5, "y": 0.5},
            17: {"x": 0.5, "y": 0.5},
            4: {"x": 0.3, "y": 0.3},
            8: {"x": 0.5, "y": 0.2},
            6: {"y": 0.5},
            12: {"x": 0.5, "y": 0.8},
            10: {"y": 0.5},
            16: {"x": 0.5, "y": 0.8},
            14: {"y": 0.5},
            20: {"x": 0.5, "y": 0.8},
            18: {"y": 0.5},
        }
    )
    assert is_open_palm(landmarks) is False
