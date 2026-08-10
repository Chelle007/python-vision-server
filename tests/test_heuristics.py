"""Static hand gesture rules.

Fixtures are built with :func:`hand_pose` rather than by hand-setting y values.
The rules measure finger extension along the wrist -> middle-MCP axis and
divide by the palm length, so a fixture that leaves x at its default and only
moves y has a zero-length palm and no measurable axis at all.
"""

import math

from vision_server.gestures.hand import HAND_RULES, rules_from_label
from vision_server.gestures.hand.classify import NONE, classify_hand
from vision_server.gestures.hand.fingers import finger_margins
from vision_server.gestures.hand.fist import is_fist
from vision_server.gestures.hand.index_up import is_index_up
from vision_server.gestures.hand.open_palm import is_open_palm
from vision_server.gestures.hand.peace import is_peace_sign
from conftest import hand_pose


def test_hand_rules_registry_contains_expected_gestures():
    names = {name for name, _ in HAND_RULES}
    assert names == {"fist", "open_palm", "peace", "index_up"}


def test_is_fist_detects_folded_fingers():
    assert is_fist(hand_pose()) is True


def test_is_fist_rejects_open_hand():
    assert is_fist(hand_pose(index=True, middle=True, ring=True, pinky=True)) is False


def test_is_fist_rejects_index_up():
    assert is_fist(hand_pose(index=True)) is False


def test_is_index_up():
    assert is_index_up(hand_pose(index=True)) is True


def test_is_peace_sign():
    assert is_peace_sign(hand_pose(index=True, middle=True)) is True


def test_is_open_palm_with_all_fingers_extended():
    landmarks = hand_pose(
        index=True, middle=True, ring=True, pinky=True, thumb_out=True
    )
    assert is_open_palm(landmarks) is True


def test_is_open_palm_rejects_partial_extension():
    landmarks = hand_pose(index=True, thumb_out=True)
    assert is_open_palm(landmarks) is False


def test_open_palm_requires_thumb_spread():
    """All four fingers up with a tucked thumb is not the release gesture."""
    landmarks = hand_pose(
        index=True, middle=True, ring=True, pinky=True, thumb_out=False
    )
    assert is_open_palm(landmarks) is False
    assert classify_hand(landmarks) == NONE


# --- Rotation invariance ---------------------------------------------------


def test_fist_survives_ninety_degree_wrist_rotation():
    """The old image-space rules read a horizontal fist as an open hand."""
    for degrees in (0, 45, 90, 135, 180, 270):
        assert is_fist(hand_pose(rotation_deg=degrees)) is True, degrees


def test_index_up_survives_wrist_rotation():
    """Pointing sideways still points."""
    for degrees in (0, 45, 90, 180, 270):
        assert is_index_up(hand_pose(index=True, rotation_deg=degrees)) is True, degrees


def test_margins_are_scale_invariant():
    """Sitting further from the camera must not change the classification."""
    near = finger_margins(hand_pose(index=True, scale=1.0))
    far = finger_margins(hand_pose(index=True, scale=0.35))
    for finger, value in near.items():
        assert math.isclose(value, far[finger], abs_tol=1e-6)


# --- Mutual exclusivity and the "no gesture" state -------------------------


def test_exactly_one_gesture_fires_per_hand():
    poses = [
        hand_pose(),
        hand_pose(index=True),
        hand_pose(index=True, middle=True),
        hand_pose(index=True, middle=True, ring=True, pinky=True, thumb_out=True),
    ]
    for landmarks in poses:
        fired = [name for name, on in rules_from_label(classify_hand(landmarks)).items() if on]
        assert len(fired) == 1, fired


def test_unlisted_finger_pattern_is_no_gesture():
    """Ring up alone is not one of the four gestures, so it is none of them."""
    landmarks = hand_pose(ring=True)
    assert classify_hand(landmarks) == NONE
    assert not any(rules_from_label(classify_hand(landmarks)).values())


def test_half_extended_finger_is_no_gesture():
    """A finger inside the dead zone makes the whole hand ambiguous.

    This is the fist -> peace transition: the index has started to extend but
    has not committed. Reporting index_up here is what fired phantom jumps.
    """
    landmarks = hand_pose(index=True, index_extension=0.15)
    assert classify_hand(landmarks) == NONE


def test_hand_pointing_at_camera_is_no_gesture():
    """No measurable palm axis means no answer, rather than a guessed one."""
    landmarks = hand_pose(scale=0.0)
    assert classify_hand(landmarks) == NONE
    assert not any(rules_from_label(classify_hand(landmarks)).values())
