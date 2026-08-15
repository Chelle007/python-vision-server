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
from vision_server.gestures.hand.index_down import is_index_down
from vision_server.gestures.hand.index_left import is_index_left
from vision_server.gestures.hand.index_right import is_index_right
from vision_server.gestures.hand.index_up import is_index_up
from vision_server.gestures.hand.open_palm import is_open_palm
from vision_server.gestures.hand.peace import is_peace_sign
from vision_server.gestures.hand.ok_sign import is_ok_sign
from vision_server.gestures.hand.rock_sign import is_rock_sign
from conftest import hand_pose


def test_hand_rules_registry_contains_expected_gestures():
    names = {name for name, _ in HAND_RULES}
    assert names == {
        "fist",
        "open_palm",
        "peace",
        "index_up",
        "index_left",
        "index_right",
        "index_down",
        "ok_sign",
        "rock_sign",
    }


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


# --- OK sign (inventory) ---------------------------------------------------
# Index curled, the other three extended, thumb closing the ring. The pattern
# is unique to this gesture; the thumb only confirms it.


def _ok_sign(**kwargs):
    return hand_pose(middle=True, ring=True, pinky=True, thumb_pinch=True, **kwargs)


def test_is_ok_sign():
    assert is_ok_sign(_ok_sign()) is True


def test_ok_sign_does_not_collide_with_walking():
    """The reason this gesture replaced a thumbs-up.

    A thumbs-up curls all four fingers, so it arrived at the classifier wearing
    the fist's pattern and the thumb had to arbitrate between "open the menu"
    and "walk forward". The OK sign leaves three fingers extended, so a fist
    can never be mistaken for it no matter what the thumb is doing — which is
    what the second half of this test pins down.
    """
    landmarks = _ok_sign()
    assert is_fist(landmarks) is False
    assert classify_hand(landmarks) == "ok_sign"

    # Every placement below is a thumb held well clear of the curled fingers,
    # i.e. exactly the poses the old thumbs-up rule was trying to catch. All of
    # them are now plain fists, so walking survives them.
    for tip in ((-0.45, 1.60), (-0.77, 1.58), (-1.05, 1.05), (-1.25, 0.55)):
        assert classify_hand(hand_pose(thumb_tip_at=tip)) == "fist", tip


def test_open_ring_is_not_an_ok_sign():
    """Three fingers up with the thumb parked is the pose the pinch excludes."""
    landmarks = hand_pose(middle=True, ring=True, pinky=True, thumb_out=True)
    assert is_ok_sign(landmarks) is False
    assert classify_hand(landmarks) == NONE


def test_ok_sign_survives_wrist_rotation():
    """Both tips are read in the hand's own frame, so a tilted OK still counts."""
    for degrees in (0, 45, 90, 180, 270):
        assert is_ok_sign(_ok_sign(rotation_deg=degrees)) is True, degrees


def test_foreshortened_ok_sign_is_no_gesture():
    """Below the palm floor a hand aimed at the camera fakes a closed ring.

    The projection collapses every landmark toward every other, so an open
    pinch measures the same as a shut one. Such a hand still has a measurable
    axis, so it is not rejected outright — it simply is not eligible, and the
    frame reports NONE rather than falling through to a neighbouring gesture.
    """
    landmarks = _ok_sign(scale=0.4)
    assert is_ok_sign(landmarks) is False
    assert classify_hand(landmarks) == NONE


# --- Pointing left / right -------------------------------------------------
# Same one-extended-index pattern as jump, split by direction. Unlike every
# other rule here, direction is read in SCREEN space, because "point left"
# describes the room rather than the hand.


def test_point_right_and_left():
    # The fixture's rotation turns the whole hand about the wrist, so 90 puts
    # the index along screen +x and 270 along -x.
    assert is_index_right(hand_pose(index=True, rotation_deg=90)) is True
    assert is_index_left(hand_pose(index=True, rotation_deg=270)) is True


def test_pointing_is_read_in_screen_space_not_hand_space():
    """A rolled wrist must not change which way the player is pointing.

    Reading direction in the hand's own frame is what broke two thumbs-up
    attempts; here it would be worse, because left and right would swap.
    """
    for degrees in (75, 90, 105):
        assert is_index_right(hand_pose(index=True, rotation_deg=degrees)) is True, degrees


def test_diagonal_point_is_no_gesture():
    """The cones do not touch, so a diagonal commits to neither."""
    for degrees in (45, 135, 225, 315):
        assert classify_hand(hand_pose(index=True, rotation_deg=degrees)) == NONE, degrees


def test_point_down():
    assert is_index_down(hand_pose(index=True, rotation_deg=180)) is True


def test_point_down_is_read_in_screen_space_too():
    """Down is a claim about the room, same as left and right."""
    for degrees in (165, 180, 195):
        assert is_index_down(hand_pose(index=True, rotation_deg=degrees)) is True, degrees


# --- Rock sign -------------------------------------------------------------


def test_is_rock_sign():
    assert is_rock_sign(hand_pose(index=True, pinky=True)) is True


def test_rock_sign_is_not_peace_or_index_up():
    landmarks = hand_pose(index=True, pinky=True)
    assert is_peace_sign(landmarks) is False
    assert is_index_up(landmarks) is False


def test_rock_sign_survives_wrist_rotation():
    for degrees in (0, 45, 90, 180, 270):
        assert is_rock_sign(hand_pose(index=True, pinky=True, rotation_deg=degrees)) is True, degrees


# --- Rotation invariance ---------------------------------------------------


def test_fist_survives_ninety_degree_wrist_rotation():
    """The old image-space rules read a horizontal fist as an open hand."""
    for degrees in (0, 45, 90, 135, 180, 270):
        assert is_fist(hand_pose(rotation_deg=degrees)) is True, degrees


def test_index_up_requires_pointing_up():
    """An extended index is three gestures now, so jump must mean UP.

    This replaces an older test asserting that a sideways index still counted
    as index_up. That is deliberately no longer true: pointing left and right
    step the inventory, and jump would fire on both if direction were ignored.
    """
    assert is_index_up(hand_pose(index=True)) is True
    for degrees in (20, 340):
        assert is_index_up(hand_pose(index=True, rotation_deg=degrees)) is True, degrees
    for degrees in (90, 270):
        assert is_index_up(hand_pose(index=True, rotation_deg=degrees)) is False, degrees


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
        _ok_sign(),
        hand_pose(index=True, pinky=True),
        hand_pose(index=True, rotation_deg=90),
        hand_pose(index=True, rotation_deg=270),
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
