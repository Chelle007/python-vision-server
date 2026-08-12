from pytest import approx

from vision_server.config import WATCH_TAP_OFF_FRAMES, WATCH_TAP_ON_FRAMES
from vision_server.gestures.hand.watch_tap import (
    WatchTapDebouncer,
    apply_watch_tap_fields,
    is_watch_tap,
    tap_distance,
)
from vision_server.udp import default_payload

from conftest import hand_pose

# hand_pose builds every hand at the same wrist position, so a two-hand fixture
# has to move one of them. Palm length is 0.12 * scale, which is what the
# offsets below are quoted in.
_PALM = 0.12


def _shift(landmarks, dx, dy):
    for landmark in landmarks:
        landmark.x += dx
        landmark.y += dy
    return landmarks


def _watch_hand():
    """The hand being tapped. Only its wrist and palm axis are read."""
    return hand_pose(index=False, middle=False, ring=False, pinky=False)


def _pointer(distance_palms, *, pose="point", scale=1.0):
    """A hand whose index tip sits ``distance_palms`` from the watch wrist.

    ``hand_pose`` puts the wrist at (0.5, 0.5) and the extended index tip
    roughly 1.3 palm lengths above it, so the whole hand is shifted down by
    that much to bring the TIP onto the watch wrist, then out by the distance
    under test.
    """
    poses = {
        # index out, three curled — the tapping pose, at any pointing angle.
        "point": dict(index=True),
        # every finger out: near the wrist but not pointing.
        "open": dict(index=True, middle=True, ring=True, pinky=True),
        # nothing out: a fist resting near the wrist.
        "fist": dict(),
        # index and middle out — the old min(8, 12) distance let this through.
        "peace": dict(index=True, middle=True),
    }
    landmarks = hand_pose(scale=scale, **poses[pose])

    tip = landmarks[8]
    _shift(landmarks, 0.5 - tip.x, 0.5 - tip.y)
    return _shift(landmarks, distance_palms * _PALM * scale, 0.0)


def test_tap_distance_is_in_palm_lengths():
    assert tap_distance(_watch_hand(), _pointer(0.5)) == approx(0.5)
    assert tap_distance(_watch_hand(), _pointer(2.0)) == approx(2.0)


def test_tap_distance_does_not_change_with_distance_from_camera():
    """The bug the ratio replaces: a flat image-space threshold got looser as
    the player leaned back, until any two hands in frame read as a tap.

    The same half-palm gap is four times wider in image coordinates up close
    than at arm's length, and has to read the same either way.
    """
    near = tap_distance(hand_pose(scale=2.0), _pointer(0.5, scale=2.0))
    far = tap_distance(hand_pose(scale=0.5), _pointer(0.5, scale=0.5))

    assert near == approx(0.5)
    assert far == approx(0.5)


def test_is_watch_tap_requires_both_hands():
    assert is_watch_tap(None, _pointer(0.1)) is False
    assert is_watch_tap(_watch_hand(), None) is False


def test_is_watch_tap_true_when_touching():
    assert is_watch_tap(_watch_hand(), _pointer(3.0)) is False
    assert is_watch_tap(_watch_hand(), _pointer(0.1)) is True


def test_is_watch_tap_requires_a_pointing_hand():
    """Near the wrist is not enough — the fingers have to be tapping."""
    assert is_watch_tap(_watch_hand(), _pointer(0.1, pose="open")) is False
    assert is_watch_tap(_watch_hand(), _pointer(0.1, pose="fist")) is False


def test_is_watch_tap_ignores_the_middle_finger():
    """The old distance took min(index tip, middle tip), so a peace sign held
    against the wrist fired. The pose gate now rejects it outright."""
    assert is_watch_tap(_watch_hand(), _pointer(0.1, pose="peace")) is False


def test_is_watch_tap_survives_wrist_rotation():
    """Tapping your own wrist points the finger sideways/down, never up, so
    nothing here may depend on which way it points."""
    for rotation in (0, 90, 180, 270):
        pointer = hand_pose(index=True, rotation_deg=rotation)
        tip = pointer[8]
        _shift(pointer, 0.5 - tip.x, 0.5 - tip.y)
        assert is_watch_tap(_watch_hand(), pointer) is True, rotation


def test_debouncer_needs_consecutive_frames_to_commit():
    debounce = WatchTapDebouncer(on_frames=3, off_frames=3)

    assert debounce.update(True) is False
    assert debounce.update(True) is False
    assert debounce.update(True) is True


def test_debouncer_drops_a_lone_flicker_frame():
    """One bad landmark frame is one unwanted pause: Unity fires on the rising
    edge of watchTap."""
    debounce = WatchTapDebouncer(on_frames=3, off_frames=3)

    for _ in range(10):
        assert debounce.update(True) is False
        assert debounce.update(False) is False


def test_debouncer_holds_through_a_dropped_frame():
    debounce = WatchTapDebouncer(on_frames=2, off_frames=3)

    debounce.update(True)
    assert debounce.update(True) is True
    assert debounce.update(False) is True
    assert debounce.update(True) is True
    # ...and still releases on a real one.
    assert debounce.update(False) is True
    assert debounce.update(False) is True
    assert debounce.update(False) is False


def test_debouncer_reset_commits_immediately():
    debounce = WatchTapDebouncer(on_frames=1, off_frames=5)
    assert debounce.update(True) is True

    debounce.reset()
    assert debounce.tapping is False
    assert debounce.raw is False


def test_config_debounce_values_are_meaningful():
    assert WATCH_TAP_ON_FRAMES >= 2
    assert WATCH_TAP_OFF_FRAMES >= 2


def test_apply_watch_tap_fields_clears_solo_gestures():
    data = default_payload()
    data["leftFist"] = True
    data["rightFist"] = True
    data["leftIndexUp"] = True
    data["lstm_gesture"] = "Turn_Key"

    debounce = WatchTapDebouncer(on_frames=1, off_frames=1)
    assert apply_watch_tap_fields(data, _watch_hand(), _pointer(0.1), debounce) is True
    assert data["watchTap"] is True
    assert data["watchTapRaw"] is True
    assert data["watchTapDistance"] < 0.2
    assert data["leftFist"] is False
    assert data["rightFist"] is False
    assert data["leftIndexUp"] is False
    assert data["lstm_gesture"] == "Idle"


def test_apply_watch_tap_fields_reports_raw_before_commit():
    """The overlay reads watchTapRaw to show the pose gate passing while the
    commit is still counting out."""
    data = default_payload()
    debounce = WatchTapDebouncer(on_frames=3, off_frames=3)

    assert apply_watch_tap_fields(data, _watch_hand(), _pointer(0.1), debounce) is False
    assert data["watchTapRaw"] is True
    assert data["watchTap"] is False


def test_apply_watch_tap_fields_with_one_hand_missing():
    data = default_payload()
    debounce = WatchTapDebouncer(on_frames=1, off_frames=1)

    assert apply_watch_tap_fields(data, None, _pointer(0.1), debounce) is False
    assert data["watchTapDistance"] is None
    assert data["watchTapRaw"] is False
