"""Two-hand watch tap gesture and solo-gesture suppression.

The wrist wearing the "watch" is the MOVE hand's; the fingers doing the tapping
are the ACTION hand's. Callers pass them in that order, so swapping the hand
roles swaps which physical wrist is tapped without touching this module.

Three things decide a tap, and the first two are what stop it firing by
accident:

* **Scale.** The distance from the tapping fingertip to the watch wrist is
  divided by the palm length, like every other measurement in this package.
  The original threshold was a flat 0.09 in normalised image coordinates,
  which is a fixed slice of the FRAME rather than of the HAND: lean back from
  the camera and the hand shrinks while the threshold does not, until 0.09
  spans a whole hand and any two hands in the same region of the frame read as
  a tap. That is the same bug ``THUMB_SPREAD_RATIO`` was fixed for, in the
  direction that costs false triggers rather than recall.

* **Pose.** The tapping hand has to be pointing — index extended, the other
  three curled — so a hand merely passing near the other wrist is not a tap.
  Deliberately direction-FREE: it uses the finger pattern from
  :func:`~.fingers.finger_states` and not the ``index_up`` label, because
  tapping the left wrist points the finger left and down. Reading the label
  would gate the gesture on a direction it never has.

* **Time.** Raw per-frame, one bad landmark frame is one pause: Unity fires on
  the rising edge of ``watchTap``. So it debounces like the solo gestures do —
  see :class:`WatchTapDebouncer`.

The pose check also settles which fingertip to measure from. The old distance
took ``min(index_tip, middle_tip)``, so a middle finger near the wrist counted;
with the pattern pinned, the index is the only finger out and the only one
worth measuring.
"""

from __future__ import annotations

import math

from vision_server.config import (
    WATCH_TAP_OFF_FRAMES,
    WATCH_TAP_ON_FRAMES,
    WATCH_TAP_RATIO,
)

from .cursor_fields import INVALID_COORD
from .fingers import FINGERS, finger_states, hand_frame

_WRIST = 0
_INDEX_TIP = 8

# The tapping hand: one finger out, three folded. Any pointing direction.
_TAP_PATTERN = {"index": True, "middle": False, "ring": False, "pinky": False}


def _landmark_distance(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def _tap_scale(watch_landmarks, pointer_landmarks) -> float | None:
    """Palm length to measure the tap distance in.

    The watch hand's, since the distance is measured to its wrist — but either
    hand will do when that one is too foreshortened to measure, because during
    a real tap the two hands are touching and therefore at the same depth.
    ``None`` when neither is readable; a frame that cannot be scaled is not
    one to guess a tap from.
    """
    for landmarks in (watch_landmarks, pointer_landmarks):
        frame = hand_frame(landmarks) if landmarks is not None else None
        if frame is not None:
            return frame[1]
    return None


def is_tapping_pose(pointer_landmarks) -> bool:
    """True when the hand is pointing with one finger, whichever way it points."""
    if pointer_landmarks is None:
        return False

    states = finger_states(pointer_landmarks)
    if states is None:
        return False

    # An ambiguous finger reads None, which matches neither half of the
    # pattern, so a hand mid-transition is not a tapping hand.
    return all(states[name] == _TAP_PATTERN[name] for name in FINGERS)


def tap_distance(watch_landmarks, pointer_landmarks) -> float | None:
    """Index tip to watch wrist, in palm lengths. ``None`` if unmeasurable."""
    if watch_landmarks is None or pointer_landmarks is None:
        return None

    scale = _tap_scale(watch_landmarks, pointer_landmarks)
    if scale is None:
        return None

    gap = _landmark_distance(pointer_landmarks[_INDEX_TIP], watch_landmarks[_WRIST])
    return gap / scale


def is_watch_tap(watch_landmarks, pointer_landmarks) -> bool:
    """Raw per-frame test, before debouncing.

    Callers driving Unity want :class:`WatchTapDebouncer` around this rather
    than the bare value.
    """
    if not is_tapping_pose(pointer_landmarks):
        return False

    distance = tap_distance(watch_landmarks, pointer_landmarks)
    return distance is not None and distance <= WATCH_TAP_RATIO


class WatchTapDebouncer:
    """Frame-count hysteresis over the raw tap test.

    Simpler than :class:`~.debounce.GestureDebouncer` because there is only one
    label and no way to arrive at a competing one, so a single counter of
    consecutive disagreeing frames is enough. The asymmetry is the same idea
    though, and for the same reasons:

    * ``on`` — Unity pauses on the RISING edge, so one flicker frame is one
      unwanted pause. Entering slowly is the cheapest defence there is, and it
      costs only latency on a gesture nobody makes in a hurry.
    * ``off`` — a dropped frame mid-tap would otherwise release and re-arm,
      and the re-arm is a second rising edge: pause, unpause, pause.
    """

    def __init__(
        self,
        on_frames: int = WATCH_TAP_ON_FRAMES,
        off_frames: int = WATCH_TAP_OFF_FRAMES,
    ):
        self._on_frames = on_frames
        self._off_frames = off_frames
        self.tapping = False
        self.raw = False
        self._streak = 0

    def reset(self) -> None:
        """Drop back to "not tapping", committing immediately.

        For when the hands behind the counter change identity — a new locked
        player or a MOVE/ACTION role swap — where the streak so far was counted
        on somebody else's hands.
        """
        self.tapping = False
        self.raw = False
        self._streak = 0

    def update(self, raw: bool) -> bool:
        """Feed this frame's raw test; return the committed value.

        Pass ``False`` on frames where either hand was not seen, so a tracking
        dropout has to outlast the off-count exactly like a misread pose.
        """
        self.raw = raw

        if raw == self.tapping:
            self._streak = 0
            return self.tapping

        self._streak += 1
        needed = self._off_frames if self.tapping else self._on_frames
        if self._streak >= needed:
            self.tapping = raw
            self._streak = 0

        return self.tapping


def _clear_solo_hand_gestures(data: dict) -> None:
    """Blank the one-hand gesture fields while a two-hand tap owns both hands.

    ``*GestureRaw`` is left alone on purpose: it reports what the classifier
    saw, and a capture is more useful if it still shows the poses the hands
    were actually in during the tap.
    """
    data["leftFist"] = False
    data["leftOpenPalm"] = False
    data["leftIndexUp"] = False
    data["leftPeace"] = False
    data["leftOkSign"] = False
    data["leftRockSign"] = False
    data["leftIndexLeft"] = False
    data["leftIndexRight"] = False
    data["leftIndexDown"] = False
    data["rightFist"] = False
    data["rightOpenPalm"] = False
    data["rightIndexUp"] = False
    data["rightPeace"] = False
    data["rightOkSign"] = False
    data["rightRockSign"] = False
    data["rightIndexLeft"] = False
    data["rightIndexRight"] = False
    data["rightIndexDown"] = False
    data["moveGesture"] = "none"
    data["actionGesture"] = "none"
    data["fistRotX"] = 0.0
    data["fistRotY"] = 0.0
    data["fistRotZ"] = 0.0
    data["palmX"] = INVALID_COORD
    data["palmY"] = INVALID_COORD
    data["indexTipX"] = INVALID_COORD
    data["indexTipY"] = INVALID_COORD
    data["lstm_gesture"] = "Idle"


def apply_watch_tap_fields(
    data: dict,
    watch_landmarks,
    pointer_landmarks,
    debouncer: WatchTapDebouncer,
) -> bool:
    """Write the tap fields onto the payload; return the committed tap.

    ``debouncer`` is owned by the caller and must be updated on EVERY frame,
    seen hands or not — that is what makes a dropout cost the off-count.
    """
    data["watchTapDistance"] = tap_distance(watch_landmarks, pointer_landmarks)
    data["watchTapRaw"] = is_watch_tap(watch_landmarks, pointer_landmarks)
    data["watchTap"] = debouncer.update(data["watchTapRaw"])

    if data["watchTap"]:
        _clear_solo_hand_gestures(data)

    return data["watchTap"]
