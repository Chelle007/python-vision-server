"""Two-hand watch tap gesture and solo-gesture suppression.

The wrist wearing the "watch" is the MOVE hand's; the fingers doing the tapping
are the ACTION hand's. Callers pass them in that order, so swapping the hand
roles swaps which physical wrist is tapped without touching this module.
"""

import math

from .cursor_fields import INVALID_COORD

TOUCH_THRESHOLD = 0.09


def _landmark_distance(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def _tap_distance(watch_landmarks, pointer_landmarks) -> float | None:
    if watch_landmarks is None or pointer_landmarks is None:
        return None

    watch_wrist = watch_landmarks[0]
    return min(
        _landmark_distance(pointer_landmarks[8], watch_wrist),
        _landmark_distance(pointer_landmarks[12], watch_wrist),
    )


def is_watch_tap(watch_landmarks, pointer_landmarks) -> bool:
    distance = _tap_distance(watch_landmarks, pointer_landmarks)
    return distance is not None and distance <= TOUCH_THRESHOLD


def _clear_solo_hand_gestures(data: dict) -> None:
    data["leftFist"] = False
    data["leftOpenPalm"] = False
    data["leftIndexUp"] = False
    data["leftPeace"] = False
    data["rightFist"] = False
    data["rightOpenPalm"] = False
    data["rightIndexUp"] = False
    data["rightPeace"] = False
    data["openPalm"] = False
    data["isFist"] = False
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
) -> bool:
    distance = _tap_distance(watch_landmarks, pointer_landmarks)
    data["watchTapDistance"] = distance
    data["watchTap"] = distance is not None and distance <= TOUCH_THRESHOLD

    if data["watchTap"]:
        _clear_solo_hand_gestures(data)

    return data["watchTap"]
