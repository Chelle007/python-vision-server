"""Per-finger extended/curled state, measured in the hand's own frame.

The original rules compared ``tip.y`` to ``pip.y`` in image space. That is only
meaningful while the fingers point up the screen: rotate the hand toward
horizontal and every tip sits level with its pip, so a flat hand reads as a
fist and the player walks forward without asking. It is also a knife-edge test
— the two y values differ by a hair at the crossover, so landmark noise alone
flips the answer frame to frame.

Both problems come from the same place, so both are fixed here:

* **Rotation** — project ``tip - pip`` onto the hand's own "up" axis (wrist to
  middle-finger MCP) instead of the screen's. A fist is a fist at any wrist
  angle.
* **Scale** — divide by the palm length, so the numbers mean the same thing
  near the camera and far from it.
* **Knife edge** — return the signed margin rather than a bool, and let callers
  apply a dead zone. Inside the dead zone a finger is *ambiguous*, not
  arbitrarily extended or curled, which is what lets `classify` refuse to
  answer instead of guessing a neighbouring gesture.

Ambiguity is a real answer here. A hand caught mid-transition between two poses
should report "I don't know" and let the debouncer hold the previous gesture,
rather than committing to whatever the noise happened to favour.
"""

from __future__ import annotations

import math

from vision_server.config import FINGER_CURL_MARGIN, FINGER_EXTEND_MARGIN

# (tip, pip) per finger — the same landmark pairs the old rules used.
FINGER_JOINTS: dict[str, tuple[int, int]] = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}

FINGERS = tuple(FINGER_JOINTS)

_WRIST = 0
_MIDDLE_MCP = 9
_THUMB_TIP = 4
_INDEX_MCP = 5
_INDEX_TIP = 8
_PALM_POINTS = (0, 5, 9, 13, 17)

# Below this the palm has no measurable length in image space, which happens
# when the hand points straight at (or away from) the camera. The axis is then
# unrecoverable, not merely noisy.
_DEGENERATE_PALM = 0.02


def hand_frame(landmarks) -> tuple[tuple[float, float], float] | None:
    """Return ``(unit_axis, palm_length)`` for the hand, or ``None``.

    The axis runs wrist -> middle MCP, i.e. "up" as the hand itself sees it.
    ``None`` means the hand is foreshortened to the point where no axis can be
    measured; callers should treat that frame as unreadable rather than
    substituting a screen-space guess.
    """
    wrist = landmarks[_WRIST]
    middle_mcp = landmarks[_MIDDLE_MCP]

    ax = middle_mcp.x - wrist.x
    ay = middle_mcp.y - wrist.y
    length = math.hypot(ax, ay)

    if length < _DEGENERATE_PALM:
        return None

    return (ax / length, ay / length), length


def finger_margins(landmarks, frame=None) -> dict[str, float] | None:
    """Signed extension margin per finger, in palm lengths.

    Positive means the tip reaches past its pip along the hand's axis
    (extended); negative means it folds back behind it (curled). Roughly
    ±0.5 at full extension / full curl, so the dead-zone thresholds sit well
    inside the usable range.

    ``frame`` lets a caller that already measured the hand pass it in rather
    than have it recomputed — see :func:`~.classify.classify_hand`, which needs
    it for both the fingers and the thumb.
    """
    if frame is None:
        frame = hand_frame(landmarks)
    if frame is None:
        return None

    (axis_x, axis_y), palm_len = frame

    margins = {}
    for name, (tip_i, pip_i) in FINGER_JOINTS.items():
        tip = landmarks[tip_i]
        pip = landmarks[pip_i]
        projection = (tip.x - pip.x) * axis_x + (tip.y - pip.y) * axis_y
        margins[name] = projection / palm_len

    return margins


def finger_states(landmarks, frame=None) -> dict[str, bool | None] | None:
    """Per-finger ``True`` extended / ``False`` curled / ``None`` ambiguous.

    ``None`` for the whole hand when the frame itself is unreadable.
    """
    margins = finger_margins(landmarks, frame)
    if margins is None:
        return None

    states: dict[str, bool | None] = {}
    for name, margin in margins.items():
        if margin >= FINGER_EXTEND_MARGIN:
            states[name] = True
        elif margin <= -FINGER_CURL_MARGIN:
            states[name] = False
        else:
            states[name] = None

    return states


def index_direction(landmarks) -> tuple[float, float] | None:
    """Unit ``(horiz, vert)`` the index finger points, in SCREEN coordinates.

    ``horiz`` is +1 pointing screen-right, ``vert`` is +1 pointing screen-up.

    Screen space is deliberate, and it is the one measurement in this module
    that is not taken in the hand's own frame. Everything else here answers
    "what shape is the hand", which must survive the wrist being rolled — but
    "point left" is a statement about the room, not about the hand, and a
    player who rolls their wrist while pointing left still means left. Reading
    it in the hand frame would make the answer depend on the wrist angle, which
    is exactly the bug that made two attempts at a thumbs-up unusable, back
    when the inventory was opened with one.

    The frame is mirrored before landmarks are taken (``cv2.flip`` in app.py),
    so this already matches what the player sees: moving the hand to their own
    right moves it screen-right too.
    """
    tip, mcp = landmarks[_INDEX_TIP], landmarks[_INDEX_MCP]

    dx, dy = tip.x - mcp.x, tip.y - mcp.y
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None

    # Image y grows downward, so negate it to make +vert mean "up the screen".
    return dx / length, -dy / length


def pinch_distance(landmarks, frame=None) -> float | None:
    """Gap between the thumb tip and the index tip, in palm lengths.

    This is the confirming term for the OK sign, whose finger pattern (index
    curled, the other three extended) already tells it apart from every other
    gesture in the set. The one pose that shares that pattern is three fingers
    up with the index merely folded down, and the two differ by where the thumb
    is: closing the ring puts it on the index tip, while the folded-index pose
    leaves it by the palm.

    A distance between two landmarks, deliberately, and not an angle or a
    direction. Both points sit at the front of the hand at roughly the same
    depth, so the projection shortens them together and the ratio holds up as
    the wrist rolls — which is exactly what the thumb measurements this
    replaced could not do.

    Divided by the palm length so it does not tighten as the player leans back
    and the hand shrinks in frame.
    """
    if frame is None:
        frame = hand_frame(landmarks)
    if frame is None:
        return None

    _, palm_len = frame
    thumb, index = landmarks[_THUMB_TIP], landmarks[_INDEX_TIP]

    return math.hypot(thumb.x - index.x, thumb.y - index.y) / palm_len


def thumb_spread(landmarks, frame=None) -> float | None:
    """Thumb-tip distance from the palm centre, in palm lengths.

    The old open-palm rule used a fixed 0.08 in normalised image coordinates,
    which tightens as the player leans back and the hand shrinks. Dividing by
    the palm length removes the dependence on how far away they are sitting.

    Returned as a ratio rather than a bool so the threshold stays callable from
    a calibration script — that is how the current value was chosen.
    """
    if frame is None:
        frame = hand_frame(landmarks)
    if frame is None:
        return None

    _, palm_len = frame

    palm_x = sum(landmarks[i].x for i in _PALM_POINTS) / len(_PALM_POINTS)
    palm_y = sum(landmarks[i].y for i in _PALM_POINTS) / len(_PALM_POINTS)
    thumb = landmarks[_THUMB_TIP]

    return math.hypot(thumb.x - palm_x, thumb.y - palm_y) / palm_len
