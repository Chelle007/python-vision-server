"""One hand, one gesture label.

The four static rules used to be independent predicates, each free to answer
yes on the same frame, and each with no way to say "none of these". Two things
fell out of that:

* A pose could satisfy two rules at once, leaving Unity to break the tie by
  whichever field it happened to read first.
* Every pose had to land in *some* bucket, so the fist -> peace transition —
  where the index extends a frame or two before the middle — passes through a
  textbook index-up pose and fires a jump on the way into a crouch.

Collapsing them into one winner-take-all label fixes both. The finger pattern
is read once, and only the exact patterns below name a gesture; anything else,
including any pattern containing an ambiguous finger, is ``NONE``. A transition
is no longer a gesture — it is the absence of one, which is what the debouncer
needs in order to keep holding the previous label.

Every gesture here falls out of the four-finger pattern, and where a pattern is
not the whole story the thumb only ever *confirms* the label the pattern already
named — it never has to choose between two of them. That is a deliberate
constraint on the gesture set rather than an accident of it: the previous
inventory gesture was a thumbs-up, which shares the fist's all-curled pattern
and so left the thumb arbitrating between "open the menu" and "walk forward".
No measurement of the thumb was reliable enough to arbitrate anything. The OK
sign that replaced it has a pattern of its own.

The individual ``is_*`` rules in this package are now thin views onto this
function, so the registry, the eval harness and the live server can no longer
disagree about what a hand is doing.
"""

from __future__ import annotations

from vision_server.config import (
    INDEX_POINT_CONE,
    OK_SIGN_MIN_PALM,
    OK_SIGN_PINCH,
    THUMB_SPREAD_RATIO,
)

from .fingers import (
    FINGERS,
    finger_states,
    hand_frame,
    index_direction,
    pinch_distance,
    thumb_spread,
)

NONE = "none"

# (index, middle, ring, pinky) -> label. Exhaustive by intent: a pattern absent
# from this table is deliberately not a gesture.
#
# The thumb is not in the pattern. Both entries whose pose depends on it —
# open_palm and ok_sign — are confirmed against it below, and failing that
# check reports NONE rather than falling through to another gesture.
_PATTERN_TO_LABEL: dict[tuple[bool, bool, bool, bool], str] = {
    (False, False, False, False): "fist",
    (True, False, False, False): "index_up",
    (True, True, False, False): "peace",
    (True, True, True, True): "open_palm",
    (True, False, False, True): "rock_sign",
    (False, True, True, True): "ok_sign",
}

# Payload/rule names, in registry order. Every label here is emitted as its own
# boolean for Unity; NONE emits all-False.
LABELS = (
    "fist",
    "open_palm",
    "peace",
    "index_up",
    "index_left",
    "index_right",
    "index_down",
    "ok_sign",
    "rock_sign",
)


def _is_ok_sign(landmarks, frame) -> bool:
    """Confirm the OK sign's pattern by checking the ring is actually closed.

    The pattern — index curled, middle/ring/pinky extended — is already unique
    to this gesture. What it does not distinguish is an OK sign from three
    fingers held up with the index simply folded away, so the thumb is asked
    one question: is it touching the index tip.

    The palm floor guards the direction that costs a wrong menu. A hand aimed
    at the camera projects its landmarks on top of one another, which shrinks
    an unpinched gap just as effectively as closing it would; below the floor
    the geometry cannot tell those apart, so the hand is not eligible.
    """
    _, palm_len = frame
    if palm_len < OK_SIGN_MIN_PALM:
        return False

    pinch = pinch_distance(landmarks, frame)
    return pinch is not None and pinch <= OK_SIGN_PINCH


def _pointing_label(landmarks) -> str:
    """Split one extended index into up / down / left / right by where it points.

    Cones rather than a nearest-axis vote, so the diagonals fall into a dead
    zone and report ``NONE`` instead of being forced into whichever neighbour
    happens to win by a hair. That mirrors what the finger dead zone already
    does one level down: a hand caught between two poses is in neither.

    ``INDEX_POINT_CONE`` above cos(45 degrees) = 0.707 is what keeps the four
    cones from overlapping, so no direction can satisfy two of them.
    """
    direction = index_direction(landmarks)
    if direction is None:
        return NONE

    horiz, vert = direction

    if vert >= INDEX_POINT_CONE:
        return "index_up"
    if horiz >= INDEX_POINT_CONE:
        return "index_right"
    if horiz <= -INDEX_POINT_CONE:
        return "index_left"
    if vert <= -INDEX_POINT_CONE:
        return "index_down"

    # Pointing diagonally — no gesture claims it.
    return NONE


def classify_hand(landmarks) -> str:
    """Return the single gesture label for one hand, or ``NONE``."""
    # Measured once and threaded through: the fingers and the thumb are both
    # read relative to the same axis, and deriving it twice would also run the
    # unreadable-hand check twice.
    frame = hand_frame(landmarks)
    if frame is None:
        return NONE

    states = finger_states(landmarks, frame)
    pattern = tuple(states[name] for name in FINGERS)
    if any(state is None for state in pattern):
        # At least one finger is inside the dead zone: the hand is between
        # poses, not in one.
        return NONE

    label = _PATTERN_TO_LABEL.get(pattern, NONE)

    # An open hand with the thumb tucked in is a distinct pose from a spread
    # palm, and the release gesture means the spread one.
    if label == "open_palm" and thumb_spread(landmarks, frame) < THUMB_SPREAD_RATIO:
        return NONE

    # The pattern names the OK sign; the closed ring is what confirms it.
    if label == "ok_sign" and not _is_ok_sign(landmarks, frame):
        return NONE

    # One extended index is four gestures, told apart by where it points.
    if label == "index_up":
        return _pointing_label(landmarks)

    return label


def rules_from_label(label: str) -> dict[str, bool]:
    """Expand a label into the per-gesture booleans the payload carries.

    At most one is ever True — that guarantee is the point of the label.

    ``classify_hand`` then this is the single path from landmarks to payload
    booleans, shared by the live server, the annotated-video export and the
    segment scorer. It used to be copy-pasted into all three, so eval runs
    could silently score different behaviour than the one that shipped.
    """
    return {name: name == label for name in LABELS}
