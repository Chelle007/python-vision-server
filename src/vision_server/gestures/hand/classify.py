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

Two gestures do not fall out of the four-finger pattern alone. ``rock_sign`` is
just another pattern, but ``thumbs_up`` shares the fist's — all four fingers
curled — so the thumb has to break the tie. See :func:`_is_thumbs_up`.

The individual ``is_*`` rules in this package are now thin views onto this
function, so the registry, the eval harness and the live server can no longer
disagree about what a hand is doing.
"""

from __future__ import annotations

from vision_server.config import (
    INDEX_POINT_CONE,
    THUMB_SPREAD_RATIO,
    THUMB_UP_CLEARANCE,
    THUMB_UP_MIN_PALM,
    THUMB_UP_REACH,
)

from .fingers import (
    FINGERS,
    finger_states,
    hand_frame,
    index_direction,
    thumb_clearance,
    thumb_reach,
    thumb_spread,
)

NONE = "none"

# (index, middle, ring, pinky) -> label. Exhaustive by intent: a pattern absent
# from this table is deliberately not a gesture.
#
# The thumb is not in the pattern, so "fist" here means only "all four fingers
# curled" — a thumbs-up matches it too, and is split back out below.
_PATTERN_TO_LABEL: dict[tuple[bool, bool, bool, bool], str] = {
    (False, False, False, False): "fist",
    (True, False, False, False): "index_up",
    (True, True, False, False): "peace",
    (True, True, True, True): "open_palm",
    (True, False, False, True): "rock_sign",
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
    "thumbs_up",
    "rock_sign",
)


def _is_thumbs_up(landmarks, frame) -> bool:
    """Split a thumbs-up out of the all-fingers-curled pattern.

    Deliberately one-sided: only a clearly raised, clearly outheld thumb wins,
    and everything else stays ``fist``. Fist on the MOVE hand is walk-forward —
    the most-used gesture in the game — so the cost of a wrong answer is very
    lopsided, and an unrecognised thumbs-up (player re-does it) is much cheaper
    than a fist that stops the player walking.

    The palm floor is the same reasoning. ``hand_frame`` only rejects a hand
    with no measurable axis at all (0.02), which leaves a band of badly
    foreshortened hands whose palm length is small enough to inflate every
    ratio divided by it — on the recorded corpus those frames are the entire
    tail, reaching a nonsensical 5.8 palm lengths. Rather than loosen the
    thresholds to cover noise, a hand that cannot be measured properly simply
    is not eligible to be a thumbs-up.

    Two questions, both about distance and neither about direction: is the
    thumb held clear of the curled fingers, and is it actually extended rather
    than merely resting apart from them. Clearance alone fires on a relaxed
    idle hand, which is the single biggest source of spurious triggers.

    Nothing here references which WAY the thumb points. Two earlier versions
    did — how far the tip rose along the hand's axis, and how far it sat off to
    the side — and both rejected obvious thumbs-ups, because the gesture is
    normally made with the fist rolled so the thumb runs nearly perpendicular
    to that axis. Everything angular about the hand was doing harm; a distance
    between two landmarks reads the same at any wrist angle.
    """
    _, palm_len = frame
    if palm_len < THUMB_UP_MIN_PALM:
        return False

    clearance = thumb_clearance(landmarks, frame)
    if clearance is None or clearance < THUMB_UP_CLEARANCE:
        return False

    reach = thumb_reach(landmarks, frame)
    return reach is not None and reach >= THUMB_UP_REACH


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

    # Four curled fingers is the same pattern for both, so the thumb decides.
    if label == "fist" and _is_thumbs_up(landmarks, frame):
        return "thumbs_up"

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
