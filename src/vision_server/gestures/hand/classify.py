"""One hand, one gesture label.

The four static rules used to be independent predicates, each free to answer
yes on the same frame, and each with no way to say "none of these". Two things
fell out of that:

* A pose could satisfy two rules at once, leaving Unity to break the tie by
  whichever field it happened to read first.
* Every pose had to land in *some* bucket, so the fist -> peace transition —
  where the index extends a frame or two before the middle — passes through a
  textbook index-up pose and fires a jump on the way into a crouch.

Collapsing the four into one winner-take-all label fixes both. The finger
pattern is read once, and only the four exact patterns below name a gesture;
anything else, including any pattern containing an ambiguous finger, is
``NONE``. A transition is no longer a gesture — it is the absence of one, which
is what the debouncer needs in order to keep holding the previous label.

The individual ``is_*`` rules in this package are now thin views onto this
function, so the registry, the eval harness and the live server can no longer
disagree about what a hand is doing.
"""

from __future__ import annotations

from vision_server.config import THUMB_SPREAD_RATIO

from .fingers import FINGERS, finger_states, hand_frame, thumb_spread

NONE = "none"

# (index, middle, ring, pinky) -> label. Exhaustive by intent: a pattern absent
# from this table is deliberately not a gesture.
_PATTERN_TO_LABEL: dict[tuple[bool, bool, bool, bool], str] = {
    (False, False, False, False): "fist",
    (True, False, False, False): "index_up",
    (True, True, False, False): "peace",
    (True, True, True, True): "open_palm",
}

# Payload/rule names, in registry order. Every label here is emitted as its own
# boolean for Unity; NONE emits all-False.
LABELS = ("fist", "open_palm", "peace", "index_up")


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
