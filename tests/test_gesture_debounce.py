"""Commit delay on static gesture labels.

Each test names the in-game symptom it prevents; the thresholds themselves live
in config, so the tests pin the *behaviour* (a transient cannot commit, a
dropout cannot release) rather than the specific frame counts.
"""

from vision_server.config import (
    HAND_GESTURE_OFF_FRAMES,
    HAND_GESTURE_ON_FRAMES,
    MOVE_GESTURE_OVERRIDES,
)
from vision_server.gestures.hand import NONE, GestureDebouncer


def feed(debouncer, label, frames):
    """Push ``label`` for ``frames`` frames; return the last committed label."""
    out = debouncer.label
    for _ in range(frames):
        out = debouncer.update(label)
    return out


def settle(debouncer, label):
    """Hold a label long enough that it is definitely committed."""
    feed(debouncer, label, max(HAND_GESTURE_ON_FRAMES.values()) + max(HAND_GESTURE_OFF_FRAMES.values()))
    assert debouncer.label == label
    return debouncer


def test_starts_with_no_gesture():
    assert GestureDebouncer().label == NONE


def test_single_frame_of_index_up_does_not_commit():
    """One flicker frame used to be one jump."""
    debouncer = GestureDebouncer()
    assert debouncer.update("index_up") == NONE


def test_index_up_commits_once_held():
    debouncer = GestureDebouncer()
    assert feed(debouncer, "index_up", HAND_GESTURE_ON_FRAMES["index_up"]) == "index_up"


def test_transition_through_index_up_does_not_fire_jump():
    """fist -> peace passes through a real index-up pose on the way.

    The index extends a frame or two before the middle catches up, which is why
    starting a crouch while walking used to make the player jump.
    """
    debouncer = settle(GestureDebouncer(), "fist")

    # The transition is shorter than index_up's on-count.
    feed(debouncer, "index_up", HAND_GESTURE_ON_FRAMES["index_up"] - 1)
    assert debouncer.label != "index_up"

    settle(debouncer, "peace")
    assert debouncer.label == "peace"


def test_single_dropped_frame_does_not_release_crouch():
    """One bad frame mid-crouch used to stand the player up."""
    debouncer = settle(GestureDebouncer(), "peace")
    assert debouncer.update(NONE) == "peace"


def test_brief_hand_loss_does_not_release_crouch():
    """A tracking dropout is fed as NONE and must be counted out like any other."""
    debouncer = settle(GestureDebouncer(), "peace")
    assert feed(debouncer, NONE, HAND_GESTURE_OFF_FRAMES["peace"] - 1) == "peace"


def test_sustained_release_does_end_crouch():
    debouncer = settle(GestureDebouncer(), "peace")
    assert feed(debouncer, NONE, HAND_GESTURE_OFF_FRAMES["peace"]) == NONE


def test_intermittent_noise_never_accumulates():
    """Alternating frames must not add up to a commit.

    The counter tracks *consecutive* frames, so a gesture that never holds for
    long enough can flicker forever without ever taking effect.
    """
    debouncer = settle(GestureDebouncer(), "fist")
    for _ in range(20):
        debouncer.update("index_up")
        debouncer.update("fist")
    assert debouncer.label == "fist"


def test_leaving_a_hold_obeys_the_longer_of_the_two_counts():
    """peace -> index_up must satisfy peace's release AND index_up's commit."""
    debouncer = settle(GestureDebouncer(), "peace")
    required = max(
        HAND_GESTURE_ON_FRAMES["index_up"], HAND_GESTURE_OFF_FRAMES["peace"]
    )

    assert feed(debouncer, "index_up", required - 1) == "peace"
    assert debouncer.update("index_up") == "index_up"


def test_movement_starts_promptly():
    """Holds enter fast — a walk that lagged would be worse than the bug."""
    debouncer = GestureDebouncer()
    assert HAND_GESTURE_ON_FRAMES["fist"] <= HAND_GESTURE_ON_FRAMES["index_up"]
    assert feed(debouncer, "fist", HAND_GESTURE_ON_FRAMES["fist"]) == "fist"


def test_a_messy_release_still_stops_movement():
    """Opening a fist to stop walking must work, however ragged it looks.

    The fingers do not all straighten on the same frame, so the classifier
    emits a mixture of open_palm and the ambiguous none on the way out. An
    earlier version counted only consecutive frames of one candidate label, so
    that mixture reset the counter every other frame and the player kept
    walking for as long as they kept trying to stop.
    """
    debouncer = settle(GestureDebouncer(), "fist")
    messy = ["none", "open_palm", "none", "open_palm", "open_palm", "none"]

    released_after = None
    for i, raw in enumerate(messy, 1):
        if debouncer.update(raw) != "fist":
            released_after = i
            break

    assert released_after is not None, "never released"
    assert released_after <= HAND_GESTURE_OFF_FRAMES["fist"]


def test_release_counts_any_non_matching_frame():
    """The off-count measures leaving the gesture, not arriving somewhere."""
    debouncer = settle(GestureDebouncer(), "peace")
    alternating = ["none", "fist"] * HAND_GESTURE_OFF_FRAMES["peace"]

    for raw in alternating:
        if debouncer.update(raw) != "peace":
            return
    raise AssertionError("crouch never released under alternating input")


def test_arriving_still_needs_its_own_streak():
    """Releasing a hold must not let a one-frame gesture commit immediately.

    Once the off-count is satisfied the label can leave, but jump still has to
    earn its on-count — otherwise the release would hand a free commit to
    whatever noise happened to be on screen that frame.
    """
    debouncer = settle(GestureDebouncer(), "peace")
    feed(debouncer, NONE, HAND_GESTURE_OFF_FRAMES["peace"])
    assert debouncer.label == NONE

    for _ in range(HAND_GESTURE_ON_FRAMES["index_up"] - 1):
        assert debouncer.update("index_up") == NONE
    assert debouncer.update("index_up") == "index_up"


def test_reset_drops_a_held_gesture():
    """A new locked player must not inherit the last one's crouch."""
    debouncer = settle(GestureDebouncer(), "peace")
    debouncer.reset()
    assert debouncer.label == NONE
    assert debouncer.raw == NONE


def test_reset_also_clears_a_part_way_streak():
    debouncer = GestureDebouncer()
    feed(debouncer, "index_up", HAND_GESTURE_ON_FRAMES["index_up"] - 1)
    debouncer.reset()
    assert debouncer.update("index_up") == NONE


def test_raw_reports_the_unfiltered_label():
    """The HUD and the packet need to show what was seen, not just what stuck."""
    debouncer = settle(GestureDebouncer(), "fist")
    debouncer.update("peace")
    assert debouncer.raw == "peace"
    assert debouncer.label == "fist"


def test_move_hand_walks_and_stops_faster_than_the_action_hand():
    """Same gesture, opposite needs: leftFist walks, rightFist sits.

    Walking is held and released constantly and every frame of delay is felt as
    the player sliding past their mark, so the MOVE hand starts instantly.
    Sitting is a one-shot fired on a rising edge, so the ACTION hand keeps the
    delay. Pinned here because the two tables live apart in config and it is
    not obvious from either one that they are meant to differ.
    """
    move = GestureDebouncer(MOVE_GESTURE_OVERRIDES)
    action = GestureDebouncer()

    assert move.update("fist") == "fist"
    assert action.update("fist") == NONE

    assert MOVE_GESTURE_OVERRIDES["fist"]["off"] < HAND_GESTURE_OFF_FRAMES["fist"]


def test_move_hand_still_tolerates_a_dropped_frame():
    """Faster does not mean unfiltered — walking must survive one bad frame."""
    move = GestureDebouncer(MOVE_GESTURE_OVERRIDES)
    feed(move, "fist", 4)
    assert move.update(NONE) == "fist"


def test_move_hand_keeps_the_one_shot_and_hold_protections():
    """Only fist was loosened; everything else is inherited, not restated."""
    assert set(MOVE_GESTURE_OVERRIDES) == {"fist"}


def test_custom_thresholds_are_honoured():
    debouncer = GestureDebouncer({"fist": {"on": 1, "off": 3}})
    assert debouncer.update("fist") == "fist"
    assert feed(debouncer, NONE, 2) == "fist"
    assert debouncer.update(NONE) == NONE
