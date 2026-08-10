"""Frame-count hysteresis over one hand's gesture label.

The static rules are evaluated fresh every frame and written straight to the
payload, so a single bad landmark frame is a complete gesture event as far as
Unity is concerned. Jump fires on a rising edge, so one flicker frame is one
jump; crouch is a held pose tested per frame, so one dropped frame stands the
player up. Same missing mechanism behind both.

The fix is a commit delay, but it has to be *asymmetric*, because the two
failure modes pull in opposite directions:

* **Entering** a gesture should be slow for one-shots. Jump needs enough
  consecutive frames that a transition pose cannot reach the threshold.
* **Leaving** a gesture should be slow for holds. Crouch must survive several
  consecutive non-peace frames before it releases, so a dropout — or a brief
  MediaPipe loss of the hand entirely — does not pop the player upright.

These are two *separate* counters, and they have to be:

* ``away`` counts consecutive frames that are not the committed label — any of
  them, whatever they are. This is the release condition.
* ``candidate`` counts consecutive frames of one specific incoming label. This
  is the commit condition.

A switch needs both. Collapsing them into one counter looks equivalent and is
not: a hand opening out of a fist does not produce a clean run of ``open_palm``
frames, it produces a mixture of ``open_palm`` and the ambiguous ``none`` while
the fingers straighten. A single counter keyed on the candidate resets every
time that mixture alternates, so the streak never reaches the threshold and the
player keeps walking for as long as they keep trying to stop. Counting "not a
fist" separately from "is an open palm" is what makes releasing a hold depend
on *leaving* it rather than on cleanly arriving somewhere else.

Counting frames rather than seconds is deliberate. Frame time on this pipeline
is neither 30fps nor stable, so a wall-clock window would mean three frames on
a good stretch and one during a stall — exactly when the filtering is needed
most.

Note this filters *flicker*, not *error*. A pose misread the same way for ten
straight frames passes any debouncer happily; that is what the hand-frame work
in ``fingers.py`` is for.
"""

from __future__ import annotations

from vision_server.config import (
    HAND_GESTURE_OFF_FRAMES,
    HAND_GESTURE_ON_FRAMES,
    HAND_GESTURE_DEFAULT_ON_FRAMES,
    HAND_GESTURE_DEFAULT_OFF_FRAMES,
)

from .classify import NONE


class GestureDebouncer:
    """Commit-delayed view of one hand's gesture label.

    One instance per hand role. Sharing an instance between the MOVE and ACTION
    hands would let one hand's counters decide the other's gesture.
    """

    def __init__(self, overrides: dict[str, dict[str, int]] | None = None):
        """``overrides`` is ``{label: {"on": n, "off": n}}``, merged onto the
        defaults. Written as a diff rather than a full replacement table so a
        role that differs in one gesture says only that, instead of restating
        the nine values it agrees with — where a later edit to a shared default
        would silently miss the copy.
        """
        self._on_frames = dict(HAND_GESTURE_ON_FRAMES)
        self._off_frames = dict(HAND_GESTURE_OFF_FRAMES)
        for label, counts in (overrides or {}).items():
            if "on" in counts:
                self._on_frames[label] = counts["on"]
            if "off" in counts:
                self._off_frames[label] = counts["off"]
        self.label = NONE
        self.raw = NONE
        self._candidate = NONE
        self._streak = 0
        self._away = 0

    def reset(self) -> None:
        """Drop all state back to "no gesture", committing immediately.

        Called whenever the hand behind the label changes identity — a new
        locked player, a MOVE/ACTION role swap, or a watch tap suppressing solo
        gestures. Without this a held crouch would survive a player change,
        because the counters know nothing about whose hand they were counting.
        """
        self.label = NONE
        self.raw = NONE
        self._candidate = NONE
        self._streak = 0
        self._away = 0

    def update(self, raw_label: str) -> str:
        """Feed this frame's raw label; return the committed one.

        Pass ``NONE`` on frames where the hand was not seen at all. That keeps
        a tracking dropout on the same footing as a misread pose — both have to
        persist past the off-count before they can end a held gesture — instead
        of releasing it the instant MediaPipe loses the hand.
        """
        self.raw = raw_label

        if raw_label == self.label:
            # Back on the committed label; both partial switches are abandoned.
            self._candidate = self.label
            self._streak = 0
            self._away = 0
            return self.label

        self._away += 1

        if raw_label == self._candidate:
            self._streak += 1
        else:
            self._candidate = raw_label
            self._streak = 1

        released = self._away >= self._off_frames.get(
            self.label, HAND_GESTURE_DEFAULT_OFF_FRAMES
        )
        arrived = self._streak >= self._on_frames.get(
            raw_label, HAND_GESTURE_DEFAULT_ON_FRAMES
        )

        if released:
            if arrived:
                self.label = raw_label
                self._streak = 0
                self._away = 0
            elif self.label != NONE:
                # Leaving a hold and arriving at the next gesture are separate
                # events, and the first must not wait on the second. The player
                # has stopped making a fist, so movement stops now; whether
                # they land on an open palm or on nothing is still being
                # counted out, and the candidate's streak carries over so it
                # commits as soon as it earns its own on-count.
                self.label = NONE
                self._away = 0

        return self.label
