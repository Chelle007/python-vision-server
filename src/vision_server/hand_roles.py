"""Which physical hand plays which role (MOVE vs ACTION).

The server has always hard-wired the roles: physical left hand = MOVE (fist /
index / peace), physical right hand = ACTION (grab, cursor, fist rotation and
the dynamic-gesture LSTM). Left-handed players want that swapped.

The swap happens here rather than in Unity, and it deliberately does NOT rename
the UDP keys. ``leftFist`` and friends become *role* fields — "the MOVE hand's
fist" and "the ACTION hand's fist" — so every existing Unity consumer keeps
working and only the hand filling them changes. True physical handedness is
still available per hand in the ``hands`` array.

Same shape as :class:`~vision_server.puzzle_gate.PuzzleGate`: one piece of state
behind one setter, so today's keypress and tomorrow's Unity settings packet are
just two callers of :meth:`HandRoles.set_action_hand`.
"""

from __future__ import annotations

from vision_server.config import ACTION_HAND_DEFAULT, MIRROR_LEFT_HAND_FOR_LSTM

SIDES = ("left", "right")


def _other(side: str) -> str:
    return "left" if side == "right" else "right"


class HandRoles:
    """Maps the MOVE and ACTION roles onto physical hands."""

    def __init__(
        self,
        action_hand: str = ACTION_HAND_DEFAULT,
        *,
        mirror_left: bool = MIRROR_LEFT_HAND_FOR_LSTM,
    ):
        self.action_hand = self._normalise(action_hand)
        self.mirror_left = bool(mirror_left)
        self.source = "default"

    @staticmethod
    def _normalise(side: str) -> str:
        side = str(side).strip().lower()
        if side not in SIDES:
            raise ValueError(f"action hand must be one of {SIDES}, got {side!r}")
        return side

    @property
    def move_hand(self) -> str:
        return _other(self.action_hand)

    @property
    def mirror_action_hand(self) -> bool:
        """True when the action hand's landmarks need mirroring for the LSTM."""
        return self.mirror_left and self.action_hand == "left"

    def set_action_hand(self, side: str, *, source: str = "unknown") -> bool:
        """Set the action hand; return True only if this call actually moved it.

        Idempotent by design: Unity will resend the setting on every packet, and
        a repeat must not flush the LSTM buffer mid-gesture.
        """
        side = self._normalise(side)
        self.source = source
        if side == self.action_hand:
            return False
        self.action_hand = side
        return True

    def swap(self, *, source: str = "keyboard") -> bool:
        """Flip the roles. Always reports True since state always changes."""
        return self.set_action_hand(self.move_hand, source=source)

    def split(self, lock) -> tuple[object | None, object | None]:
        """Return ``(move_hand, action_hand)`` candidates from a PlayerLock result."""
        if self.action_hand == "right":
            return lock.left, lock.right
        return lock.right, lock.left

    def apply_to_payload(self, data: dict) -> None:
        """Echo the mapping so Unity can confirm what the server is using."""
        data["action_hand"] = self.action_hand
        data["move_hand"] = self.move_hand
        data["hand_roles_source"] = self.source
