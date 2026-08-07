"""Gate for expensive per-frame inference (currently the dynamic-gesture LSTM).

Puzzle gestures only matter while the player is actually solving a puzzle, but
the LSTM used to run on every frame a right hand was visible — roughly 30 Keras
inferences a second on top of MediaPipe hands + face mesh. That sustained load
is the leading suspect behind the soft webcam grab failures ``camera.py`` now
has to recover from.

The gate is one piece of state behind one setter, so the trigger source can
change without touching the LSTM call site. Today a keypress flips it; later
Unity flips it over UDP when a puzzle starts and ends. Both are just callers of
:meth:`PuzzleGate.set_active`.
"""

from __future__ import annotations

from vision_server.config import PUZZLE_GATE_DEFAULT_ACTIVE


class PuzzleGate:
    """Whether puzzle-solving gestures should currently be classified."""

    def __init__(self, active: bool = PUZZLE_GATE_DEFAULT_ACTIVE):
        self.active = bool(active)
        self.source = "default"

    def set_active(self, value: bool, *, source: str = "unknown") -> bool:
        """Set gate state; return True only if this call actually flipped it.

        Idempotent by design: Unity will likely resend the same state on every
        packet, and a repeat must not thrash the LSTM's action-hold timers.
        """
        value = bool(value)
        self.source = source
        if value == self.active:
            return False
        self.active = value
        return True

    def toggle(self, *, source: str = "keyboard") -> bool:
        """Flip the gate. Always reports True since state always changes."""
        return self.set_active(not self.active, source=source)

    @property
    def status(self) -> str:
        return "active" if self.active else "idle"

    def apply_to_payload(self, data: dict) -> None:
        """Echo gate state to Unity so it can confirm what the server sees."""
        data["puzzle_active"] = self.active
        data["puzzle_gate_source"] = self.source
