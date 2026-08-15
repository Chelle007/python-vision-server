"""Hand gesture rule registry."""

HAND_RULES: list[tuple[str, object]] = []


def rule(name: str):
    """Decorator that registers a hand gesture rule by name."""

    def decorator(fn):
        HAND_RULES.append((name, fn))
        return fn

    return decorator


# Import gesture modules so they self-register via @rule
from . import (  # noqa: E402, F401
    fist,
    index_down,
    index_left,
    index_right,
    index_up,
    ok_sign,
    open_palm,
    peace,
    rock_sign,
)

# Each rule above is a view onto the same winner-take-all classifier, so the
# registry can no longer report two gestures for one hand. Callers that want
# every boolean should use classify_hand + rules_from_label, not the registry —
# that path classifies once instead of once per rule.
from .classify import NONE, classify_hand, rules_from_label  # noqa: E402
from .debounce import GestureDebouncer  # noqa: E402

__all__ = [
    "HAND_RULES",
    "rule",
    "NONE",
    "classify_hand",
    "rules_from_label",
    "GestureDebouncer",
]
