"""Hand gesture rule registry."""

HAND_RULES: list[tuple[str, object]] = []


def rule(name: str):
    """Decorator that registers a hand gesture rule by name."""

    def decorator(fn):
        HAND_RULES.append((name, fn))
        return fn

    return decorator


# Import gesture modules so they self-register via @rule
from . import fist, index_up, open_palm, peace  # noqa: E402, F401

# Each rule above is a view onto the same winner-take-all classifier, so the
# registry can no longer report two gestures for one hand. Callers that want
# all four booleans should use classify_hand + rules_from_label, not the
# registry — that path classifies once instead of four times.
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
