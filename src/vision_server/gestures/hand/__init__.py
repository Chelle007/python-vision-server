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

__all__ = ["HAND_RULES", "rule"]
