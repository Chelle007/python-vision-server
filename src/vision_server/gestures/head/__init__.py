"""Head gesture rule registry."""

HEAD_RULES: list[tuple[str, object]] = []


def rule(name: str):
    """Decorator that registers a head gesture rule by name."""

    def decorator(fn):
        HEAD_RULES.append((name, fn))
        return fn

    return decorator


from . import orientation, tilt  # noqa: E402, F401

__all__ = ["HEAD_RULES", "rule"]
