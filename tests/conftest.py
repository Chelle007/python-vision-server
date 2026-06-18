"""Mock MediaPipe landmark for testing."""

from types import SimpleNamespace


def make_landmark(x=0.5, y=0.5, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def make_landmarks(overrides=None):
    """Create 21 default landmarks; override by index with overrides dict."""
    landmarks = [make_landmark() for _ in range(21)]
    if overrides:
        for index, values in overrides.items():
            landmarks[index] = make_landmark(**values)
    return landmarks
