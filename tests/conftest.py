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


def make_face_landmarks(overrides=None):
    """Create face mesh landmarks; override sparse indices (e.g. 234, 454)."""
    landmarks = [make_landmark() for _ in range(468)]
    landmarks[234] = make_landmark(x=0.35, y=0.5)
    landmarks[454] = make_landmark(x=0.65, y=0.5)
    if overrides:
        for index, values in overrides.items():
            landmarks[index] = make_landmark(**values)
    return SimpleNamespace(landmark=landmarks)
