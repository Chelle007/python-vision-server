"""Mock MediaPipe landmark for testing."""

import math
from types import SimpleNamespace

# --- Synthetic hand geometry ----------------------------------------------
# The gesture rules measure finger extension along the hand's own axis and
# normalise by palm length, so fixtures need a hand with real proportions
# rather than a list of hand-picked y values. Everything below is expressed in
# palm lengths (wrist -> middle MCP = 1.0) in a local frame where "up" runs
# along the fingers, then rotated and scaled into image coordinates.

# MCP knuckle positions: (across the palm, along the palm).
_MCP = {
    "index": (-0.32, 0.98),
    "middle": (0.0, 1.0),
    "ring": (0.30, 0.96),
    "pinky": (0.56, 0.88),
}
# Finger lengths, again in palm lengths. The pinky is the shortest, so it sets
# the worst case for how far outside the dead zone a real gesture lands.
_FINGER_LENGTH = {"index": 0.95, "middle": 1.02, "ring": 0.95, "pinky": 0.75}
# Landmark indices per finger: (mcp, pip, dip, tip).
_FINGER_INDICES = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_PIP_ALONG = 0.40
# Tip travel per unit of extension. extension=+1 puts the tip a full finger
# out; -1 curls it back past the knuckle into the palm.
_TIP_SWING = 0.50

_DEFAULT_PALM_LENGTH = 0.12
_WRIST_AT = (0.5, 0.5)


def make_landmark(x=0.5, y=0.5, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def hand_pose(
    *,
    index=False,
    middle=False,
    ring=False,
    pinky=False,
    thumb_out=False,
    rotation_deg=0.0,
    scale=1.0,
    index_extension=None,
    middle_extension=None,
    ring_extension=None,
    pinky_extension=None,
):
    """Build 21 landmarks for a hand in a given pose.

    Each finger flag picks full extension or a full curl. The matching
    ``*_extension`` argument overrides it with a value in ``[-1, 1]``, where
    small magnitudes land inside the dead zone — that is how a half-committed
    finger, i.e. a hand caught between two poses, is expressed.

    ``rotation_deg`` turns the whole hand about the wrist and ``scale`` moves it
    toward or away from the camera; neither should change a classification.
    ``scale=0`` collapses the palm entirely, standing in for a hand pointed
    straight down the camera axis.
    """
    extensions = {
        "index": index_extension if index_extension is not None else (1.0 if index else -1.0),
        "middle": middle_extension if middle_extension is not None else (1.0 if middle else -1.0),
        "ring": ring_extension if ring_extension is not None else (1.0 if ring else -1.0),
        "pinky": pinky_extension if pinky_extension is not None else (1.0 if pinky else -1.0),
    }

    theta = math.radians(rotation_deg)
    palm_length = _DEFAULT_PALM_LENGTH * scale
    # Image y grows downward, so at zero rotation "along the palm" is -y.
    up = (math.sin(theta), -math.cos(theta))
    across = (math.cos(theta), math.sin(theta))

    def place(a, u):
        return make_landmark(
            x=_WRIST_AT[0] + (a * across[0] + u * up[0]) * palm_length,
            y=_WRIST_AT[1] + (a * across[1] + u * up[1]) * palm_length,
        )

    landmarks = [make_landmark() for _ in range(21)]
    landmarks[0] = place(0.0, 0.0)

    for name, (mcp_i, pip_i, dip_i, tip_i) in _FINGER_INDICES.items():
        across_at, mcp_along = _MCP[name]
        length = _FINGER_LENGTH[name]
        pip_along = mcp_along + _PIP_ALONG * length
        tip_along = mcp_along + (_PIP_ALONG + _TIP_SWING * extensions[name]) * length

        landmarks[mcp_i] = place(across_at, mcp_along)
        landmarks[pip_i] = place(across_at, pip_along)
        landmarks[dip_i] = place(across_at, (pip_along + tip_along) / 2.0)
        landmarks[tip_i] = place(across_at, tip_along)

    thumb_tip = (-0.85, 1.05) if thumb_out else (-0.10, 0.85)
    landmarks[1] = place(-0.30, 0.25)
    landmarks[2] = place(-0.52, 0.55)
    landmarks[3] = place((-0.52 + thumb_tip[0]) / 2.0, (0.55 + thumb_tip[1]) / 2.0)
    landmarks[4] = place(*thumb_tip)

    return landmarks


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
