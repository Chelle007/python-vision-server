import math


def get_palm_position(landmarks):
    xs = [landmarks[i].x for i in [0, 5, 9, 13, 17]]
    ys = [landmarks[i].y for i in [0, 5, 9, 13, 17]]
    return sum(xs) / 5, sum(ys) / 5


def get_index_tip_position(landmarks):
    tip = landmarks[8]
    return tip.x, tip.y


def get_hand_rotation(landmarks, *, mirror: bool = False):
    """Rough pitch/yaw/roll for the hand, in degrees.

    ``mirror`` reflects the hand across the frame's vertical centre line before
    measuring, so a left hand yields the same numbers its mirror image would —
    Unity's rotation mapping then needs no per-hand special case. Only the
    horizontal terms change: reflecting x negates dx and vx. Yaw is read from z,
    which a horizontal reflection leaves untouched, so it is not flipped.
    """
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]
    flip = -1.0 if mirror else 1.0

    # Left/right tilt across knuckles
    dx = (pinky_mcp.x - index_mcp.x) * flip
    dy = pinky_mcp.y - index_mcp.y
    roll = math.degrees(math.atan2(dy, dx))

    # Up/down hand angle from wrist to middle knuckle
    vx = (middle_mcp.x - wrist.x) * flip
    vy = middle_mcp.y - wrist.y
    pitch = math.degrees(math.atan2(vy, vx))

    # Rough depth twist using z difference
    yaw = (index_mcp.z - pinky_mcp.z) * 300.0

    return pitch, yaw, roll
