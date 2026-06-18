import math


def get_palm_position(landmarks):
    xs = [landmarks[i].x for i in [0, 5, 9, 13, 17]]
    ys = [landmarks[i].y for i in [0, 5, 9, 13, 17]]
    return sum(xs) / 5, sum(ys) / 5


def get_hand_rotation(landmarks):
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]

    # Left/right tilt across knuckles
    dx = pinky_mcp.x - index_mcp.x
    dy = pinky_mcp.y - index_mcp.y
    roll = math.degrees(math.atan2(dy, dx))

    # Up/down hand angle from wrist to middle knuckle
    vx = middle_mcp.x - wrist.x
    vy = middle_mcp.y - wrist.y
    pitch = math.degrees(math.atan2(vy, vx))

    # Rough depth twist using z difference
    yaw = (index_mcp.z - pinky_mcp.z) * 300.0

    return pitch, yaw, roll
