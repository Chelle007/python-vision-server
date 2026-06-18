import math


def get_dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def is_pinching(landmarks):
    thumb = landmarks[4]
    index = landmarks[8]
    dist = get_dist(thumb, index)
    return dist < 0.05


def is_open_palm(landmarks):
    palm_x = sum(landmarks[i].x for i in [0, 5, 9, 13, 17]) / 5
    palm_y = sum(landmarks[i].y for i in [0, 5, 9, 13, 17]) / 5

    finger_tips = [8, 12, 16, 20]
    extended_count = 0

    for tip in finger_tips:
        dist = math.sqrt(
            (landmarks[tip].x - palm_x) ** 2 +
            (landmarks[tip].y - palm_y) ** 2
        )

        if dist > 0.11:
            extended_count += 1

    return extended_count >= 2


def get_palm_position(landmarks):
    xs = [landmarks[i].x for i in [0, 5, 9, 13, 17]]
    ys = [landmarks[i].y for i in [0, 5, 9, 13, 17]]
    return sum(xs) / 5, sum(ys) / 5


def is_fist(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_bases = [5, 9, 13, 17]

    folded_count = 0

    for tip, base in zip(finger_tips, finger_bases):
        if landmarks[tip].y > landmarks[base].y:
            folded_count += 1

    return folded_count >= 3


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


def is_peace_sign(landmarks):
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y

    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y

    return index_up and middle_up and ring_down and pinky_down


def is_index_up(landmarks):
    index_up = landmarks[8].y < landmarks[6].y

    middle_down = landmarks[12].y > landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y

    return index_up and middle_down and ring_down and pinky_down
