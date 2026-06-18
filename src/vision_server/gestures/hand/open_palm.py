import math

from . import rule


@rule("open_palm")
def is_open_palm(landmarks) -> bool:
    palm_x = sum(landmarks[i].x for i in [0, 5, 9, 13, 17]) / 5
    palm_y = sum(landmarks[i].y for i in [0, 5, 9, 13, 17]) / 5

    finger_tips = [8, 12, 16, 20]
    extended_count = 0

    for tip in finger_tips:
        dist = math.sqrt(
            (landmarks[tip].x - palm_x) ** 2 + (landmarks[tip].y - palm_y) ** 2
        )

        if dist > 0.11:
            extended_count += 1

    return extended_count >= 2
