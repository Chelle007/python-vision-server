import math

from . import rule


@rule("open_palm")
def is_open_palm(landmarks) -> bool:
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    palm_x = sum(landmarks[i].x for i in [0, 5, 9, 13, 17]) / 5
    palm_y = sum(landmarks[i].y for i in [0, 5, 9, 13, 17]) / 5
    thumb_dist = math.sqrt(
        (landmarks[4].x - palm_x) ** 2 + (landmarks[4].y - palm_y) ** 2
    )
    thumb_up = thumb_dist > 0.08

    return index_up and middle_up and ring_up and pinky_up and thumb_up
