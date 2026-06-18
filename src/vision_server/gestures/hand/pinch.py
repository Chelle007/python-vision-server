import math

from . import rule


def get_dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


@rule("pinch")
def is_pinching(landmarks) -> bool:
    thumb = landmarks[4]
    index = landmarks[8]
    dist = get_dist(thumb, index)
    return dist < 0.05
