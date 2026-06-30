from . import rule


@rule("fist")
def is_fist(landmarks) -> bool:
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]

    return all(landmarks[tip].y > landmarks[pip].y for tip, pip in finger_pairs)
