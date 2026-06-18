from . import rule


@rule("peace")
def is_peace_sign(landmarks) -> bool:
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y

    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y

    return index_up and middle_up and ring_down and pinky_down
