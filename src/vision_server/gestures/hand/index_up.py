from . import rule


@rule("index_up")
def is_index_up(landmarks) -> bool:
    index_up = landmarks[8].y < landmarks[6].y

    middle_down = landmarks[12].y > landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y

    return index_up and middle_down and ring_down and pinky_down
