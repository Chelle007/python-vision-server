from vision_server.config import HEAD_TILT_THRESHOLD

from . import rule


@rule("tilt")
def detect_head_tilt(face_landmarks) -> dict[str, bool]:
    """Detect head roll (ear toward shoulder) from cheek landmarks."""
    left = face_landmarks.landmark[234]
    right = face_landmarks.landmark[454]

    dx = right.x - left.x
    dy = right.y - left.y

    if abs(dx) < 1e-6:
        return {"tilt_left": False, "tilt_right": False}

    roll_norm = dy / abs(dx)

    return {
        "tilt_left": roll_norm < -HEAD_TILT_THRESHOLD,
        "tilt_right": roll_norm > HEAD_TILT_THRESHOLD,
    }
