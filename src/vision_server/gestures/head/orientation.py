from . import rule


@rule("orientation")
def compute_head_orientation(face_landmarks) -> dict[str, float]:
    """Compute head yaw and pitch from raw face mesh landmarks."""
    nose = face_landmarks.landmark[1]
    left_side = face_landmarks.landmark[234]
    right_side = face_landmarks.landmark[454]
    top = face_landmarks.landmark[10]
    bottom = face_landmarks.landmark[152]

    head_yaw = 0.0
    head_pitch = 0.0

    face_width = right_side.x - left_side.x
    nose_offset_x = nose.x - left_side.x

    if face_width > 0:
        head_yaw = round(((nose_offset_x / face_width) - 0.5) * 2, 3)

    face_height = bottom.y - top.y
    nose_offset_y = nose.y - top.y

    if face_height > 0:
        head_pitch = round(((nose_offset_y / face_height) - 0.5) * 2, 3)

    return {"head_yaw": head_yaw, "head_pitch": head_pitch}
