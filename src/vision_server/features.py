"""Shared landmark feature extraction for record, train, and inference."""


def flatten_landmarks(landmarks, *, mirror: bool = False) -> list[float]:
    """Flatten 21 MediaPipe landmarks into a 63-value vector (x, y, z each).

    ``mirror`` reflects the hand across the vertical centre line of the frame
    (x -> 1-x). MediaPipe's 21-landmark topology is identical for both hands, so
    a mirrored left hand is geometrically a right hand — that is what lets the
    right-hand-trained LSTM read left-hand input.
    """
    frame_data = []
    for lm in landmarks:
        x = 1.0 - lm.x if mirror else lm.x
        frame_data.extend([x, lm.y, lm.z])
    return frame_data
