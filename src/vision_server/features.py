"""Shared landmark feature extraction for record, train, and inference."""


def flatten_landmarks(landmarks) -> list[float]:
    """Flatten 21 MediaPipe landmarks into a 63-value vector (x, y, z each)."""
    frame_data = []
    for lm in landmarks:
        frame_data.extend([lm.x, lm.y, lm.z])
    return frame_data
