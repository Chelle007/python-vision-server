import mediapipe as mp

from vision_server.config import (
    MEDIAPIPE_HAND_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
)


def create_hands(
    *,
    max_num_hands: int = 2,
    model_complexity: int | None = None,
    min_detection_confidence: float | None = None,
):
    """Build a Hands graph. Overrides exist for the live A/B keys in app.py.

    MediaPipe bakes both settings into the graph at construction, so there is no
    way to change them on a live instance — switching means closing this one and
    building another. Callers that leave the overrides as None get the config
    values, which is every caller except the preview toggles.
    """
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        model_complexity=(
            MEDIAPIPE_HAND_MODEL_COMPLEXITY
            if model_complexity is None
            else model_complexity
        ),
        min_detection_confidence=(
            MEDIAPIPE_MIN_DETECTION_CONFIDENCE
            if min_detection_confidence is None
            else min_detection_confidence
        ),
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    )
