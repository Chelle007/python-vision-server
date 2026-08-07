import mediapipe as mp

from vision_server.config import (
    MEDIAPIPE_HAND_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
)


def create_hands(*, max_num_hands: int = 2):
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        model_complexity=MEDIAPIPE_HAND_MODEL_COMPLEXITY,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    )
