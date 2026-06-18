import mediapipe as mp

from vision_server.config import (
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
)


def create_face_mesh(*, max_num_faces: int = 1):
    mp_face = mp.solutions.face_mesh
    return mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_num_faces,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    )
