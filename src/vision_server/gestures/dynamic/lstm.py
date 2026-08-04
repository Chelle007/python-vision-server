import time
from collections import deque

import numpy as np

from vision_server.config import (
    LSTM_ACTION_MAX_HOLD_S,
    LSTM_BUFFER_SIZE,
    LSTM_CONFIDENCE_THRESHOLD,
    LSTM_HAND_MISS_CLEAR_S,
    MODEL_PATH,
)
from vision_server.features import flatten_landmarks
from vision_server.gestures.dynamic.labels import CLASSES


def _load_keras_model(model_path):
    """Load model only if TensorFlow is installed (avoids import error when TF is omitted)."""
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        print(
            "TensorFlow not installed — GestureLSTM runs without inference. "
            "Install tensorflow in this venv or use a separate training venv."
        )
        return None
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


class GestureLSTM:
    def __init__(
        self,
        model_path=MODEL_PATH,
        buffer_size=LSTM_BUFFER_SIZE,
        confidence_threshold=LSTM_CONFIDENCE_THRESHOLD,
        hand_miss_clear_s=LSTM_HAND_MISS_CLEAR_S,
        action_max_hold_s=LSTM_ACTION_MAX_HOLD_S,
    ):
        self.model = _load_keras_model(model_path)
        if self.model is not None:
            print(f"AI Brain '{model_path}' loaded successfully!")

        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.hand_miss_clear_s = hand_miss_clear_s
        self.action_max_hold_s = action_max_hold_s
        self._hand_miss_started_at = None
        self.last_output = "Idle"
        self.classes = CLASSES
        self._action_class = None
        self._action_started_at = 0.0
        self._action_expired = False

    def register_hand_seen(self):
        self._hand_miss_started_at = None

    def flush(self):
        self.frame_buffer.clear()
        self._hand_miss_started_at = None
        self.last_output = "Idle"
        self._reset_action_hold()

    def _reset_action_hold(self):
        self._action_class = None
        self._action_started_at = 0.0
        self._action_expired = False

    def register_hand_lost(self):
        """Brief tracking gaps keep the buffer; only long gaps wipe it."""
        now = time.monotonic()
        if self._hand_miss_started_at is None:
            self._hand_miss_started_at = now
            return

        if now - self._hand_miss_started_at >= self.hand_miss_clear_s:
            self.flush()

    def get_overlay_label(self):
        """Label for on-screen HUD. Buffer warmup is silent Idle."""
        return self.last_output

    def _apply_action_hold(self, raw_label: str) -> str:
        """Cap non-Idle outputs so one-shots do not stick after the motion."""
        now = time.monotonic()

        if raw_label == "Idle":
            self._reset_action_hold()
            return "Idle"

        if self._action_class != raw_label:
            self._action_class = raw_label
            self._action_started_at = now
            self._action_expired = False
            return raw_label

        if self._action_expired:
            return "Idle"

        if now - self._action_started_at >= self.action_max_hold_s:
            self._action_expired = True
            return "Idle"

        return raw_label

    def predict(self, landmarks):
        if self.model is None:
            return "No Model"

        self.register_hand_seen()
        self.frame_buffer.append(flatten_landmarks(landmarks))

        if len(self.frame_buffer) < self.buffer_size:
            return self.last_output

        input_data = np.array([list(self.frame_buffer)])
        prediction = self.model.predict(input_data, verbose=0)

        class_id = int(np.argmax(prediction))
        confidence = float(prediction[0][class_id])

        if confidence > self.confidence_threshold:
            raw_label = self.classes[class_id]
        else:
            raw_label = "Idle"

        self.last_output = self._apply_action_hold(raw_label)
        return self.last_output
