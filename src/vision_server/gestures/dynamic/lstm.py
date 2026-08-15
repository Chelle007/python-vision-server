import time
from collections import deque

import numpy as np

from vision_server.config import (
    LSTM_ACTION_MAX_HOLD_S,
    LSTM_BUFFER_SIZE,
    LSTM_CONFIDENCE_THRESHOLD,
    LSTM_GRAB_PULL_LEVER_THRESHOLD,
    LSTM_HAND_MISS_CLEAR_S,
    LSTM_MIRRORED_CLASS_SWAP,
    LSTM_PUZZLE_ALLOWED_CLASSES,
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
        grab_pull_threshold=LSTM_GRAB_PULL_LEVER_THRESHOLD,
        hand_miss_clear_s=LSTM_HAND_MISS_CLEAR_S,
        action_max_hold_s=LSTM_ACTION_MAX_HOLD_S,
    ):
        self.model = _load_keras_model(model_path)
        if self.model is not None:
            print(f"AI Brain '{model_path}' loaded successfully!")

        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.grab_pull_threshold = grab_pull_threshold
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

    def clamp_label(self, label: str) -> str:
        """Only real model classes reach Unity; "No Model" etc. become Idle."""
        return label if label in self.classes else "Idle"

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

    def _decide_label(
        self,
        class_id: int,
        confidence: float,
        *,
        grabbing: bool = False,
        mirror: bool = False,
    ) -> str:
        """Map argmax + score to a class. Grab only lowers the Pull_Lever floor."""
        raw_label = self.classes[class_id]
        if mirror:
            # Mirroring reversed the on-screen turn direction; undo it on the
            # label so a class always means the same motion the player made.
            raw_label = LSTM_MIRRORED_CLASS_SWAP.get(raw_label, raw_label)

        threshold = self.confidence_threshold
        if grabbing and raw_label == "Pull_Lever":
            threshold = self.grab_pull_threshold

        if confidence > threshold:
            return raw_label
        return "Idle"

    def _label_from_probs(
        self,
        probs,
        *,
        grabbing: bool = False,
        mirror: bool = False,
        puzzle_classes_only: bool = False,
    ) -> str:
        scores = [float(x) for x in probs]
        if puzzle_classes_only:
            allowed = LSTM_PUZZLE_ALLOWED_CLASSES
            scores = [
                s if self.classes[i] in allowed else 0.0
                for i, s in enumerate(scores)
            ]
            total = sum(scores)
            if total > 1e-8:
                scores = [s / total for s in scores]
        class_id = max(range(len(scores)), key=lambda i: scores[i])
        return self._decide_label(
            class_id, scores[class_id], grabbing=grabbing, mirror=mirror
        )

    def predict(
        self,
        landmarks,
        *,
        infer: bool = True,
        mirror: bool = False,
        grabbing: bool = False,
        puzzle_classes_only: bool = False,
    ):
        """Classify the rolling window. ``mirror`` reflects a left action hand
        into right-hand geometry before buffering (see ``flatten_landmarks``).

        ``grabbing`` is the committed action-hand fist. It never invents a
        pull; it only accepts a weaker Pull_Lever score when that class won.
        """
        if self.model is None:
            return "No Model"

        self.register_hand_seen()
        self.frame_buffer.append(flatten_landmarks(landmarks, mirror=mirror))

        if not infer:
            # Gate closed: keep filling the buffer so the first prediction after
            # it opens is instant, but skip the Keras call. Deliberately not
            # flush() — that would dump the warm buffer and cost a 30-frame
            # refill on every puzzle start. Clear the label so a stale action
            # cannot leak across the closed->open edge.
            self._reset_action_hold()
            self.last_output = "Idle"
            return "Idle"

        if len(self.frame_buffer) < self.buffer_size:
            return self.last_output

        # predict() rebuilds a data adapter and epoch iterator on every call —
        # 30ms for this model vs 1.1ms for predict_on_batch on the same input.
        # Batch-of-one inference must never go through predict().
        input_data = np.array([list(self.frame_buffer)], dtype="float32")
        prediction = self.model.predict_on_batch(input_data)

        raw_label = self._label_from_probs(
            prediction[0],
            grabbing=grabbing,
            mirror=mirror,
            puzzle_classes_only=puzzle_classes_only,
        )

        self.last_output = self._apply_action_hold(raw_label)
        return self.last_output
