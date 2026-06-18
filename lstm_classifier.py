import numpy as np
from collections import deque


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
        model_path="escape_gestures.keras",
        buffer_size=30,
        confidence_threshold=0.8,
        clear_after_misses=8,
    ):
        self.model = _load_keras_model(model_path)
        if self.model is not None:
            print(f"AI Brain '{model_path}' loaded successfully!")

        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.clear_after_misses = clear_after_misses
        self.hand_miss_streak = 0
        self.last_output = "Idle"

        # Must match FINAL_CLASSES in train_lstm.py exactly
        self.classes = ["Idle", "Turn_Key", "Pull_Lever"]

    def register_hand_seen(self):
        self.hand_miss_streak = 0

    def register_hand_lost(self):
        self.hand_miss_streak += 1
        if self.hand_miss_streak >= self.clear_after_misses:
            self.frame_buffer.clear()
            self.last_output = "Idle"

    def get_overlay_label(self):
        """Label for on-screen HUD (may include warmup progress)."""
        if len(self.frame_buffer) < self.buffer_size:
            if self.last_output in self.classes and self.last_output != "Idle":
                return self.last_output
            return f"Stabilizing... ({len(self.frame_buffer)}/{self.buffer_size})"
        return self.last_output

    def predict(self, landmarks):
        if self.model is None:
            return "No Model"

        current_frame = []
        for lm in landmarks:
            current_frame.extend([lm.x, lm.y, lm.z])

        self.frame_buffer.append(current_frame)

        if len(self.frame_buffer) < self.buffer_size:
            return self.get_overlay_label()

        input_data = np.array([list(self.frame_buffer)])
        prediction = self.model.predict(input_data, verbose=0)

        class_id = int(np.argmax(prediction))
        confidence = float(prediction[0][class_id])

        if confidence > self.confidence_threshold:
            self.last_output = self.classes[class_id]
        else:
            self.last_output = "Idle"

        return self.last_output
