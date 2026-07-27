"""Single source of truth for ports, frame counts, thresholds, and paths."""

UDP_IP = "127.0.0.1"
UDP_PORT = 5052

NUM_FRAMES = 30
NUM_FEATURES = 63  # 21 landmarks * (x, y, z)

LSTM_BUFFER_SIZE = NUM_FRAMES
LSTM_CONFIDENCE_THRESHOLD = 0.8
LSTM_CLEAR_AFTER_MISSES = 8

MODEL_PATH = "models/escape_gestures.keras"
DATA_DIR = "data"

MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.7

MAX_NUM_FACES = 2
MAX_NUM_HANDS = 4

# Player lock (distances in face-widths; timeouts in wall-clock seconds)
SEAT_RADIUS = 1.5
SEAT_EMPTY_S = 3.0
CHALLENGER_HOLD_S = 1.0
LOCK_CONFIRM_S = 0.33
FACE_MATCH_GATE = 2.0
HAND_MATCH_GATE = 2.5
# Image-space arm span is large vs face width — keep this loose for solo play.
HAND_TO_FACE_REACH = 8.0
CHALLENGER_MIN_FACE_WIDTH = 0.08
CHALLENGER_MAX_CENTER_DIST = 0.35
CHALLENGER_SIZE_REF = 0.22

HEAD_TILT_THRESHOLD = 0.6

TRAIN_EPOCHS = 60
TRAIN_BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.2
TRAIN_RANDOM_STATE = 42

# Maps human-organized data folders to model class labels
FOLDER_MAPPING = {
    "Idle": "Idle",
    "FP_Turn_Key": "Idle",
    "FP_Pull_Lever": "Idle",
    "Turn_Key": "Turn_Key",
    "Pull_Lever": "Pull_Lever",
}
