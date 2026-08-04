"""Single source of truth for ports, frame counts, thresholds, and paths."""

UDP_IP = "127.0.0.1"
UDP_PORT = 5052

NUM_FRAMES = 30
NUM_FEATURES = 63  # 21 landmarks * (x, y, z)

LSTM_BUFFER_SIZE = NUM_FRAMES
LSTM_CONFIDENCE_THRESHOLD = 0.8
# Only wipe the sequence buffer after a sustained right-hand loss (seconds).
# Brief MediaPipe dropouts must not force a full 30-frame refill.
LSTM_HAND_MISS_CLEAR_S = 1.0
# One-shot gesture stick: report action at most this long, then Idle until model Idle again.
LSTM_ACTION_MAX_HOLD_S = 1.0

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
# Tight frame-to-frame stick; wider reseed only after HAND_RESEED_S.
HAND_MATCH_GATE = 1.0
HAND_RESEED_S = 0.45
HAND_RESEED_GATE = 2.5
# Soft size preference vs face height (tie-break only, not a hard reject).
HAND_SIZE_REF_RATIO = 0.55
HAND_SIZE_SCORE_WEIGHT = 0.2
# Image-space arm span is large vs face width — keep this loose for solo play.
HAND_TO_FACE_REACH = 8.0
CHALLENGER_MIN_FACE_WIDTH = 0.08
CHALLENGER_MAX_CENTER_DIST = 0.35
CHALLENGER_SIZE_REF = 0.22

HEAD_TILT_THRESHOLD = 0.5
# Look-up on *calibrated* pitch: head_pitch <= -this (negative = look up).
# Applied AFTER pitch_neutral subtraction so rest ≈ 0.
HEAD_LOOK_UP_PITCH_THRESHOLD = 0.15
# Auto-calibrate pitch neutral after player lock.
PITCH_CAL_SAMPLE_S = 1.5
# Soft stability hint only (std above this shows hold still); completion still
# finishes after SAMPLE_S using the sample median — live face noise is often
# 0.04–0.08 and used to block forever at 0.035.
PITCH_CAL_MAX_STD = 0.10
PITCH_CAL_MIN_SAMPLES = 20

TRAIN_EPOCHS = 60
TRAIN_BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.2
TRAIN_RANDOM_STATE = 42

# Maps human-organized data folders to model class labels
FOLDER_MAPPING = {
    "Idle": "Idle",
    "FP_Turn_Key": "Idle",
    "FP_Pull_Lever": "Idle",
    "FP_Turn_Around": "Idle",
    "Turn_Key": "Turn_Key",
    "Pull_Lever": "Pull_Lever",
    "Turn_Around_CW": "Turn_Around_CW",
    "Turn_Around_CCW": "Turn_Around_CCW",
}
