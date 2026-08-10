"""Single source of truth for ports, frame counts, thresholds, and paths."""

UDP_IP = "127.0.0.1"
UDP_PORT = 5052

# Live webcam (not used by file-based Layer B eval).
CAMERA_INDEX = 0
# Lower capture size → less MediaPipe cost and less queue lag (device may only
# approximate; Actual size is whatever OpenCV negotiates).
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
# Soft webcam glitches are common on macOS/AVFoundation — do not exit on one bad read.
CAMERA_SOFT_RETRIES = 8
CAMERA_SOFT_RETRY_SLEEP_S = 0.05
CAMERA_REOPEN_SLEEP_S = 0.4

# A frame at or above this counts as stalled, reported as `slow=N` in the
# [perf] line. ~33ms is one frame at 30fps, so 100ms means the loop lost 3.
FRAME_SLOW_MS = 100.0
# Debug preview window. Unity receives everything over UDP and never reads this
# window, so on the demo machine every landmark circle, overlay line and the
# macOS HighGUI event loop is pure waste. False also disables the Q/C/P keys
# (they come from cv2.waitKey) — quit with Ctrl-C instead.
SHOW_PREVIEW = True
# Periodic percentile summary, seconds between lines. 0 disables it.
FRAME_STATS_INTERVAL_S = 5.0

# Thread budget for OpenCV / MediaPipe / TensorFlow (see runtime.py). Each
# library otherwise sizes its pool to the logical core count, so all three
# oversubscribe the few performance cores Unity is already using.
CPU_THREADS = 2
TF_INTER_OP_THREADS = 1

NUM_FRAMES = 30
NUM_FEATURES = 63  # 21 landmarks * (x, y, z)

LSTM_BUFFER_SIZE = NUM_FRAMES
LSTM_CONFIDENCE_THRESHOLD = 0.8
# Only wipe the sequence buffer after a sustained right-hand loss (seconds).
# Brief MediaPipe dropouts must not force a full 30-frame refill.
LSTM_HAND_MISS_CLEAR_S = 1.0
# One-shot gesture stick: report action at most this long, then Idle until model Idle again.
LSTM_ACTION_MAX_HOLD_S = 1.0
# LSTM inference only runs while the puzzle gate is open (see puzzle_gate.py).
# Off at startup: Unity opens it per puzzle; P toggles it by hand for testing.
PUZZLE_GATE_DEFAULT_ACTIVE = False

# Which physical hand plays the ACTION role (grab / cursor / LSTM). The other
# hand plays the MOVE role. See hand_roles.py — the payload keys stay
# left*/right* so Unity needs no change; only which hand fills them moves.
ACTION_HAND_DEFAULT = "right"
# The LSTM is trained on right-hand clips only. When the action hand is the
# left one, mirror its landmarks (x -> 1-x) so the model sees right-hand
# geometry. Set False to feed raw left-hand landmarks (worse, but useful for
# comparing against a model retrained on both hands).
MIRROR_LEFT_HAND_FOR_LSTM = True
# Mirroring also reverses on-screen rotation direction, so a physically
# clockwise left-hand turn classifies as CCW. Swap the pair back afterwards so
# a class means the same screen motion for either hand.
LSTM_MIRRORED_CLASS_SWAP = {
    "Turn_Around_CW": "Turn_Around_CCW",
    "Turn_Around_CCW": "Turn_Around_CW",
}

MODEL_PATH = "models/escape_gestures.keras"
DATA_DIR = "data"

MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.7
# Hand model: 0 = lite (~2x faster, slightly coarser landmarks), 1 = full.
# Raise to 1 on a machine with CPU headroom (e.g. the demo laptop).
MEDIAPIPE_HAND_MODEL_COMPLEXITY = 0
# NOTE: downscaling the frame before MediaPipe does NOT help. Benchmarking
# showed 320x240 and 640x480 cost the SAME (~20ms), because MediaPipe resizes
# to its model's fixed input size (~192x192) internally. Recorded here so the
# measurement does not get redone from scratch.

# Run the face mesh on every Nth frame and reuse the previous face in between.
# Head pitch and tilt are slow signals, so 15Hz is ample, and the face mesh is
# a whole MediaPipe graph per frame. 1 disables the skipping.
FACE_MESH_EVERY_N = 2

MAX_NUM_FACES = 2
# 4 is deliberate: PlayerLock needs to SEE bystanders' hands in order to reject
# the ones too far from the locked player's face (see player_lock._hand_ok).
# Lowering this makes crowd robustness worse, not better.
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

# --- Static hand gestures -------------------------------------------------
# Finger extension is measured along the hand's own axis and divided by the
# palm length (see gestures/hand/fingers.py), so these are in palm lengths and
# do not change with hand size, distance or wrist angle. Full extension is
# around +0.5 and a full curl around -0.5; the band between the two thresholds
# is the dead zone where a finger counts as neither, which is what lets a
# mid-transition hand be classified as "no gesture" instead of as whichever
# gesture it is passing through.
FINGER_EXTEND_MARGIN = 0.10
FINGER_CURL_MARGIN = 0.10
# Thumb tip distance from the palm centre, also in palm lengths. Replaces the
# old fixed 0.08 in image coordinates, which quietly got stricter as the player
# leaned back. Calibrated against the 34k recorded frames in data/: across
# frames where all four fingers are clearly extended, thumb spread runs
# p5=0.32 / p25=0.52 / p50=0.66. Converting the old 0.08 literally would have
# landed near 0.7 and thrown away over half of them. Swept 0.5/0.55/0.6/0.65 on
# test_video_1: 0.6 gives the fewest spurious open-palm triggers (right 30->27,
# left 28->26) with no cost to the intended fire rate, which does not move at
# any value in that range because it is limited by the finger dead zone rather
# than by the thumb.
THUMB_SPREAD_RATIO = 0.6

# Commit delay per gesture, in consecutive frames (see hand/debounce.py).
# Asymmetric on purpose:
#   on  = frames needed to START the gesture — high for one-shots, so a pose
#         passed through on the way to another cannot fire it.
#   off = frames needed to END it — high for holds, so a dropped frame or a
#         brief loss of the hand does not release a crouch or stop movement.
# Entering a gesture costs latency, so holds enter fast (walking should not lag
# by 100ms) while jump, which nobody notices two frames late, enters slow.
HAND_GESTURE_ON_FRAMES = {
    "fist": 2,
    "peace": 2,
    "open_palm": 2,
    "index_up": 3,
    "none": 1,
}
HAND_GESTURE_OFF_FRAMES = {
    "fist": 4,
    "peace": 5,
    "open_palm": 3,
    "index_up": 2,
    # Leaving "no gesture" is governed purely by the incoming gesture's on-count.
    "none": 0,
}
HAND_GESTURE_DEFAULT_ON_FRAMES = 2
HAND_GESTURE_DEFAULT_OFF_FRAMES = 2

# The tables above are the ACTION hand's. The MOVE hand overrides them, because
# the same gesture means different things on each hand and the two wants are
# opposite:
#
#   fist on the ACTION hand = sit / grab. Sit is a one-shot fired on a rising
#   edge, so a spurious frame costs you a chair; delay is worth paying.
#   fist on the MOVE hand   = walk forward. Held continuously, released
#   constantly, and every frame of delay is felt directly as the player sliding
#   past where they meant to stop.
#
# So the MOVE hand's fist starts instantly and releases after a single
# tolerated frame. Everything it does not name is inherited, which is the point
# of writing it as an override: jump keeps on=3 because it is the one-shot that
# phantom-fired, and crouch keeps off=5 because that long release is the whole
# reason a dropped frame no longer stands the player up.
MOVE_GESTURE_OVERRIDES = {"fist": {"on": 1, "off": 2}}

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
