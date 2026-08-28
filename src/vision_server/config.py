"""Single source of truth for ports, frame counts, thresholds, and paths."""

UDP_IP = "127.0.0.1"
UDP_PORT = 5052

# Inbound control channel (Unity -> server), see control.py. A second socket
# rather than a reply on the one above: 5052 is Unity's listen port, so the
# server cannot bind it. Loopback only — both processes live on the demo
# machine, and binding 0.0.0.0 would raise a Windows firewall prompt on first
# run for no gain.
UDP_CONTROL_IP = "127.0.0.1"
UDP_CONTROL_PORT = 5053
# Ceiling on datagrams read per frame. The queue is drained empty every frame,
# so this is never hit in normal use — it exists so a flood cannot hold the
# frame loop past its budget.
UDP_CONTROL_MAX_DATAGRAMS = 32
# Read buffer per datagram. Control messages are a few dozen bytes; anything
# larger is not ours and gets truncated rather than growing the buffer.
UDP_CONTROL_RECV_BYTES = 2048

# Gesture coach report (server -> Unity). TCP, not UDP: one dropped datagram
# would lose a whole session's stats. Port 5053 is already the UDP *control*
# channel (Unity -> server), so this cannot share it.
TCP_REPORT_IP = "127.0.0.1"
TCP_REPORT_PORT = 5055
TCP_REPORT_CONNECT_S = 0.4

# Retroactive labelling window. When a gesture finally commits, frames in this
# lookback are treated as attempts at it. Idle players never commit, so they
# never enter the data — a known, accepted bias.
GESTURE_REPORT_LOOKBACK_S = 2.0
# Wrist–middle-MCP length in image-normalised coords. Below this the hand is
# in frame but too far back for stable classification.
GESTURE_REPORT_MIN_PALM = 0.08
# Size-normalised mean landmark step. Above this, tracking is jumpy
# (motion, autofocus, MediaPipe noise) — not a brightness reading.
GESTURE_REPORT_JITTER = 0.045
# Mean luma (0–255) of the hand crop, or the whole frame if no hands.
# Below this the Lighting chip is "bad". Webcam auto-exposure can hide a
# dark room, so this is still a proxy — just not landmark jitter.
GESTURE_REPORT_DARK_MEAN = 48.0
# Coach pain-list row ids. Click is Unity-only and is filled on that side.
GESTURE_REPORT_ROWS = (
    "jump",
    "crouch",
    "inventory",
    "grab",
    "watch_tap",
    "pull_lever",
)

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

# Calibration preview stream (server -> Unity), see preview.py. Its own port
# because the payload is binary JPEG, and feeding that to the JSON parser
# listening on UDP_PORT would only produce noise.
#
# The stream exists so exactly one process opens the webcam. Unity used to run
# its own WebCamTexture for the calibrate panel, which works on macOS but not
# on Windows, where camera access is exclusive and the server already holds the
# device. Sending pictures of the frame the server already has avoids the
# contention entirely.
UDP_PREVIEW_PORT = 5054
# Only streams while the calibrate panel is open, so this is a menu-time cost,
# not a gameplay one. Still rate-limited: the frame loop is the same one
# running MediaPipe.
PREVIEW_FPS = 10.0
# Match the capture width so Unity receives the same picture the debug preview
# window shows, at the same aspect, and no resize happens at all. Larger
# captures are still scaled down to this. Note aspect ratio is preserved either
# way — a preview that looks stretched in Unity is a RawImage rect that is not
# 4:3, not a resolution problem.
PREVIEW_WIDTH = CAMERA_WIDTH
PREVIEW_JPEG_QUALITY = 50
# Hard ceiling per datagram. A UDP payload cannot exceed 65507 bytes and
# sendto would raise; an oversized frame is dropped instead, since the next one
# is 100ms away. Real 640x480 frames encode to ~8-10KB; only pathological
# noise approaches this.
PREVIEW_MAX_BYTES = 60000
# Socket buffer for the preview stream, both ends. NOT optional: macOS caps a
# UDP datagram at net.inet.udp.maxdgram (9216 bytes by default) and sendto
# fails with EMSGSIZE above that, which a full-size JPEG frame clears easily.
# Raising SO_SNDBUF lifts the cap — measured here as 9216 -> 60000+. Unity must
# set ReceiveBufferSize to match or the frames arrive truncated.
PREVIEW_SOCKET_BUFFER_BYTES = 262144

# A frame at or above this counts as stalled, reported as `slow=N` in the
# [perf] line. ~33ms is one frame at 30fps, so 100ms means the loop lost 3.
FRAME_SLOW_MS = 100.0
# Debug preview window. Unity receives everything over UDP and never reads this
# window, so on the demo machine every landmark circle, overlay line and the
# macOS HighGUI event loop is pure waste. False also disables the Q/C/P keys
# (they come from cv2.waitKey) — quit with Ctrl-C instead.
SHOW_PREVIEW = True
# Top-left HUD on the OpenCV preview (LSTM, puzzle mode, gesture labels).
# Toggle at runtime with T; skeletons, lock ring and pitch meter stay on.
SHOW_OVERLAY_HUD = True
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
# Softer Pull_Lever cutoff while the action fist is already committed.
# Fist is not treated as a pull on its own: argmax must still be Pull_Lever.
# 0.6 sits under the 0.8 Idle floor so a slow/partial pull can land, but
# stays high enough that a grab plus a weak Turn_Key/Idle score cannot.
LSTM_GRAB_PULL_LEVER_THRESHOLD = 0.6
# Live ATM puzzle only uses Pull_Lever. A hand filling the frame looks like a
# wide circle to the 5-class model, so Turn_Around_* (and unused Turn_Key)
# steal argmax and the grab floor never applies. Restrict + renormalise only
# on the live puzzle path; Layer A/B keep the full softmax.
LSTM_PUZZLE_ALLOWED_CLASSES = ("Idle", "Pull_Lever")
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

# TESTED at 0.5 and reverted. In the dark it clearly beat 0.7, but in a lit
# room it was much worse: background objects got detected as hands. A dark
# frame hides the clutter that a lit one exposes, so the loosened threshold
# only pays while evidence is scarce — once the room is lit it admits junk,
# and MAX_NUM_HANDS=4 gives that junk four slots to fill. Note this
# constant is shared with the face mesh (tracking/face.py), so 0.5 also
# loosened face detection; if 0.5 is ever revisited, split the two first so
# hands and faces can be tuned apart.
# Use MEDIAPIPE_HAND_MODEL_COMPLEXITY for the dark case instead: it helped at
# distance without a lighting-dependent downside.
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.7
# Hand model: 0 = lite (~2x faster, slightly coarser landmarks), 1 = full.
# Default stays 0; press M in the preview to try 1 live (see app.py). Measured
# on the Windows demo machine in a dark room, before.mov vs after.mov, ~70-80s
# each. The gain only appears at DISTANCE — with hands close to the camera both
# settings track 100%, so a hands-up-close spot check shows no difference and
# is the wrong test:
#   tracked over clip:  0 -> 66.3%   1 -> 79.5%
#   per-5s late in the clip, hands at arm's length in the dark:
#     0 -> 93 14 62 75  0 12 74  0     (visible hands, simply not found)
#     1 -> 85  0 82 88 64 68 100 86
# The cost is real and is NOT free: with two hands up the loop is compute-bound,
# not camera-bound, so the extra time comes straight off the frame rate.
#   0: hands med=34ms, total med=50ms, fps 19.2-20.3
#   1: hands med=56ms, total med=72-75ms, fps 12.9-13.7
# Left at 0 pending more hands-on time from the team: 1 trades a third of the
# frame rate for detection at distance, and at 13fps the LSTM gesture window
# stretches too (NUM_FRAMES is a frame count, not a duration).
#
# The cheap alternative was tried and rejected: detection confidence 0.5
# recovered at-distance detection in the dark for free, but detected background
# objects as hands once the lights were on.
MEDIAPIPE_HAND_MODEL_COMPLEXITY = 1
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

# --- OK sign (inventory open) ----------------------------------------------
# Thumb tip touching index tip with the other three fingers extended, measured
# as the distance between those two tips in palm lengths (see
# fingers.pinch_distance).
#
# This gesture replaced a thumbs-up, and the reason is the FINGER PATTERN, not
# the threshold. A thumbs-up curls all four fingers, so it presents the same
# pattern as a fist and the thumb alone had to break the tie. That tie could
# not be broken reliably: the thumb is the one digit whose 3D pose survives
# least well in a 2D projection, and every candidate measurement — clearance
# from the fingertips, reach from its own base, rise along the hand's axis —
# either read real thumbs-ups as fists or read relaxed fists as thumbs-ups.
# Since fist on the MOVE hand is walk-forward, both failures were expensive:
# one opened the inventory mid-stride, the other stopped the player dead.
#
# The OK sign curls the index and extends the other three. No other gesture in
# the set uses that pattern, so the tie never has to be broken and no amount of
# thumb noise can turn a walk into a menu.
#
# The pinch is only a confirmation. It separates the OK sign from the one other
# pose sharing the pattern — three fingers up with the index simply folded down
# — where the thumb rests by the palm rather than on the index tip. Touching
# pads measure ~0.10-0.25 palm lengths (a fingertip is about 0.15 wide, so the
# two landmarks cannot reach 0 even pressed hard together) against ~0.8 and up
# with the thumb parked. 0.35 sits in the empty middle of that gap.
OK_SIGN_PINCH = 0.35
# hand_frame only rejects a hand with NO measurable axis (0.02), which leaves
# badly foreshortened hands whose small palm length distorts every ratio
# divided by it — on the recorded corpus those frames are the whole tail,
# reaching an impossible 5.8 palm lengths. That matters here in the direction
# that costs a false open: a hand pointing at the camera projects every
# landmark on top of every other, so an unpinched thumb can still land within
# OK_SIGN_PINCH. Below this floor the hand is not eligible for the gesture at
# all. 0.08 keeps 86% of recorded frames.
OK_SIGN_MIN_PALM = 0.08

# --- Pointing (inventory next / back) --------------------------------------
# One extended index used to be a single gesture. It is now four, split by
# where the finger points in SCREEN space (see fingers.index_direction): up is
# jump, left and right step the inventory selection, and down is its own
# gesture rather than the leftover it used to be.
#
# The value is a cosine, so it defines a cone half-angle: 0.80 is ~37 degrees
# either side of each axis. It MUST stay above cos(45 degrees) = 0.707, or the
# up/down and sideways cones overlap and a diagonal point satisfies both.
#
# Measured on the 6.6k index-up frames in data/, all of them genuine jumps:
# they point firmly up (vert p50 = +0.99, p5 = +0.70), so the split is nearly
# free. At 0.80, 92.1% still read as up, 3.8% land sideways and 4.0% fall in
# the dead zone between cones. Loosening to 0.70 keeps 95% as up but removes
# the dead zone entirely; tightening to 0.85 drops jump recall to 89.5% for
# little gain.
#
# NOTE this is a real behaviour change for jump: an index pointing sideways
# used to fire it and now does not.
#
# Adding the down cone costs nothing measurable: it can only claim frames that
# already read NONE, and over all 34.6k recorded frames in data/ exactly 7
# (0.02%, all in Pull_Lever) fall inside it. The corpus contains no deliberate
# point-down, so that is the whole false-fire surface, and it is well under the
# 3-frame commit delay below.
INDEX_POINT_CONE = 0.80

# --- Watch tap (pause) ------------------------------------------------------
# Distance from the tapping index tip to the watched wrist, in palm lengths
# (see hand/watch_tap.py). Was a flat 0.09 in image coordinates, which is a
# fixed slice of the FRAME rather than of the HAND: over test_video_1 the palm
# measures p5=0.117 / p50=0.178 / p95=0.230, so the same 0.09 ran from 0.5 palm
# lengths up close to 0.77 as the player leaned back — loosening by half at
# exactly the distance where the hands are hardest to tell apart. Same bug, and
# the same fix, as THUMB_SPREAD_RATIO.
#
# On the 3 labelled tap segments, contact frames sit at p50 = 0.28 / p75 = 0.32
# palm lengths, against 1.5 (p50) for frames where the hands are simply both in
# shot. 0.45 leaves ~1.4x headroom over real contact and stays far below that.
#
# The number is deliberately not doing the heavy lifting — the pose gate is.
# Swept 0.35/0.45/0.55/0.75/1.10 against on-counts 2/3/4/6: with the pose gate
# on, every combination gives the SAME two triggers outside the labelled tap
# windows, and both of those are the player's finger already resting on the
# wrist 0.8s and 2.8s before the window opens (the labels file calls its
# boundaries approximate). Recall varies by 4 points across that whole grid.
#
# What each term is worth, measured over the 2.7 min of non-tap footage as
# spurious RISING edges, since Unity pauses on the edge and one edge is one
# unwanted pause:
#
#   old rule (0.09 image-space, min(index, middle), no debounce)   8 edges
#   scale only      (0.55 palms, no pose gate, no debounce)       13 edges
#   scale + pose    (0.55 palms, no debounce)                      2 edges
#   scale + pose + debounce (0.55, on=3)                           2 edges
#
# Of the old rule's 8, six are genuine: an open palm resting near the wrist
# (2 edges at 96s) and a fist doing the same during idle (4 edges at 120-121s).
# The pose gate rejects both outright — neither hand is pointing — which is why
# it, and not the threshold, is what fixes the gesture. The remaining two are
# the boundary artifacts above, and they are the only ones the new rule keeps.
#
# Recall drops 75% -> 60% of labelled tap frames, and that is the honest cost.
# It buys back nothing at the segment level (3/3 segments still fire, at every
# ratio tried) because the loss is concentrated in frames where the hand is
# approaching or withdrawing inside a generously drawn label window, not in the
# contact itself.
WATCH_TAP_RATIO = 0.45
# Unity pauses on the RISING edge of watchTap, so a single flicker frame is a
# single unwanted pause and entering slowly is worth the latency — the same
# reasoning as ok_sign above, on a gesture nobody makes in a hurry. It earns
# nothing on the corpus above, where the pose gate has already removed the
# flicker it defends against; it is kept because that gate is measured on one
# player in good light, and the cost is 100ms.
#
# Leaving is delayed for a different reason: a mid-tap dropout would release
# and re-arm, and the re-arm is a second rising edge — pause, unpause, pause.
WATCH_TAP_ON_FRAMES = 3
WATCH_TAP_OFF_FRAMES = 3

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
    # Both are UI one-shots (open inventory / next slot), so they enter slowly:
    # nobody notices a quarter-second of latency opening a menu, and the hand
    # passes through half-formed poses on the way into either one.
    #
    # ok_sign used to be thumbs_up at 8 frames, where the long hold was doing
    # real detection work rather than filter hygiene — it was the third term
    # propping up a thumb measurement that could not separate the gesture from
    # a fist on its own. The OK sign is separated by its finger pattern instead
    # (see OK_SIGN_PINCH), so the hold is back to being ordinary debouncing and
    # 5 frames (~165ms) is enough to ride out the curl of the index on the way
    # in without making the menu feel sluggish.
    "ok_sign": 5,
    "rock_sign": 3,
    # Same one-shot reasoning as rock_sign, and they share its exposure to a
    # finger sweeping past on the way somewhere else.
    "index_left": 3,
    "index_right": 3,
    "index_down": 3,
    "none": 1,
}
HAND_GESTURE_OFF_FRAMES = {
    "fist": 4,
    "peace": 5,
    "open_palm": 3,
    "index_up": 2,
    # Held while the menu is read, and the pose relaxes by opening the pinch,
    # which passes through patterns that are not gestures. Releasing slowly
    # keeps the menu from flickering shut mid-read.
    "ok_sign": 4,
    "rock_sign": 3,
    "index_left": 3,
    "index_right": 3,
    "index_down": 3,
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
#
# rock_sign is the same argument, arrived at later. It used to be "next item" —
# a one-shot, where the tables above give it on=3 so a finger sweeping past
# cannot step the inventory. It is now walk BACKWARD, which is the fist's
# profile exactly: held continuously, released constantly, and every frame of
# delay felt as the player sliding past their mark. Left inherited, backward
# would start ~3 frames after forward does and stop a frame later, which is a
# difference the player feels between two halves of the same control.
#
# Deliberately MOVE-only, like the fist. rightRockSign keeps the one-shot
# timing: it is unbound today, and if it is ever bound it will be to a one-shot,
# because that is what the ACTION hand's gestures are.
MOVE_GESTURE_OVERRIDES = {
    "fist": {"on": 1, "off": 2},
    "rock_sign": {"on": 1, "off": 2},
}

# Labels on the ACTION hand whose hold does not time out against an unreadable
# hand. An off-count is a bet that a dropout is short, and for grab that bet is
# simply wrong: `classify_hand` reports NONE whenever the palm axis it measures
# against foreshortens, which is exactly what happens when a fist is rotated
# toward the camera or held further away. Measured over the 21k grab-pose frames
# in data/ (Pull_Lever, Turn_Key, both views), the fist rate falls from 61% at
# palm_len 0.13-0.16 to 19% below 0.04, because ring and pinky flip from a
# median margin of -0.19 to +0.18 and the all-curled pattern stops matching.
#
# The resulting NONE runs are p50 3 frames but p90 22, so 42% of them outlast
# the 4-frame off-count and drop whatever the player was holding. No frame count
# survives that tail, and one long enough to try would make real releases feel
# stuck.
#
# So the release condition changes shape rather than size: a latched label is
# held through NONE indefinitely, and ends only once some *other* label is read
# — which still costs the usual off-count, so this weakens nothing about how a
# deliberate release behaves. Over the same frames, once a grab is underway 37%
# read NONE while only 2.5% read a competing label, so nearly the whole failure
# mode is recovered and almost no real release is delayed.
#
# ACTION-only, and deliberately not in the tables above: fist on the MOVE hand
# is walk-forward, where holding through a dropout means the player keeps
# walking after they meant to stop. Latching that would be a worse bug than the
# one this fixes.
#
# Set to () to disable — the debouncer falls back to pure frame counting.
ACTION_GESTURE_LATCH = ("fist",)

# Consecutive frames with no ACTION hand at all before a latched hold is
# abandoned. Without it the latch has no exit: a player who lowers their arm
# leaves a fist committed forever, and never gets the object out of their hand.
#
# "No hand" is a different signal from "unreadable hand", and only this one gets
# a timeout. A hand MediaPipe can see but `classify_hand` will not name is the
# blind spot the latch exists to cover, and it lasts as long as the pose does. A
# hand MediaPipe cannot find at all is out of frame or by the player's side, and
# waiting on that is how a grab gets stuck.
#
# Generous on purpose: whole-hand tracking loss is also what poor lighting looks
# like, and dropping held objects when the room dims is its own bug. ~1s at
# 30fps, counted in frames because frame time here is neither 30fps nor stable.
ACTION_LATCH_HAND_LOST_FRAMES = 30

HEAD_TILT_THRESHOLD = 0.35
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
