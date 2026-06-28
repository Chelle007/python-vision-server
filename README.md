# Python Vision Server

Python backend for **Gaming with Bare Hands** (CSIT321 FYP). Captures webcam input, tracks hands and head with MediaPipe, classifies gestures, and sends data to Unity over UDP.

---

## What This Does


| Component                              | Role                                                                |
| -------------------------------------- | ------------------------------------------------------------------- |
| **MediaPipe Hands**                    | Tracks 21 landmarks per hand (up to 2 hands)                        |
| **MediaPipe Face Mesh**                | Head yaw/pitch for camera control                                   |
| **Hand heuristics** (`gestures/hand/`) | Static gestures — fist, open palm, index up, peace, rotation |
| **Head heuristics** (`gestures/head/`) | Head orientation (yaw/pitch); extensible for nod/shake/tilt         |
| **LSTM** (`gestures/dynamic/`)         | Dynamic puzzle gestures — `Idle`, `Turn_Key`, `Pull_Lever`          |
| **UDP**                                | Sends JSON every frame to Unity at `127.0.0.1:5052`                 |


### Hand roles


| Hand      | Used for                                | Detection         |
| --------- | --------------------------------------- | ----------------- |
| **Left**  | Movement (move, jump, crouch)           | Heuristics        |
| **Right** | Grab, release, inspect, puzzle gestures | Heuristics + LSTM |


---

## Requirements

- Python **3.11** (recommended)
- Webcam
- Unity game running with `UDPReceiver.cs` on port **5052**

---

## Setup (first time)

```bash
cd python-vision-server

python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 1. Pinned deps (TensorFlow, NumPy 1.26.x, MediaPipe — required for LSTM train/load)
pip install -r requirements.txt

# 2. Install this repo as a package (imports + vision-server commands)
pip install -e .

# Optional: run tests
pip install pytest
```

> Use **Python 3.11**. `requirements.txt` pins versions that work together (e.g. NumPy 1.26.4 with TensorFlow 2.16.2).

---

## Run the Vision Server

**Start Unity first**, then:

```bash
source .venv/bin/activate
python scripts/run_server.py
# or: vision-server
```

- A webcam window opens with gesture overlays
- Press **Q** to quit
- Check terminal for: `Combined Vision Server Running. Sending UDP to 127.0.0.1:5052`

### On-screen labels


| Label                   | Meaning                                     |
| ----------------------- | ------------------------------------------- |
| `LEFT FIST = MOVE`           | Left-hand movement gesture detected         |
| `RIGHT FIST = GRAB`          | Right-hand grab detected                    |
| `HEAD TILT LEFT = LOOK BACK` | Head rolled left (180° camera flip)         |
| `HEAD TILT RIGHT = LOOK BACK`| Head rolled right (180° camera flip)        |
| `AI LSTM: Turn_Key`     | Dynamic puzzle gesture detected             |
| `Stabilizing... (n/30)` | LSTM buffer filling (needs 30 frames first) |


---

## Record Training Data

Use `scripts/record_data.py` to capture 30-frame landmark sequences for the LSTM.

### 1. Set the gesture name

Open `src/vision_server/recording.py` and change:

```python
GESTURE_NAME = "Idle"   # e.g. "Turn_Key", "Pull_Lever", "FP_Turn_Key"
```

### 2. Run the recorder

```bash
python scripts/record_data.py
# or: vision-record
```

### 3. Controls


| Key   | Action                                        |
| ----- | --------------------------------------------- |
| **R** | Start recording a clip (hand must be visible) |
| **Q** | Quit                                          |


Each clip = **30 frames** (~0.5s at 60fps). Saved as `.npy` in `data/<GESTURE_NAME>/`.

### 4. Data folders

```
data/
├── Idle/              # Normal idle / random hand movement
├── Turn_Key/          # Correct key-turn motion
├── Pull_Lever/        # Correct lever-pull motion
├── FP_Turn_Key/       # Wrong motions → trained as Idle (reduce false positives)
└── FP_Pull_Lever/
```

**Target:** ~300 clips per folder. Use your **right hand**, same distance/angle you use in gameplay.

---

## Train the LSTM Model

After recording data:

```bash
python scripts/train_lstm.py
# or: vision-train
```

This will:

1. Load all `.npy` files from `data/` (see `FOLDER_MAPPING` in `src/vision_server/config.py`)
2. Split 80% train / 20% test
3. Train for 60 epochs
4. Save model as `models/escape_gestures.keras`

Restart the vision server to load the new model.

### Adding a new dynamic gesture

1. Add folder under `data/` and record clips
2. Update `CLASSES` in `src/vision_server/gestures/dynamic/labels.py`
3. Update `FOLDER_MAPPING` in `src/vision_server/config.py`
4. Retrain → new `models/escape_gestures.keras`

---

## Project Structure

```
python-vision-server/
├── pyproject.toml
├── requirements.txt
├── README.md
├── scripts/
│   ├── run_server.py       # Live tracking + UDP to Unity
│   ├── record_data.py      # Record training clips
│   └── train_lstm.py       # Train LSTM from data/
├── src/vision_server/
│   ├── app.py              # Main loop
│   ├── config.py           # Ports, frame counts, thresholds, paths
│   ├── features.py         # Shared landmark flattening
│   ├── udp.py              # UDP socket + payload builder
│   ├── overlay.py          # On-screen HUD
│   ├── recording.py        # Data recorder logic
│   ├── training.py         # LSTM training logic
│   ├── tracking/           # MediaPipe wrappers (hands, face)
│   └── gestures/
│       ├── hand/           # Static hand rules (registry-based)
│       ├── head/           # Head rules (registry-based)
│       └── dynamic/        # LSTM inference + class labels
├── models/escape_gestures.keras   # gitignored
├── data/                          # Training data (.npy clips), gitignored
└── tests/
```

---

## Tests

```bash
pytest
```

---

## UDP Payload (sent to Unity every frame)

Key fields teammates may use:


| Field                                  | Type   | Description                            |
| -------------------------------------- | ------ | -------------------------------------- |
| `head_yaw`, `head_pitch`               | float  | Head orientation (−1 to 1)             |
| `tilt_left`, `tilt_right`              | bool   | Head roll left/right (ear to shoulder) |
| `leftFist`, `leftIndexUp`, `leftPeace` | bool   | Left-hand movement                     |
| `rightFist`, `rightOpenPalm`           | bool   | Right-hand interaction                 |
| `rightIndexUp`                         | bool   | Right-hand index up (e.g. stand from sit) |
| `palmX`, `palmY`                       | float  | Right palm screen position             |
| `fistRotX/Y/Z`                         | float  | Right-hand rotation (inspect)          |
| `lstm_gesture`                         | string | `Idle`, `Turn_Key`, or `Pull_Lever`    |
| `hands[]`                              | array  | Per-hand landmarks + world landmarks   |


---

## Troubleshooting


| Problem              | Try                                                                             |
| -------------------- | ------------------------------------------------------------------------------- |
| `No Model` on screen | Run `scripts/train_lstm.py` or check `models/escape_gestures.keras` exists      |
| Unity not responding | Unity must be running first; check port 5052                                    |
| Hand not detected    | Better lighting, plain background, hand closer to camera                        |
| LSTM always `Idle`   | Confidence < 0.8 — retrain with more data or check gesture motion               |
| Webcam won't open    | Close other apps using the camera                                               |
| Import errors        | Run `pip install -r requirements.txt` then `pip install -e .` from project root |


