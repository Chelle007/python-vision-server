# Python Vision Server

Python backend for **Gaming with Bare Hands** (CSIT321 FYP). Captures webcam input, tracks hands and head with MediaPipe, classifies gestures, and sends data to Unity over UDP.

---

## What This Does


| Component                              | Role                                                                |
| -------------------------------------- | ------------------------------------------------------------------- |
| **MediaPipe Hands**                    | Tracks 21 landmarks per hand (up to 2 hands)                        |
| **MediaPipe Face Mesh**                | Head yaw/pitch for camera control                                   |
| **Hand heuristics** (`gestures/hand/`) | Static gestures — fist, open palm, index up, peace, OK sign, rock sign, rotation |
| **Head heuristics** (`gestures/head/`) | Head orientation (yaw/pitch); extensible for nod/shake/tilt         |
| **LSTM** (`gestures/dynamic/`)         | Dynamic gestures — Idle, Turn_Key, Pull_Lever, Turn_Around_CW/CCW |
| **UDP**                                | Sends JSON every frame to Unity at `127.0.0.1:5052`                 |


### Hand roles


| Hand      | Used for                                | Detection         |
| --------- | --------------------------------------- | ----------------- |
| **Left**  | Movement (move, jump, crouch)           | Heuristics        |
| **Right** | Grab, release, inspect, puzzle gestures | Heuristics + LSTM |


---

## Requirements

- **Python 3.11** (required — not 3.12 or 3.13)
- Webcam
- Unity game running with `UDPReceiver.cs` on port **5052**

> **Players running the shipped `.exe`:** you do **not** need Python. Unity bundles `VisionServer/vision-server.exe`. The steps below are for **development** only.

---

## Setup (first time)

Run in **Command Prompt** or **PowerShell** from the repo root:

```bat
cd python-vision-server
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Optional — tests:

```bat
pip install pytest
```

<details>
<summary>macOS / Linux</summary>

```bash
cd python-vision-server
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install pytest
```

</details>

> Use **Python 3.11** exactly. `requirements.txt` pins versions that work together (e.g. NumPy 1.26.4 with TensorFlow 2.16.2). If you already have newer TensorFlow/MediaPipe installed globally, use a **fresh** `.venv` or run `pip install -r requirements.txt --force-reinstall` inside the venv.

---

## Run the Vision Server

**Start Unity first**, then activate the venv and run:

```bat
.venv\Scripts\activate
python scripts\run_server.py
```

To use a different webcam, pass its index:

```bat
python scripts\run_server.py --camera-index 1
```

Headless (no OpenCV debug window — same as the shipped game):

```bat
python scripts\run_server.py --headless
```

Same entry point as the `vision-server` console command after `pip install -e .`.

<details>
<summary>macOS / Linux</summary>

```bash
source .venv/bin/activate
python scripts/run_server.py
python scripts/run_server.py --camera-index 1
python scripts/run_server.py --headless
```

</details>

- A webcam window opens with gesture overlays
- Press **Q** to quit
- Press **T** to hide/show the top-left HUD text (skeletons, lock ring, and pitch meter stay on)
- Check terminal for: `Combined Vision Server Running. Sending UDP to 127.0.0.1:5052`

### On-screen labels

Press **T** in the webcam window to hide or show this HUD. Hand skeletons, the player-lock ring, and the pitch meter stay visible.


| Label                   | Meaning                                     |
| ----------------------- | ------------------------------------------- |
| `LEFT FIST = MOVE`           | Left-hand movement gesture detected         |
| `RIGHT FIST = GRAB`          | Right-hand grab detected                    |
| `HEAD TILT LEFT = TURN` | Head rolled left — hold to turn body (Unity) |
| `HEAD TILT RIGHT = TURN`| Head rolled right — hold to turn body (Unity) |
| `HEAD TILT LEFT/RIGHT = TURN BACK` | Look-up + exclusive tilt → snap body 180° |
| `CALIBRATING` / `CALIBRATED` | Pitch neutral auto-cal after player lock (C to redo) |
| `AI LSTM: Turn_Around_CW` | Clockwise circle (detected; reserved for puzzles) |
| `AI LSTM: Turn_Around_CCW`| Counter-clockwise circle (detected; reserved for puzzles) |
| `AI LSTM: Turn_Key`     | Dynamic puzzle gesture detected             |
| `AI LSTM: Idle`         | No dynamic gesture (warmup is silent)       |


---

## Record Training Data

Use `scripts/record_data.py` to capture 30-frame landmark sequences for the LSTM.

### 1. Set the gesture name

Open `src/vision_server/recording.py` and change:

```python
GESTURE_NAME = "Idle"   # e.g. "Turn_Key", "Pull_Lever", "FP_Turn_Key"
```

### 2. Run the recorder

```bat
python scripts\record_data.py
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
├── Turn_Around_CW/    # Clockwise circle (puzzle-ready; not used for body turn-back)
├── Turn_Around_CCW/   # Counter-clockwise circle (puzzle-ready)
├── FP_Turn_Key/       # Wrong motions → trained as Idle (reduce false positives)
├── FP_Pull_Lever/
└── FP_Turn_Around/    # Almost-circles / waves → Idle
```

**Target:** ~300 clips per true-gesture folder (~150 CW + 150 CCW if balancing turn-around); ~50–80 in each `FP_*` folder. Use your **right hand**, same distance/angle you use in gameplay.

---

## Train the LSTM Model

After recording data:

```bat
python scripts\train_lstm.py
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
├── eval/                   # Model evaluation (Layers A/B) — see eval/README.md
│   ├── run_layer_a.py
│   ├── run_layer_b.py
│   ├── measure_jitter.py
│   └── videos/             # Fixed test videos (you record)
├── src/vision_server/
│   ├── app.py              # Main loop
│   ├── config.py           # Ports, frame counts, thresholds, paths
│   ├── features.py         # Shared landmark flattening
│   ├── udp.py              # UDP socket + payload builder
│   ├── overlay.py          # On-screen HUD
│   ├── recording.py        # Data recorder logic
│   ├── training.py         # LSTM training logic
│   ├── evaluation/         # Shared eval metrics (LSTM, jitter, video)
│   ├── tracking/           # MediaPipe wrappers (hands, face)
│   └── gestures/
│       ├── hand/           # Static hand rules (registry-based)
│       ├── head/           # Head rules (registry-based)
│       └── dynamic/        # LSTM inference + class labels
├── models/escape_gestures.keras   # gitignored
├── data/                          # Training data (.npy clips), gitignored
└── tests/
```

### Evaluation (Layers A / B)

```bat
python eval\run_layer_a.py
python eval\run_layer_b.py eval\videos\test_video_1.mp4
```

<details>
<summary>macOS / Linux</summary>

```bash
python eval/run_layer_a.py
python eval/run_layer_b.py eval/videos/test_video_1.mp4
```

</details>

See `eval/README.md` and the repo-root `Gesture_Model_Evaluation_Plan.md`.

---

## Tests

```bat
pytest
```

---

## UDP Payload (sent to Unity every frame)

Key fields teammates may use:


| Field                                  | Type   | Description                            |
| -------------------------------------- | ------ | -------------------------------------- |
| `head_yaw`, `head_pitch`               | float  | Orientation (−1…1); pitch is rest-relative after cal (neg = look up) |
| `head_pitch_raw`, `pitch_calibrated`, `pitch_cal_status`, `pitch_cal_neutral` | debug | Raw pitch + cal state (Unity can ignore) |
| `tilt_left`, `tilt_right`              | bool   | Head roll L/R; hold-to-turn, or snap 180° if looking up |
| `leftFist`, `leftIndexUp`, `leftPeace` | bool   | Left-hand movement. `leftIndexUp` (jump) now requires pointing **up** — sideways is next/prev below |
| `leftOkSign`                           | bool   | Inventory open — a one-shot, read the **rising edge**. Replaced a thumbs-up, which shared the fist's finger pattern and so competed with walk-forward; the OK sign has a pattern of its own (see `OK_SIGN_PINCH`) |
| `leftRockSign`                         | bool   | Move backward — *held*, not a one-shot, so it starts and stops on the same counts as walk-forward (see `MOVE_GESTURE_OVERRIDES`). `rightRockSign` keeps the slower one-shot timing |
| `leftIndexLeft`, `leftIndexRight`      | bool   | Point left / right — inventory prev / next. Also one-shots; direction is screen-relative |
| `leftIndexDown`, `rightIndexDown`      | bool   | Point down — same cone treatment as the other three; no Unity binding yet |
| `rightFist`, `rightOpenPalm`           | bool   | Right-hand interaction                 |
| `rightIndexUp`                         | bool   | Right-hand index up (e.g. stand from sit) |
| `rightOkSign`, `rightRockSign`         | bool   | Same two gestures on the action hand (sent for symmetry; unused so far) |
| `palmX`, `palmY`                       | float  | Right palm screen position             |
| `fistRotX/Y/Z`                         | float  | Right-hand rotation (inspect)          |
| `lstm_gesture`                         | string | `Idle`, `Turn_Key`, `Pull_Lever`, `Turn_Around_CW`, `Turn_Around_CCW` |
| `hands[]`                              | array  | Per-hand landmarks + world landmarks   |


---

## Ship a Windows build (PyInstaller)

Do **not** run PyInstaller on macOS. Build on **Windows**.

1. Put `models/escape_gestures.keras` in `models/` (gitignored; copy it over).
2. One-time venv (same as Setup above): `py -3.11 -m venv .venv`, activate, `pip install -r requirements.txt`, `pip install -e .`
3. Double-click `scripts/build_windows.bat` (or run it from a prompt).
4. Copy the whole `dist/VisionServer` folder next to the Unity game `.exe`.
5. Launch the **game**. Unity starts `VisionServer/vision-server.exe` hidden with the webcam index chosen on the consent screen. No OpenCV window, no console.

Unity Editor play-mode does **not** auto-start the server — run `python scripts\run_server.py` yourself while developing (and set `CAMERA_INDEX` in `config.py` or pass `--camera-index`).

---

## Troubleshooting


| Problem              | Try                                                                             |
| -------------------- | ------------------------------------------------------------------------------- |
| `No Model` on screen | Run `scripts/train_lstm.py` or check `models/escape_gestures.keras` exists      |
| Unity not responding | Unity must be running first; check port 5052                                    |
| Hand not detected    | Better lighting, plain background, hand closer to camera                        |
| LSTM always `Idle`   | Confidence < 0.8 — retrain with more data or check gesture motion               |
| Webcam won't open    | Close other apps using the camera; try another index with `--camera-index 1` (or 2, 3, …) |
| Import / pip errors  | Confirm **Python 3.11** venv is active (`python --version`). Delete `.venv`, recreate, then `pip install -r requirements.txt --force-reinstall` and `pip install -e .` |
| Wrong Python version | Install Python 3.11 from python.org; use `py -3.11 -m venv .venv` on Windows   |


