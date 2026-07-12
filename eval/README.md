# Evaluation tooling

Separate from `scripts/` (live server / record / train). Implements the
**Gesture Model Evaluation Plan** Layers A and B helpers.

```
eval/
├── README.md
├── run_layer_a.py              # Offline LSTM metrics (test split)
├── measure_jitter.py           # Landmark jitter on a video
├── run_layer_b.py              # Detection / recovery / LSTM timeline on a video
├── export_annotated_video.py   # HUD + GT label replay → mp4
├── videos/                     # Put test_video_1.mp4 … here (you record these)
└── exports/                    # Annotated exports, per version (v1/, v2/, …)
```

Shared logic lives in `src/vision_server/evaluation/`.

---

## Prerequisites

```bash
cd python-vision-server
source .venv/bin/activate
pip install -e .
```

Need `data/` clips + `models/escape_gestures.keras` for Layer A.
Need recorded mp4s under `eval/videos/` for Layer B.

---

## Layer A — Offline LSTM

```bash
python eval/run_layer_a.py
```

Uses the same fixed split as training (`test_size=0.2`, `random_state=42`).
Prints overall accuracy, per-class accuracy, Idle→action FP rate, and confusion matrix.

---

## Layer B — Fixed test video

After you record the three videos into `eval/videos/`:

```bash
# Full automated pass (detection, recovery, jitter, LSTM timeline)
python eval/run_layer_b.py eval/videos/test_video_1.mp4

# Jitter only
python eval/measure_jitter.py eval/videos/test_video_1.mp4
```

**Automated (always):** detection rate, recovery frames, jitter score, LSTM label timeline.

**Automated (with labels file):** static accuracy (incl. Watch Tap), L/R assignment, Pull_Lever segment accuracy, idle false-trigger rate.  
Use `eval/videos/test_video_1_labels.json` (auto-detected as `<stem>_labels.json`, or pass `--labels`). Windows are approximate intent spans — recording does **not** need exact 3s/5s holds.

Video 1 script uses **Watch Tap** (heuristic) in place of Turn_Key for gameplay scoring. Layer A still reports LSTM class `Turn_Key` until that class is removed from the model.

### Annotated export (visual check)

Replay a test video with the live-server HUD plus a bottom **GT LABEL** line from the labels file. Useful for checking whether auto-eval windows look right.

```bash
python eval/export_annotated_video.py eval/videos/test_video_1.mp4 --version v1
# → eval/exports/v1/test_video_1_annotated.mp4
```

Open the mp4 in QuickTime and scrub. Top-left = detections; bottom = intended label window.

---

## Layer C

No script here — live Unity + you performing the gameplay checklist
(Watch Tap ×3, Pull_Lever ×3, etc. — see the evaluation plan).
Fill the spreadsheet from trial notes / recordings.
