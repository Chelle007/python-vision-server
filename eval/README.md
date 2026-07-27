# Evaluation tooling (agent / session playbook)

Separate from `scripts/` (live server / record / train). Implements **Layers A and B**
from the Gesture Model Evaluation Plan. **Layer C** is live Unity (no script here).

**If you are an agent asked to “run eval” / “baseline” / “log results”:** start here,
then follow the checklist for Quick Check vs Full Run.

---

## Related docs (outside this folder)

| Doc | Path |
| --- | ---- |
| Evaluation plan (targets, scripts, when to Full Run) | `FYP Repo/Gesture_Model_Evaluation_Plan.md` |
| Results spreadsheet | `FYP Repo/Gesture_Model_Evaluation_Log.xlsx` |

Git repo for this code: `python-vision-server` (commit hash goes in Excel **Commit** column).

---

## Layout

```
eval/
├── README.md                     # this file
├── run_layer_a.py                # Layer A — offline LSTM test split
├── run_layer_b.py                # Layer B — video metrics (+ optional labels)
├── measure_jitter.py             # jitter-only helper
├── export_annotated_video.py     # HUD + GT label → mp4 for visual check
├── videos/                       # fixed test videos (gitignored *.mp4/*.mov)
│   ├── .gitkeep
│   ├── test_video_1.mp4          # main sequence (3× script) — record once
│   ├── test_video_1_labels.json  # intent windows for auto segment scoring
│   ├── test_video_2.mov          # tracking stress
│   └── test_video_3.mov          # negative / false-trigger
└── exports/                      # annotated replays (gitignored *.mp4)
    └── v1/                       # one folder per eval version
        └── test_video_1_annotated.mp4
```

Shared logic: `src/vision_server/evaluation/`  
(`dataset.py`, `lstm_metrics.py`, `jitter.py`, `video_metrics.py`,
`segment_scoring.py`, `export_annotated.py`)

---

## Prerequisites

```bash
cd python-vision-server
source .venv/bin/activate
pip install -e .
```

| Need | For |
| ---- | --- |
| `data/` clip folders + `models/escape_gestures.keras` | Layer A |
| `eval/videos/test_video_{1,2,3}.*` | Layer B Full Run |
| `eval/videos/test_video_1_labels.json` | Layer B segment scores (static / Watch Tap / L/R / Pull_Lever / idle FP) |
| Unity + vision server | Layer C (human) |

Videos may be `.mp4` or `.mov` — OpenCV accepts both. Prefer names `test_video_N.*`.

**Do not commit** large videos/exports (already in `.gitignore`). **Do commit** labels JSON + scripts.

---

## Current baseline (v1)

As of 2026-07-12 / commit `1a6ba4b`:

- Layer **A** + **B** logged in Excel (Full Run partial — **C still pending**)
- Video 1 has labels + annotated export under `eval/exports/v1/`
- **Jitter:** no absolute pass/fail target — compare to baseline later (lower = better)
- Gameplay puzzle gesture under test is **Watch Tap** (heuristic), not LSTM `Turn_Key`. Layer A still scores `Idle` / `Turn_Key` / `Pull_Lever` until `Turn_Key` is retired from the model.

---

## When someone says “run eval”

1. Read `Gesture_Model_Evaluation_Plan.md` trigger table → **Quick Check** vs **Full Run**.
2. Default to Quick Check unless they say baseline / demo / milestone / Full Run.
3. Run the matching checklist below.
4. Log into `Gesture_Model_Evaluation_Log.xlsx` (see [Excel logging](#excel-logging)).
5. Record `git rev-parse --short HEAD` in **Commit**; append `+dirty` if uncommitted changes affect the run.

### Full Run checklist (A → B → C → Excel)

```bash
cd python-vision-server
source .venv/bin/activate

# A
python eval/run_layer_a.py

# B (all three; video 1 auto-loads labels JSON if present)
python eval/run_layer_b.py eval/videos/test_video_1.mp4
python eval/run_layer_b.py eval/videos/test_video_2.mov   # or .mp4
python eval/run_layer_b.py eval/videos/test_video_3.mov

# Optional visual QA for labels / detections
python eval/export_annotated_video.py eval/videos/test_video_1.mp4 --version vN
# → eval/exports/vN/test_video_1_annotated.mp4  (scrub in QuickTime)
```

Then:

- **Layer C (human):** Unity + server; script in the plan (static ×2, grab/release ×3, idle 30s, Watch Tap ×3, Pull_Lever ×3); Full Run = 3 trials + screen record `layer_c_vN_….mp4`.
- **Excel:** fill Comparison Log + Layer A/B/C sheets; use **N/A** for columns that do not apply; color Pass/Fail and Comparison Log metrics vs targets (see below).

### Quick Check checklist (typical)

```bash
python eval/run_layer_a.py                                          # if ML change
python eval/run_layer_b.py eval/videos/test_video_1.mp4              # if tracking/heuristic change
# Layer C: 1 trial, no recording, if integration/feel change
```

Skip videos 2/3 unless the change is about robustness or false triggers.

---

## Layer A — Offline LSTM

```bash
python eval/run_layer_a.py
# optional: --model models/escape_gestures.keras
```

- Same split as training: `test_size=0.2`, `random_state=42` via `evaluation/dataset.py`.
- Prints overall / per-class accuracy, Idle→action FP, confusion matrix + spreadsheet paste line.
- **Targets:** overall ≥ 85%; each class ≥ 80%; FP as low as possible (Excel soft bar ≤ 10% for green).

---

## Layer B — Fixed test videos

### Commands

```bash
python eval/run_layer_b.py eval/videos/test_video_1.mp4
python eval/run_layer_b.py eval/videos/test_video_1.mp4 --no-jitter   # faster re-run
python eval/run_layer_b.py eval/videos/test_video_1.mp4 --labels path/to.json
python eval/run_layer_b.py eval/videos/test_video_1.mp4 --no-labels
python eval/measure_jitter.py eval/videos/test_video_1.mp4
```

Default: selfie-flip on (matches live server). Use `--no-mirror` only if diagnosing flip issues.

### What is automated

| Metric | Always | Needs `*_labels.json` |
| ------ | ------ | --------------------- |
| Detection rate | ✅ | |
| Recovery frames | ✅ | |
| Jitter score | ✅ (unless `--no-jitter`) | |
| LSTM histogram / transitions | ✅ | |
| Static gesture acc (fist/index/peace/open) | | ✅ |
| Watch Tap acc | | ✅ (counted in “static incl. Watch Tap”) |
| L/R assignment | | ✅ |
| Pull_Lever segment acc (`LSTM Acc (video)`) | | ✅ |
| Idle / idleRandom false-trigger rate | | ✅ |

**False positives in plan sense** = Watch Tap + LSTM actions (`Pull_Lever` / `Turn_Key`), **not** fist/open-palm lighting up during waving.

### Labels philosophy (important)

- Recording does **not** need exact 3s/5s holds — order + clear gestures matter.
- Labels are **approximate intent windows** (`start`/`end` seconds + `label`), not stopwatch ground truth.
- File naming: `test_video_1.mp4` → `test_video_1_labels.json` (auto-detected).
- Segment “hit” if the intended signal fires on enough frames in the window (default threshold ~35% in `segment_scoring.py`).
- `rightInspect` / idle segments are tracked for FP or left unscored as appropriate.
- Videos 2–3 usually have **no** labels → put **N/A** in L/R, static, LSTM segment columns. Video 3 can still log LSTM FP from the histogram (0 PL/TK frames).

### Video roles

| File | Role |
| ---- | ---- |
| `test_video_1` | Main benchmark — 3× fixed script (idle/static → Watch Tap → Pull_Lever) |
| `test_video_2` | Tracking stress (enter/leave, fast motion, occlusion) |
| `test_video_3` | Negative — wave/point/random; expect no Watch Tap / LSTM false actions |

### Annotated export (visual check)

```bash
python eval/export_annotated_video.py eval/videos/test_video_1.mp4 --version v1
# → eval/exports/v1/test_video_1_annotated.mp4
```

- Top-left: live-server-style HUD (detections).
- Bottom: `GT LABEL: … [start-end] t=…` from labels JSON.
- Prefer **export** over live preview so the user can scrub (especially Pull_Lever windows).

### Layer B targets (from plan)

| Metric | Target |
| ------ | ------ |
| Detection rate | ≥ 90% (video 1 primary) |
| L/R assignment | ≥ 95% |
| Static (+ Watch Tap) | ≥ 85% |
| LSTM per class on video (e.g. Pull_Lever) | ≥ 85% |
| Recovery | &lt; ~30 frames @ 60fps (use measured FPS context) |
| Jitter | **Lower than previous baseline** (no absolute cutoff) |
| False trigger | As low as possible |

---

## Layer C — Live gameplay

No automation in `eval/`. User runs Unity + `python scripts/run_server.py`.

Full Run: 3 trials, average metrics, archive recording, save e.g. `layer_c_v1_baseline.mp4`.  
Script order (plan): left static ×2 each → fist/open ×3 → idle random 30s → Watch Tap ×3 → Pull_Lever ×3.

Agent fills Excel from user-reported counts / notes / recording filename. Latency columns optional if not measured → **N/A**.

### Recording + feel scoring (agreed convention)

**Do not use OBS for scored feel/latency** — it makes the machine laggy and contaminates “feels responsive?”.

| Step | How | Why |
| ---- | --- | --- |
| **1. Score feel** | Run trials **without** any screen recorder | True responsiveness / flicker / lag notes |
| **2. Archive clip** | Re-run (or one dedicated take) with **macOS `Cmd+Shift+5`** → Record Selected Portion (or QuickTime). Prefer Unity (+ optional vision HUD side-by-side). | Light enough for a log/demo file |
| **3. Excel Notes** | Always write the method, e.g. `Feel scored without recorder; archive via Cmd+Shift+5 (not OBS).` | Next session / agent must give the same advice |

If Watch Tap is **not wired in Unity yet**: Unity clip for playable gestures; Watch Tap can be detection-only from vision HUD (note `Watch Tap: detection-only, not in Unity yet`) or **N/A** until integrated.

**If the user asks how to record Layer C:** answer with this convention (score without recording → archive with Cmd+Shift+5 → note it in Excel). Do not recommend OBS for scored feel.

---

## Excel logging

File: `FYP Repo/Gesture_Model_Evaluation_Log.xlsx`

| Sheet | Use |
| ----- | --- |
| Comparison Log | One row per version/session summary |
| Layer A - LSTM | Per-run A metrics |
| Layer B - Test Video | One row **per video** (1 / 2 / 3) for Full Run |
| Layer C - Live Gameplay | Live trials |

### Conventions

- **Commit** column: `git rev-parse --short HEAD`; add `+dirty` if relevant uncommitted changes.
- **N/A** for columns that do not apply (layer not run, no labels, not a negative clip for FP, etc.). Do not leave ambiguous blanks on filled rows.
- **PASS** = green fill; **FAIL** = red fill.
- Comparison Log metric colors vs targets:
  - Higher-is-better green if meet/exceed: A Acc ≥85%, B Det ≥90%, B LSTM ≥85%, C Success ≥85%.
  - Lower-is-better green if ≤ soft bar: A/C FP ≤10%.
  - B Jitter: leave uncolored (relative to baseline only).
- Versioning: `v1` = first baseline; bump `v2`, … when logging a new Full Run after a meaningful change.
- Exports folder version should match Excel version (`eval/exports/v2/` for v2).

### Spreadsheet paste hints

Scripts print lines like:

```text
--- Spreadsheet paste (Layer A) ---
overall=… Idle=… Turn_Key=… Pull_Lever=… FP_Idle_to_action=…

--- Spreadsheet paste (Layer B auto) ---
detection=… recovery_frames=… jitter=…

--- Spreadsheet paste (Layer B labels) ---
lr=… static_incl_watchTap=… lstm_Pull_Lever=… false_trigger=…
```

---

## Command cheat sheet

```bash
cd python-vision-server && source .venv/bin/activate

python eval/run_layer_a.py
python eval/run_layer_b.py eval/videos/test_video_1.mp4
python eval/run_layer_b.py eval/videos/test_video_2.mov
python eval/run_layer_b.py eval/videos/test_video_3.mov
python eval/measure_jitter.py eval/videos/test_video_1.mp4
python eval/export_annotated_video.py eval/videos/test_video_1.mp4 --version v1

git rev-parse --short HEAD
git status -sb   # remember +dirty if needed
```

---

## Design notes for future changes

- Prefer extending `src/vision_server/evaluation/` and thin `eval/*.py` CLIs.
- New LSTM / puzzle gestures: append segments at the **end** of video 1 labels + plan script (newer = further back); keep early idle/static comparable.
- Re-record videos only if webcam/setup changes; otherwise keep fixed files for fair comparison.
- After changing label windows, re-run `run_layer_b.py` on video 1 and optionally re-export annotated mp4 before updating Excel.
