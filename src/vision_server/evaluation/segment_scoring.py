"""Score Layer B metrics against approximate intent-segment labels."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from vision_server.config import MODEL_PATH
from vision_server.gestures.dynamic import GestureLSTM
from vision_server.gestures.hand import (
    NONE,
    GestureDebouncer,
    classify_hand,
    rules_from_label,
)
from vision_server.gestures.hand.watch_tap import is_watch_tap
from vision_server.tracking import create_hands

# Labels that count toward static-gesture accuracy. Keyed by the name used in
# the clip annotations, so an entry with no annotated segments simply never
# scores — the inventory gestures are listed ahead of the clips being recorded,
# which is also what makes their miss rate measurable at all (they were
# calibrated against false positives only; see THUMB_UP_CLEARANCE).
STATIC_LABELS = {
    "leftFist": ("left", "fist"),
    "leftIndexUp": ("left", "index_up"),
    "leftPeace": ("left", "peace"),
    "leftThumbsUp": ("left", "thumbs_up"),
    "leftRockSign": ("left", "rock_sign"),
    "leftIndexLeft": ("left", "index_left"),
    "leftIndexRight": ("left", "index_right"),
    "rightFist": ("right", "fist"),
    "rightOpenPalm": ("right", "open_palm"),
}

# Segments where Watch Tap / LSTM must stay off
NEGATIVE_LABELS = {"idle", "idleRandom"}


@dataclass
class SegmentScore:
    label: str
    start: float
    end: float
    hit: bool | None
    detail: str = ""


@dataclass
class SegmentEvalResult:
    labels_path: str
    static_correct: int = 0
    static_total: int = 0
    watch_tap_correct: int = 0
    watch_tap_total: int = 0
    pull_lever_correct: int = 0
    pull_lever_total: int = 0
    lr_correct: int = 0
    lr_total: int = 0
    false_trigger_frames: int = 0
    negative_frames: int = 0
    segments: list[SegmentScore] = field(default_factory=list)

    @property
    def static_accuracy(self) -> float:
        return self.static_correct / self.static_total if self.static_total else float("nan")

    @property
    def watch_tap_accuracy(self) -> float:
        return (
            self.watch_tap_correct / self.watch_tap_total
            if self.watch_tap_total
            else float("nan")
        )

    @property
    def pull_lever_accuracy(self) -> float:
        return (
            self.pull_lever_correct / self.pull_lever_total
            if self.pull_lever_total
            else float("nan")
        )

    @property
    def lstm_video_accuracy(self) -> float:
        """Pull_Lever segment accuracy (main LSTM class under test on Video 1)."""
        return self.pull_lever_accuracy

    @property
    def lr_assignment_accuracy(self) -> float:
        return self.lr_correct / self.lr_total if self.lr_total else float("nan")

    @property
    def false_trigger_rate(self) -> float:
        return (
            self.false_trigger_frames / self.negative_frames
            if self.negative_frames
            else float("nan")
        )

    @property
    def combined_static_including_watch_tap(self) -> float:
        """Static + Watch Tap segments — matches plan 'static gesture accuracy (incl. Watch Tap)'."""
        c = self.static_correct + self.watch_tap_correct
        t = self.static_total + self.watch_tap_total
        return c / t if t else float("nan")

    def format_report(self) -> str:
        def pct(x: float) -> str:
            return f"{x:.4f}" if x == x else "n/a"

        lines = [
            "=== Layer B — Segment labels scoring ===",
            f"Labels: {self.labels_path}",
            f"Static gesture accuracy: {pct(self.static_accuracy)}  "
            f"({self.static_correct}/{self.static_total})",
            f"Watch Tap accuracy:      {pct(self.watch_tap_accuracy)}  "
            f"({self.watch_tap_correct}/{self.watch_tap_total})",
            f"Static incl. Watch Tap:  {pct(self.combined_static_including_watch_tap)}",
            f"LSTM Pull_Lever acc:     {pct(self.pull_lever_accuracy)}  "
            f"({self.pull_lever_correct}/{self.pull_lever_total})",
            f"L/R assignment accuracy: {pct(self.lr_assignment_accuracy)}  "
            f"({self.lr_correct}/{self.lr_total})",
            f"False trigger rate (idle*): {pct(self.false_trigger_rate)}  "
            f"({self.false_trigger_frames}/{self.negative_frames} frames)",
            "",
            "Per-segment:",
        ]
        for s in self.segments:
            mark = "✓" if s.hit else ("·" if s.hit is None else "✗")
            lines.append(
                f"  {mark} [{s.start:6.1f}-{s.end:6.1f}] {s.label:14s}  {s.detail}"
            )
        return "\n".join(lines)


def load_segments(labels_path: str | Path) -> list[dict]:
    data = json.loads(Path(labels_path).read_text())
    return list(data["segments"])


def default_labels_path_for_video(video_path: str | Path) -> Path | None:
    """e.g. test_video_1.mp4 → test_video_1_labels.json beside it."""
    p = Path(video_path)
    candidate = p.with_name(f"{p.stem}_labels.json")
    return candidate if candidate.is_file() else None


def score_video_segments(
    video_path: str | Path,
    labels_path: str | Path,
    *,
    model_path: str = MODEL_PATH,
    mirror: bool = True,
    hit_threshold: float = 0.35,
) -> SegmentEvalResult:
    """
    Replay video; for each labeled window, check if the intended signal fires
    often enough (``hit_threshold`` of frames with usable hands).
    """
    segments = load_segments(labels_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hands = create_hands(max_num_hands=2)
    lstm = GestureLSTM(model_path=model_path)
    # Score the debounced labels, not the raw per-frame ones. The live server
    # only ever sends debounced booleans to Unity, so scoring the raw rules
    # here would measure a pipeline that does not exist.
    debouncers = {"left": GestureDebouncer(), "right": GestureDebouncer()}

    # Per-segment accumulators
    stats: list[dict] = []
    for seg in segments:
        stats.append(
            {
                "label": seg["label"],
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "gesture_hits": 0,
                "gesture_frames": 0,
                "lr_hits": 0,
                "lr_frames": 0,
                "fp_frames": 0,
                "neg_frames": 0,
            }
        )

    frame_i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_i / fps
            frame_i += 1

            active = [s for s in stats if s["start"] <= t < s["end"]]
            if not active:
                continue

            if mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            left = right = None
            if result.multi_hand_landmarks and result.multi_handedness:
                for hness, lms in zip(
                    result.multi_handedness, result.multi_hand_landmarks
                ):
                    side = hness.classification[0].label.lower()
                    if side == "left":
                        left = lms.landmark
                    else:
                        right = lms.landmark

            # Missing hands feed NONE so a dropout has to outlast the
            # off-count, exactly as in the live server.
            left_label = debouncers["left"].update(
                classify_hand(left) if left is not None else NONE
            )
            right_label = debouncers["right"].update(
                classify_hand(right) if right is not None else NONE
            )
            left_g = rules_from_label(left_label)
            right_g = rules_from_label(right_label)
            wt = is_watch_tap(left, right)

            lstm_label = "Idle"
            if right is not None:
                lstm.register_hand_seen()
                lstm_label = lstm.predict(right)
            else:
                lstm.register_hand_lost()

            for s in active:
                lab = s["label"]

                if lab in NEGATIVE_LABELS:
                    s["neg_frames"] += 1
                    if wt or lstm_label in ("Pull_Lever", "Turn_Key"):
                        s["fp_frames"] += 1
                    continue

                if lab in STATIC_LABELS:
                    side, rule_name = STATIC_LABELS[lab]
                    g = left_g if side == "left" else right_g
                    hand = left if side == "left" else right
                    # L/R: when any hand is seen, expected side should be present
                    if left is not None or right is not None:
                        s["lr_frames"] += 1
                        if hand is not None:
                            s["lr_hits"] += 1
                    if hand is None:
                        continue
                    s["gesture_frames"] += 1
                    if g and g.get(rule_name):
                        s["gesture_hits"] += 1
                    continue

                if lab == "watchTap":
                    s["gesture_frames"] += 1
                    if wt:
                        s["gesture_hits"] += 1
                    continue

                if lab == "Pull_Lever":
                    if right is None:
                        continue
                    s["gesture_frames"] += 1
                    if lstm_label == "Pull_Lever":
                        s["gesture_hits"] += 1
                    continue

                # rightInspect / unknown: not scored for accuracy
    finally:
        cap.release()
        hands.close()

    out = SegmentEvalResult(labels_path=str(labels_path))
    for s in stats:
        lab = s["label"]
        ratio = (
            s["gesture_hits"] / s["gesture_frames"] if s["gesture_frames"] else 0.0
        )
        hit: bool | None
        detail: str

        if lab in NEGATIVE_LABELS:
            hit = None
            detail = f"fp_frames={s['fp_frames']}/{s['neg_frames']}"
            out.false_trigger_frames += s["fp_frames"]
            out.negative_frames += s["neg_frames"]
        elif lab in STATIC_LABELS:
            hit = s["gesture_frames"] > 0 and ratio >= hit_threshold
            detail = f"fire={s['gesture_hits']}/{s['gesture_frames']} ({ratio:.0%})"
            out.static_total += 1
            if hit:
                out.static_correct += 1
            if s["lr_frames"]:
                out.lr_total += 1
                if s["lr_hits"] / s["lr_frames"] >= 0.5:
                    out.lr_correct += 1
        elif lab == "watchTap":
            hit = s["gesture_frames"] > 0 and ratio >= hit_threshold
            detail = f"fire={s['gesture_hits']}/{s['gesture_frames']} ({ratio:.0%})"
            out.watch_tap_total += 1
            if hit:
                out.watch_tap_correct += 1
        elif lab == "Pull_Lever":
            hit = s["gesture_frames"] > 0 and ratio >= hit_threshold
            detail = f"fire={s['gesture_hits']}/{s['gesture_frames']} ({ratio:.0%})"
            out.pull_lever_total += 1
            if hit:
                out.pull_lever_correct += 1
        else:
            hit = None
            detail = "not scored"
            ratio = float("nan")

        out.segments.append(
            SegmentScore(
                label=lab,
                start=s["start"],
                end=s["end"],
                hit=hit,
                detail=detail,
            )
        )

    return out
