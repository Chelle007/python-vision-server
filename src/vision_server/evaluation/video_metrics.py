"""Layer B — automated metrics from replaying a fixed test video."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from vision_server.config import MODEL_PATH
from vision_server.evaluation.jitter import measure_jitter
from vision_server.gestures.dynamic import GestureLSTM
from vision_server.tracking import create_hands


@dataclass
class VideoEvalResult:
    video_path: str
    total_frames: int
    detected_frames: int
    detection_rate: float
    recovery_frames: list[int] = field(default_factory=list)
    mean_recovery_frames: float = float("nan")
    jitter_score: float = float("nan")
    lstm_counts: dict[str, int] = field(default_factory=dict)
    lstm_transitions: list[tuple[int, str]] = field(default_factory=list)

    def format_report(self) -> str:
        recovery = (
            f"{self.mean_recovery_frames:.1f}"
            if not np.isnan(self.mean_recovery_frames)
            else "n/a"
        )
        jitter = (
            f"{self.jitter_score:.6f}" if not np.isnan(self.jitter_score) else "n/a"
        )
        lines = [
            "=== Layer B — Test Video Metrics ===",
            f"Video: {self.video_path}",
            f"Frames: {self.total_frames}",
            f"Detection rate: {self.detection_rate:.4f}  "
            f"({self.detected_frames}/{self.total_frames})  (target ≥ 0.90)",
            f"Mean recovery (frames after leave): {recovery}  "
            f"(target < ~30 @ 60fps)",
            f"Jitter score: {jitter}",
            "",
            "LSTM label histogram (for manual segment scoring):",
        ]
        if self.lstm_counts:
            for label, count in sorted(self.lstm_counts.items()):
                lines.append(f"  {label:20s}  {count}")
        else:
            lines.append("  (LSTM not run)")

        if self.lstm_transitions:
            lines.append("")
            lines.append("LSTM transitions (frame → label):")
            for frame_i, label in self.lstm_transitions[:40]:
                lines.append(f"  {frame_i:6d}  {label}")
            if len(self.lstm_transitions) > 40:
                lines.append(f"  ... ({len(self.lstm_transitions) - 40} more)")

        lines.extend(
            [
                "",
                "Note: pass --labels / <stem>_labels.json for static / Watch Tap /",
                "L-R / Pull_Lever segment accuracy and idle false-trigger rate.",
            ]
        )
        return "\n".join(lines)


def evaluate_test_video(
    video_path: str,
    *,
    run_lstm: bool = True,
    run_jitter: bool = True,
    model_path: str = MODEL_PATH,
    max_num_hands: int = 2,
    mirror: bool = True,
) -> VideoEvalResult:
    """
    Replay video through MediaPipe (+ optional LSTM).

    Automates: detection rate, recovery time, jitter, LSTM timeline.
    Leaves segment-level accuracy for manual scoring.

    ``mirror=True`` matches the live server (selfie flip) so handedness
    labels align with gameplay.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    hands = create_hands(max_num_hands=max_num_hands)
    lstm = GestureLSTM(model_path=model_path) if run_lstm else None

    total = 0
    detected = 0
    recovery_samples: list[int] = []
    frames_since_lost: int | None = None
    lstm_counts: Counter[str] = Counter()
    transitions: list[tuple[int, str]] = []
    last_lstm: str | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            if mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if not result.multi_hand_landmarks:
                if frames_since_lost is None:
                    frames_since_lost = 0
                else:
                    frames_since_lost += 1
                if lstm is not None:
                    lstm.register_hand_lost()
                continue

            detected += 1
            if frames_since_lost is not None:
                recovery_samples.append(frames_since_lost)
                frames_since_lost = None

            right_landmarks = None
            if result.multi_handedness:
                for hand_info, lms in zip(
                    result.multi_handedness, result.multi_hand_landmarks
                ):
                    if hand_info.classification[0].label.lower() == "right":
                        right_landmarks = lms.landmark
                        break
            if right_landmarks is None:
                right_landmarks = result.multi_hand_landmarks[0].landmark

            if lstm is not None:
                lstm.register_hand_seen()
                label = lstm.predict(right_landmarks)
                lstm_counts[label] += 1
                if label != last_lstm:
                    transitions.append((total, label))
                    last_lstm = label
    finally:
        cap.release()
        hands.close()

    detection_rate = (detected / total) if total else 0.0
    mean_recovery = (
        float(np.mean(recovery_samples)) if recovery_samples else float("nan")
    )

    jitter_score = float("nan")
    if run_jitter:
        jitter = measure_jitter(video_path, max_num_hands=1)
        jitter_score = jitter.overall_jitter

    return VideoEvalResult(
        video_path=video_path,
        total_frames=total,
        detected_frames=detected,
        detection_rate=detection_rate,
        recovery_frames=recovery_samples,
        mean_recovery_frames=mean_recovery,
        jitter_score=jitter_score,
        lstm_counts=dict(lstm_counts),
        lstm_transitions=transitions,
    )
