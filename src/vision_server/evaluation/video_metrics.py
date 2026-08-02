"""Layer B — automated metrics from replaying a fixed test video."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from vision_server.config import (
    MAX_NUM_FACES,
    MAX_NUM_HANDS,
    MODEL_PATH,
)
from vision_server.evaluation.jitter import measure_jitter
from vision_server.gestures.dynamic import GestureLSTM
from vision_server.tracking import (
    PlayerLock,
    collect_faces,
    collect_hands,
    create_face_mesh,
    create_hands,
)


@dataclass
class VideoEvalResult:
    video_path: str
    total_frames: int
    detected_frames: int
    detection_rate: float
    recovery_frames: list[int] = field(default_factory=list)
    mean_recovery_frames: float = float("nan")
    jitter_score: float = float("nan")
    mean_proc_ms: float = float("nan")
    p95_proc_ms: float = float("nan")
    player_lock: bool = False
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
        mean_ms = (
            f"{self.mean_proc_ms:.2f}" if not np.isnan(self.mean_proc_ms) else "n/a"
        )
        p95_ms = (
            f"{self.p95_proc_ms:.2f}" if not np.isnan(self.p95_proc_ms) else "n/a"
        )
        lock_mode = "PlayerLock ON" if self.player_lock else "PlayerLock OFF (v1 proxy)"
        lines = [
            "=== Layer B — Test Video Metrics ===",
            f"Video: {self.video_path}",
            f"Mode: {lock_mode}",
            f"Frames: {self.total_frames}",
            f"Detection rate: {self.detection_rate:.4f}  "
            f"({self.detected_frames}/{self.total_frames})  (target ≥ 0.90)",
            f"Mean recovery (frames after leave): {recovery}  "
            f"(target < ~30 @ 60fps)",
            f"Jitter score: {jitter}",
            f"Proc latency mean: {mean_ms} ms  "
            f"(vision-server ms/frame; soft bar ≤ ~33 ms @ 30fps)",
            f"Proc latency p95:  {p95_ms} ms",
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
                "Proc latency is vision-server processing only (not Unity E2E).",
            ]
        )
        return "\n".join(lines)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    arr = np.sort(np.asarray(values, dtype=float))
    idx = min(len(arr) - 1, max(0, int(round((p / 100.0) * (len(arr) - 1)))))
    return float(arr[idx])


def evaluate_test_video(
    video_path: str,
    *,
    run_lstm: bool = True,
    run_jitter: bool = True,
    model_path: str = MODEL_PATH,
    max_num_hands: int = 2,
    mirror: bool = True,
    player_lock: bool = True,
) -> VideoEvalResult:
    """
    Replay video through MediaPipe (+ optional LSTM / PlayerLock).

    Automates: detection rate, recovery time, jitter, LSTM timeline,
    and vision-server per-frame proc latency (mean / p95 ms).

    ``player_lock=True`` matches the live server (faces + hands + PlayerLock).
    ``player_lock=False`` is the pre-lock 2-hand path (v1 proxy).

    ``mirror=True`` matches the live server (selfie flip) so handedness
    labels align with gameplay.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    if player_lock:
        hands = create_hands(max_num_hands=MAX_NUM_HANDS)
        face_mesh = create_face_mesh(max_num_faces=MAX_NUM_FACES)
        lock_tracker = PlayerLock()
    else:
        hands = create_hands(max_num_hands=max_num_hands)
        face_mesh = None
        lock_tracker = None

    lstm = GestureLSTM(model_path=model_path) if run_lstm else None

    total = 0
    detected = 0
    recovery_samples: list[int] = []
    frames_since_lost: int | None = None
    lstm_counts: Counter[str] = Counter()
    transitions: list[tuple[int, str]] = []
    last_lstm: str | None = None
    proc_ms_samples: list[float] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1

            t0 = time.perf_counter()
            if mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb)
            right_landmarks = None
            hand_present = False

            if player_lock:
                face_results = face_mesh.process(rgb)
                lock = lock_tracker.update(
                    collect_faces(face_results),
                    collect_hands(hand_results),
                )
                if lock.flush_lstm and lstm is not None:
                    lstm.flush()
                if lock.left is not None or lock.right is not None:
                    hand_present = True
                if lock.right is not None:
                    right_landmarks = lock.right.landmarks
            else:
                if hand_results.multi_hand_landmarks:
                    hand_present = True
                    if hand_results.multi_handedness:
                        for hand_info, lms in zip(
                            hand_results.multi_handedness,
                            hand_results.multi_hand_landmarks,
                        ):
                            if hand_info.classification[0].label.lower() == "right":
                                right_landmarks = lms.landmark
                                break
                    if right_landmarks is None:
                        right_landmarks = hand_results.multi_hand_landmarks[0].landmark

            if not hand_present:
                if frames_since_lost is None:
                    frames_since_lost = 0
                else:
                    frames_since_lost += 1
                if lstm is not None:
                    lstm.register_hand_lost()
                proc_ms_samples.append((time.perf_counter() - t0) * 1000.0)
                continue

            detected += 1
            if frames_since_lost is not None:
                recovery_samples.append(frames_since_lost)
                frames_since_lost = None

            if lstm is not None:
                if right_landmarks is not None:
                    lstm.register_hand_seen()
                    label = lstm.predict(right_landmarks)
                else:
                    lstm.register_hand_lost()
                    label = lstm.get_overlay_label()
                    if label not in lstm.classes:
                        label = "Idle"
                lstm_counts[label] += 1
                if label != last_lstm:
                    transitions.append((total, label))
                    last_lstm = label

            proc_ms_samples.append((time.perf_counter() - t0) * 1000.0)
    finally:
        cap.release()
        hands.close()
        if face_mesh is not None:
            face_mesh.close()

    detection_rate = (detected / total) if total else 0.0
    mean_recovery = (
        float(np.mean(recovery_samples)) if recovery_samples else float("nan")
    )
    mean_proc_ms = (
        float(np.mean(proc_ms_samples)) if proc_ms_samples else float("nan")
    )
    p95_proc_ms = _percentile(proc_ms_samples, 95.0)

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
        mean_proc_ms=mean_proc_ms,
        p95_proc_ms=p95_proc_ms,
        player_lock=player_lock,
        lstm_counts=dict(lstm_counts),
        lstm_transitions=transitions,
    )
