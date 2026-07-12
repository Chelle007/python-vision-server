"""Layer B — landmark jitter score on fixed test videos."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision_server.tracking import create_hands

# MediaPipe hand landmark indices
WRIST = 0
MIDDLE_MCP = 9

# Frames where hand-size-normalized movement is below this count as "still"
# (Idle-ish). Intentional gesture motion is excluded from the jitter average.
STILL_MOTION_THRESHOLD = 0.02


@dataclass
class JitterResult:
    overall_jitter: float
    still_frame_pairs: int
    detected_frames: int
    total_frames: int
    detection_rate: float

    def format_report(self) -> str:
        return "\n".join(
            [
                "=== Layer B — Jitter ===",
                f"Frames: {self.total_frames}  |  hand detected: {self.detected_frames}  "
                f"({self.detection_rate:.1%})",
                f"Still frame-pairs used: {self.still_frame_pairs}",
                f"Overall jitter score: {self.overall_jitter:.6f}  (lower = smoother)",
            ]
        )


def _landmark_xy(hand_landmarks) -> np.ndarray:
    """(21, 2) array of normalized x,y."""
    return np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float64)


def _hand_size(xy: np.ndarray) -> float:
    size = float(np.linalg.norm(xy[MIDDLE_MCP] - xy[WRIST]))
    return max(size, 1e-6)


def measure_jitter(
    video_path: str,
    *,
    still_threshold: float = STILL_MOTION_THRESHOLD,
    max_num_hands: int = 1,
) -> JitterResult:
    """
    Replay a video through MediaPipe Hands and compute a size-normalized
    landmark jitter score, mainly on still-hand frame pairs.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    hands = create_hands(max_num_hands=max_num_hands)
    prev_xy: np.ndarray | None = None
    jitter_values: list[float] = []
    detected = 0
    total = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if not result.multi_hand_landmarks:
                prev_xy = None
                continue

            detected += 1
            xy = _landmark_xy(result.multi_hand_landmarks[0])
            size = _hand_size(xy)

            if prev_xy is not None:
                per_lm = np.linalg.norm(xy - prev_xy, axis=1)
                jitter_raw = float(np.mean(per_lm))
                jitter_norm = jitter_raw / size
                # Only accumulate on still-ish pairs (Idle segments)
                if jitter_norm < still_threshold:
                    jitter_values.append(jitter_norm)

            prev_xy = xy
    finally:
        cap.release()
        hands.close()

    overall = float(np.mean(jitter_values)) if jitter_values else float("nan")
    detection_rate = (detected / total) if total else 0.0

    return JitterResult(
        overall_jitter=overall,
        still_frame_pairs=len(jitter_values),
        detected_frames=detected,
        total_frames=total,
        detection_rate=detection_rate,
    )
