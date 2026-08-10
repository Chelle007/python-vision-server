"""Export a test video with live-server-style HUD + ground-truth label overlay."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp

from vision_server.config import MODEL_PATH
from vision_server.evaluation.segment_scoring import (
    default_labels_path_for_video,
    load_segments,
)
from vision_server.gestures.dynamic import GestureLSTM
from vision_server.gestures.hand import (
    NONE,
    GestureDebouncer,
    classify_hand,
    rules_from_label,
)
from vision_server.gestures.hand.cursor_fields import (
    apply_cursor_fields,
    reset_last_point,
)
from vision_server.gestures.hand.geometry import get_hand_rotation
from vision_server.gestures.hand.watch_tap import apply_watch_tap_fields
from vision_server.overlay import build_overlay_lines, draw_overlay, draw_text_with_bg
from vision_server.tracking import create_hands
from vision_server.udp import default_payload


def _active_label(segments: list[dict], t: float) -> tuple[str | None, float | None, float | None]:
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        if start <= t < end:
            return str(seg["label"]), start, end
    return None, None, None


def export_annotated_video(
    video_path: str | Path,
    output_path: str | Path,
    *,
    labels_path: str | Path | None = None,
    model_path: str = MODEL_PATH,
    mirror: bool = True,
    draw_landmarks: bool = True,
) -> Path:
    """
    Replay ``video_path`` through the gesture pipeline and write an annotated mp4.

    Overlay matches the live server HUD, plus a ``GT LABEL:`` line when a labels
    file is available.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if labels_path is None:
        labels_path = default_labels_path_for_video(video_path)
    segments: list[dict] = load_segments(labels_path) if labels_path else []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    hands = create_hands(max_num_hands=2)
    lstm = GestureLSTM(model_path=model_path)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    last_point = [-1.0, -1.0]
    # Same commit delay as the live server, so the exported HUD shows the
    # labels Unity would actually have received rather than the raw rules.
    debouncers = {"left": GestureDebouncer(), "right": GestureDebouncer()}

    frame_i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            t = frame_i / fps
            frame_i += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)

            data = default_payload()
            right_hand_seen = False
            lstm_display = "Idle"
            left_landmarks = None
            right_landmarks = None

            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                ):
                    landmarks = hand_landmarks.landmark
                    side = handedness.classification[0].label.lower()

                    if draw_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                    if side == "left":
                        left_landmarks = landmarks
                    else:
                        right_landmarks = landmarks
                        right_hand_seen = True

            # Both sides are debounced every frame, absent ones included, so a
            # dropout is counted out rather than instantly ending a gesture.
            left_label = debouncers["left"].update(
                classify_hand(left_landmarks) if left_landmarks is not None else NONE
            )
            right_label = debouncers["right"].update(
                classify_hand(right_landmarks) if right_landmarks is not None else NONE
            )

            left_gestures = rules_from_label(left_label)
            data["leftFist"] = left_gestures["fist"]
            data["leftOpenPalm"] = left_gestures["open_palm"]
            data["leftIndexUp"] = left_gestures["index_up"]
            data["leftPeace"] = left_gestures["peace"]
            data["leftThumbsUp"] = left_gestures["thumbs_up"]
            data["leftRockSign"] = left_gestures["rock_sign"]
            data["leftIndexLeft"] = left_gestures["index_left"]
            data["leftIndexRight"] = left_gestures["index_right"]
            data["moveGesture"] = left_label
            data["moveGestureRaw"] = debouncers["left"].raw

            right_gestures = rules_from_label(right_label)
            data["rightFist"] = right_gestures["fist"]
            data["rightOpenPalm"] = right_gestures["open_palm"]
            data["rightIndexUp"] = right_gestures["index_up"]
            data["rightPeace"] = right_gestures["peace"]
            data["rightThumbsUp"] = right_gestures["thumbs_up"]
            data["rightRockSign"] = right_gestures["rock_sign"]
            data["rightIndexLeft"] = right_gestures["index_left"]
            data["rightIndexRight"] = right_gestures["index_right"]
            data["actionGesture"] = right_label
            data["actionGestureRaw"] = debouncers["right"].raw

            if right_landmarks is not None:
                lstm.register_hand_seen()
                apply_cursor_fields(data, right_landmarks, right_gestures, last_point)
                pitch, yaw, roll = get_hand_rotation(right_landmarks)
                data["fistRotX"] = round(pitch, 3)
                data["fistRotY"] = round(yaw, 3)
                data["fistRotZ"] = round(roll, 3)
                lstm_display = lstm.predict(right_landmarks)
                data["lstm_gesture"] = (
                    lstm_display if lstm_display in lstm.classes else "Idle"
                )

            if apply_watch_tap_fields(data, left_landmarks, right_landmarks):
                reset_last_point(last_point)
                lstm_display = "Idle"

            if not right_hand_seen:
                reset_last_point(last_point)
                lstm.register_hand_lost()
                lstm_display = lstm.get_overlay_label()
                data["lstm_gesture"] = (
                    lstm_display if lstm_display in lstm.classes else "Idle"
                )

            overlay_lines = build_overlay_lines(data, lstm_display)
            draw_overlay(frame, overlay_lines)

            # Ground-truth intent label + timestamp (for checking auto-eval windows)
            gt, g_start, g_end = _active_label(segments, t)
            gt_text = (
                f"GT LABEL: {gt} [{g_start:.1f}-{g_end:.1f}s]  t={t:.1f}s"
                if gt
                else f"GT LABEL: (none)  t={t:.1f}s"
            )
            draw_text_with_bg(frame, gt_text, 10, height - 24, (0, 200, 255), scale=0.7)

            writer.write(frame)

            if total and frame_i % 300 == 0:
                print(f"  … {frame_i}/{total} frames ({100 * frame_i / total:.0f}%)")
    finally:
        cap.release()
        writer.release()
        hands.close()

    return output_path
