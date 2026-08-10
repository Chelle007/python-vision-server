import cv2

from vision_server.config import CAMERA_HEIGHT, HEAD_LOOK_UP_PITCH_THRESHOLD

# Overlay sizes were tuned for ~720p previews; scale by frame height so 640x480
# (and other negotiated sizes) stay readable without dominating the image.
_UI_DESIGN_HEIGHT = 720.0


def _ui_scale(frame) -> float:
    h = float(frame.shape[0]) if frame is not None and frame.size else CAMERA_HEIGHT
    return max(0.5, min(1.25, h / _UI_DESIGN_HEIGHT))


def draw_text_with_bg(frame, text, x, y, color, scale=0.8, thickness=2, padding=8):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = y - text_h - padding
    bottom = y + baseline + padding
    cv2.rectangle(
        frame,
        (x - padding, top),
        (x + text_w + padding, bottom),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def lstm_display_color(gesture_label):
    if gesture_label in (
        "Turn_Key",
        "Pull_Lever",
        "Turn_Around_CW",
        "Turn_Around_CCW",
    ):
        return (0, 255, 0)
    if gesture_label == "No Model":
        return (0, 0, 255)
    return (255, 255, 255)


def _lock_status_label(status: str, lock_id: int) -> tuple[str, tuple[int, int, int]]:
    if status == "locked":
        return f"PLAYER LOCK #{lock_id}", (0, 255, 0)
    if status == "challenger":
        return "SWITCHING PLAYER...", (0, 165, 255)
    return "NO PLAYER LOCK", (160, 160, 160)


def _pitch_cal_message(status: str) -> tuple[str, tuple[int, int, int]] | None:
    # OpenCV putText is ASCII-only; avoid em-dash / unicode (shows as ???).
    if status == "idle":
        return None
    if status == "calibrating":
        return (
            "CALIBRATING - sit still, look at the screen",
            (0, 255, 255),
        )
    if status == "hold_still":
        return ("HOLD STILL - too much head motion", (0, 165, 255))
    if status == "calibrated":
        return ("CALIBRATED  (press C to redo)", (0, 255, 0))
    return None


def _gesture_debug_line(data: dict) -> str:
    """``MOVE fist(fist)  ACT none(peace)`` — committed label, raw in brackets.

    They differ exactly while a change is being counted out, so a bracket that
    never gets promoted is a threshold set too high, and one that flickers
    without moving the committed label is the filter doing its job.
    """
    move = str(data.get("moveGesture", "none"))
    move_raw = str(data.get("moveGestureRaw", "none"))
    action = str(data.get("actionGesture", "none"))
    action_raw = str(data.get("actionGestureRaw", "none"))
    return f"MOVE {move}({move_raw})  ACT {action}({action_raw})"


def build_overlay_lines(data: dict, lstm_display: str) -> list[tuple[str, tuple[int, int, int]]]:
    overlay_lines = [
        (f"AI LSTM: {lstm_display}", lstm_display_color(lstm_display)),
    ]

    if data.get("puzzle_active"):
        overlay_lines.append(("PUZZLE MODE: ON (P)", (0, 255, 255)))
    else:
        overlay_lines.append(("PUZZLE MODE: OFF (P)", (150, 150, 150)))

    action_hand = str(data.get("action_hand", "right")).upper()
    overlay_lines.append(
        (f"ACTION HAND: {action_hand} (H to swap)", (255, 200, 0))
    )

    status = data.get("lock_status", "unlocked")
    overlay_lines.append(_lock_status_label(status, data.get("lock_id", 0)))

    cal_msg = _pitch_cal_message(str(data.get("pitch_cal_status", "idle")))
    if cal_msg is not None:
        overlay_lines.append(cal_msg)

    thr = HEAD_LOOK_UP_PITCH_THRESHOLD
    is_calibrated = bool(data.get("pitch_calibrated", False))
    head_pitch = float(data.get("head_pitch", 0.0))
    # Look-up only meaningful on calibrated scale (rest ~0). Used to switch tilt labels.
    looking_up = is_calibrated and head_pitch <= -thr

    # left*/right* are role fields, so the label names the physical hand
    # currently holding that role rather than a fixed side.
    mv = str(data.get("move_hand", "left")).upper()
    ac = action_hand

    # Raw classifier output next to the committed label. Without both, a
    # gesture that fails to fire is indistinguishable from one the debouncer
    # is still counting, and the frame thresholds cannot be tuned by eye.
    overlay_lines.append((_gesture_debug_line(data), (170, 170, 170)))

    if data["leftFist"]:
        overlay_lines.append((f"{mv} FIST = MOVE", (0, 255, 0)))

    if data["leftIndexUp"]:
        overlay_lines.append((f"{mv} INDEX = JUMP", (0, 255, 0)))

    if data["leftPeace"]:
        overlay_lines.append((f"{mv} PEACE = CROUCH", (0, 255, 0)))

    if data["rightFist"]:
        overlay_lines.append((f"{ac} FIST = GRAB", (0, 255, 0)))

    if data["rightOpenPalm"]:
        overlay_lines.append((f"{ac} OPEN PALM = RELEASE", (0, 255, 0)))

    if data["rightPeace"]:
        overlay_lines.append((f"{ac} PEACE = STAND", (0, 255, 0)))

    if data["watchTap"]:
        overlay_lines.append(("WATCH TAP", (0, 255, 0)))

    if data["watchTapDistance"] is not None:
        overlay_lines.append(
            (f"Watch dist: {data['watchTapDistance']:.2f}", (200, 200, 200))
        )

    # Same label style for continuous turn vs look-up + tilt turn-back.
    if data["tilt_left"]:
        label = (
            "HEAD TILT LEFT = TURN BACK"
            if looking_up
            else "HEAD TILT LEFT = TURN"
        )
        overlay_lines.append((label, (0, 255, 0)))
    if data["tilt_right"]:
        label = (
            "HEAD TILT RIGHT = TURN BACK"
            if looking_up
            else "HEAD TILT RIGHT = TURN"
        )
        overlay_lines.append((label, (0, 255, 0)))

    return overlay_lines


def _lock_status_color(status: str) -> tuple[int, int, int]:
    if status == "locked":
        return (0, 255, 0)
    if status == "challenger":
        return (0, 165, 255)
    return (160, 160, 160)


def draw_lock_ring(
    frame,
    *,
    status: str,
    center: tuple[float, float] | None,
    ring_size: tuple[float, float],
    progress: float,
) -> None:
    """Camera-style corner brackets around the face — tracks it without covering it."""
    if center is None:
        return

    h, w = frame.shape[:2]
    s = _ui_scale(frame)
    color = _lock_status_color(status)

    box_w = max(ring_size[0] * 1.3, 0.08) * w
    box_h = max(ring_size[1] * 1.3, 0.1) * h
    cx = center[0] * w
    cy = center[1] * h

    x0 = int(cx - box_w / 2)
    y0 = int(cy - box_h / 2)
    x1 = int(cx + box_w / 2)
    y1 = int(cy + box_h / 2)

    corner = int(min(box_w, box_h) * 0.22)
    thickness = max(1, int(round(2 * s)))

    for (x, y), dx, dy in (
        ((x0, y0), 1, 1),
        ((x1, y0), -1, 1),
        ((x0, y1), 1, -1),
        ((x1, y1), -1, -1),
    ):
        cv2.line(frame, (x, y), (x + dx * corner, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy * corner), color, thickness, cv2.LINE_AA)

    if status == "challenger" and progress > 0:
        # Progress bar tracking the top edge as the takeover hold fills.
        fill_x1 = int(x0 + (x1 - x0) * min(1.0, max(0.0, progress)))
        bar_off = max(4, int(round(6 * s)))
        bar_thick = max(2, int(round(3 * s)))
        cv2.line(
            frame, (x0, y0 - bar_off), (fill_x1, y0 - bar_off), color, bar_thick, cv2.LINE_AA
        )



def draw_pitch_indicator(
    frame,
    head_pitch: float,
    *,
    threshold: float = HEAD_LOOK_UP_PITCH_THRESHOLD,
) -> None:
    """Right-side vertical meter for head_pitch (−1 bottom … +1 top inverted for readability).

    Image y grows downward, but the meter shows look-up toward the top of the bar
    so it matches the physical motion.
    """
    h, w = frame.shape[:2]
    s = _ui_scale(frame)
    bar_h = int(h * 0.35)
    bar_w = max(10, int(round(18 * s)))
    margin = max(8, int(round(16 * s)))
    x0 = w - margin - bar_w
    y0 = int(h * 0.12)
    y1 = y0 + bar_h
    x1 = x0 + bar_w
    tick = max(2, int(round(4 * s)))

    # Background track
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), 1)

    mid_y = (y0 + y1) // 2
    cv2.line(frame, (x0 - tick, mid_y), (x1 + tick, mid_y), (100, 100, 100), 1)

    # Threshold marks for look-up (negative pitch → upper half of meter)
    pitch = max(-1.0, min(1.0, float(head_pitch)))
    # Map pitch: -1 (look up) → top, +1 (look down) → bottom
    t = (1.0 - pitch) * 0.5  # 0 at +1(down), 1 at -1(up)
    marker_y = int(y0 + (1.0 - t) * bar_h)

    thr = max(0.01, float(threshold))
    # Look-up arm line at pitch = -thr
    t_up = (1.0 - (-thr)) * 0.5
    y_up = int(y0 + (1.0 - t_up) * bar_h)
    thr_tick = max(3, int(round(6 * s)))
    cv2.line(frame, (x0 - thr_tick, y_up), (x1 + thr_tick, y_up), (0, 255, 255), 1)

    looking_up = pitch <= -thr
    color = (0, 255, 255) if looking_up else (0, 200, 0)
    # Fill from center toward current pitch
    fill_y0 = min(marker_y, mid_y)
    fill_y1 = max(marker_y, mid_y)
    inset = max(1, int(round(2 * s)))
    if fill_y1 > fill_y0:
        cv2.rectangle(
            frame, (x0 + inset, fill_y0), (x1 - inset, fill_y1), color, -1
        )

    # Needle
    cv2.line(
        frame,
        (x0 - inset, marker_y),
        (x1 + inset, marker_y),
        (255, 255, 255),
        max(1, int(round(2 * s))),
    )

    label = f"{pitch:+.2f}"
    draw_text_with_bg(
        frame,
        label,
        x0 - int(round(70 * s)),
        y0 + int(round(18 * s)),
        color,
        scale=0.55 * s,
        thickness=1,
        padding=max(2, int(round(4 * s))),
    )
    draw_text_with_bg(
        frame,
        "UP",
        x0 - int(round(28 * s)),
        y0 + int(round(14 * s)),
        (180, 180, 180),
        scale=0.4 * s,
        thickness=1,
        padding=max(1, int(round(2 * s))),
    )
    draw_text_with_bg(
        frame,
        "DN",
        x0 - int(round(28 * s)),
        y1 + int(round(14 * s)),
        (180, 180, 180),
        scale=0.4 * s,
        thickness=1,
        padding=max(1, int(round(2 * s))),
    )


def draw_overlay(frame, overlay_lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    s = _ui_scale(frame)
    # Base values matched ~720p; shrink with frame height for 640x480 etc.
    x = max(6, int(round(10 * s)))
    y = max(18, int(round(36 * s)))
    step = max(16, int(round(36 * s)))
    scale = 0.7 * s
    thickness = max(1, int(round(2 * s)))
    padding = max(3, int(round(6 * s)))
    for txt, color in overlay_lines:
        draw_text_with_bg(
            frame, txt, x, y, color, scale=scale, thickness=thickness, padding=padding
        )
        y += step
