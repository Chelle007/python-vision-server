import cv2


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
    if gesture_label in ("Turn_Key", "Pull_Lever"):
        return (0, 255, 0)
    if gesture_label.startswith("Stabilizing"):
        return (0, 255, 255)
    if gesture_label == "No Model":
        return (0, 0, 255)
    return (255, 255, 255)


def _lock_status_label(status: str, lock_id: int) -> tuple[str, tuple[int, int, int]]:
    if status == "locked":
        return f"PLAYER LOCK #{lock_id}", (0, 255, 0)
    if status == "challenger":
        return "SWITCHING PLAYER…", (0, 165, 255)
    return "NO PLAYER LOCK", (160, 160, 160)


def build_overlay_lines(data: dict, lstm_display: str) -> list[tuple[str, tuple[int, int, int]]]:
    overlay_lines = [
        (f"AI LSTM: {lstm_display}", lstm_display_color(lstm_display)),
    ]

    status = data.get("lock_status", "unlocked")
    overlay_lines.append(_lock_status_label(status, data.get("lock_id", 0)))

    if data["leftFist"]:
        overlay_lines.append(("LEFT FIST = MOVE", (0, 255, 0)))

    if data["leftIndexUp"]:
        overlay_lines.append(("LEFT INDEX = JUMP", (0, 255, 0)))

    if data["leftPeace"]:
        overlay_lines.append(("LEFT PEACE = CROUCH", (0, 255, 0)))

    if data["rightFist"]:
        overlay_lines.append(("RIGHT FIST = GRAB", (0, 255, 0)))

    if data["rightOpenPalm"]:
        overlay_lines.append(("RIGHT OPEN PALM = RELEASE", (0, 255, 0)))

    if data["rightPeace"]:
        overlay_lines.append(("RIGHT PEACE = STAND", (0, 255, 0)))

    if data["watchTap"]:
        overlay_lines.append(("WATCH TAP", (0, 255, 0)))

    if data["watchTapDistance"] is not None:
        overlay_lines.append(
            (f"Watch dist: {data['watchTapDistance']:.2f}", (200, 200, 200))
        )

    if data["tilt_left"]:
        overlay_lines.append(("HEAD TILT LEFT = TURN BACK", (0, 255, 0)))

    if data["tilt_right"]:
        overlay_lines.append(("HEAD TILT RIGHT = TURN BACK", (0, 255, 0)))

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
    thickness = 2

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
        cv2.line(frame, (x0, y0 - 6), (fill_x1, y0 - 6), color, 3, cv2.LINE_AA)



def draw_overlay(frame, overlay_lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 40
    for txt, color in overlay_lines:
        draw_text_with_bg(frame, txt, 10, y, color)
        y += 40
