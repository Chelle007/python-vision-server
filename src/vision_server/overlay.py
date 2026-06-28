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


def build_overlay_lines(data: dict, lstm_display: str) -> list[tuple[str, tuple[int, int, int]]]:
    overlay_lines = [
        (f"AI LSTM: {lstm_display}", lstm_display_color(lstm_display)),
    ]

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

    if data["tilt_left"]:
        overlay_lines.append(("HEAD TILT LEFT = TURN BACK", (0, 255, 0)))

    if data["tilt_right"]:
        overlay_lines.append(("HEAD TILT RIGHT = TURN BACK", (0, 255, 0)))

    return overlay_lines


def draw_overlay(frame, overlay_lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 40
    for txt, color in overlay_lines:
        draw_text_with_bg(frame, txt, 10, y, color)
        y += 40
