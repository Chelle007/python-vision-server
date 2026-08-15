"""Live readout of every term the OK sign is judged on.

The gesture has to clear five separate checks, and the server only ever tells
you the verdict — so a gesture that will not fire gives you nothing to act on.
This prints each term against its threshold as you hold the pose, which turns
"it is not working" into "the ring finger is sitting in the dead zone".

Nothing here talks to Unity and nothing writes to config; it is a throwaway
diagnostic. Delete it once the pose is confirmed.

    python scripts/check_ok_sign.py

Q or Esc quits.
"""

from __future__ import annotations

import cv2
import mediapipe as mp

from vision_server.config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    FINGER_CURL_MARGIN,
    FINGER_EXTEND_MARGIN,
    OK_SIGN_MIN_PALM,
    OK_SIGN_PINCH,
)
from vision_server.gestures.hand.classify import classify_hand
from vision_server.gestures.hand.fingers import (
    finger_margins,
    hand_frame,
    pinch_distance,
)
from vision_server.tracking import create_hands

PASS = (120, 255, 120)
FAIL = (110, 110, 255)
PLAIN = (220, 220, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# What the OK sign needs from each finger: the index curls to meet the thumb,
# the other three stay out of the way.
WANT_EXTENDED = {"index": False, "middle": True, "ring": True, "pinky": True}


def _finger_row(name: str, margin: float) -> tuple[str, bool]:
    """One finger's margin, the state it implies, and whether that is wanted."""
    if margin >= FINGER_EXTEND_MARGIN:
        state = "extended"
    elif margin <= -FINGER_CURL_MARGIN:
        state = "curled"
    else:
        # The dead zone is the most common reason a real pose reports nothing,
        # and it is invisible from the outside — worth naming explicitly.
        state = "AMBIGUOUS"

    wanted = "extended" if WANT_EXTENDED[name] else "curled"
    return f"  {name:7s} {margin:+.2f}  {state}", state == wanted


def _report(landmarks) -> list[tuple[str, bool | None]]:
    """Every term the classifier applies, in the order it applies them."""
    lines: list[tuple[str, bool | None]] = []

    frame = hand_frame(landmarks)
    if frame is None:
        lines.append(("  hand points at the camera - no axis", False))
        return lines

    _, palm = frame
    lines.append((f"  palm    {palm:.3f}  >= {OK_SIGN_MIN_PALM:.2f}", palm >= OK_SIGN_MIN_PALM))

    margins = finger_margins(landmarks, frame)
    for name in WANT_EXTENDED:
        lines.append(_finger_row(name, margins[name]))

    pinch = pinch_distance(landmarks, frame)
    lines.append((f"  pinch   {pinch:.2f}  <= {OK_SIGN_PINCH:.2f}", pinch <= OK_SIGN_PINCH))

    return lines


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Close anything else using it.")

    hands = create_hands()
    draw = mp.solutions.drawing_utils

    while True:
        ok, image = cap.read()
        if not ok:
            break

        # Mirrored exactly as the server does it, so the hand you move to your
        # right moves right on screen and the readout matches the live preview.
        image = cv2.flip(image, 1)
        result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        y = 30
        cv2.putText(image, "OK SIGN CHECK - hold the pose", (12, y), FONT, 0.6, PLAIN, 2)
        y += 30

        if not result.multi_hand_landmarks:
            cv2.putText(image, "no hand visible", (12, y), FONT, 0.5, FAIL, 1)
        else:
            for index, hand in enumerate(result.multi_hand_landmarks):
                draw.draw_landmarks(image, hand, mp.solutions.hands.HAND_CONNECTIONS)

                label = classify_hand(hand.landmark)
                fired = label == "ok_sign"
                cv2.putText(
                    image,
                    f"hand {index + 1}: {label}",
                    (12, y),
                    FONT,
                    0.55,
                    PASS if fired else PLAIN,
                    2,
                )
                y += 22

                for text, good in _report(hand.landmark):
                    cv2.putText(image, text, (12, y), FONT, 0.45, PASS if good else FAIL, 1)
                    y += 19
                y += 8

        cv2.imshow("OK sign check", image)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
