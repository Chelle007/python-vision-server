import os
import time

import cv2
import mediapipe as mp
import numpy as np

from vision_server.config import DATA_DIR, NUM_FRAMES
from vision_server.features import flatten_landmarks
from vision_server.tracking.hands import create_hands

# Change this to "Pull_Lever", "Idle", etc.
GESTURE_NAME = "Idle"


def main():
    dataset_dir = os.path.join(DATA_DIR, GESTURE_NAME)
    os.makedirs(dataset_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = create_hands(max_num_hands=1)

    is_recording = False
    frames_recorded = 0
    current_sequence = []

    total_clips = len(
        [name for name in os.listdir(dataset_dir) if name.endswith(".npy")]
    )

    cap = cv2.VideoCapture(0)

    print("--- DATA RECORDER LOADED ---")
    print(f"Target Gesture: {GESTURE_NAME}")
    print(f"Saving to: {dataset_dir}")
    print(f"Currently have {total_clips} clips saved.")
    print("Press 'R' to record a clip. Press 'Q' to quit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                if is_recording:
                    current_sequence.append(
                        flatten_landmarks(hand_landmarks.landmark)
                    )
                    frames_recorded += 1

                    if frames_recorded == NUM_FRAMES:
                        is_recording = False
                        filename = os.path.join(
                            dataset_dir, f"{int(time.time() * 1000)}.npy"
                        )
                        np.save(filename, np.array(current_sequence))

                        total_clips += 1
                        print(f"Saved clip: {filename} (Total: {total_clips})")
                        current_sequence = []

            cv2.putText(
                frame,
                f"MODE: RECORDING [{GESTURE_NAME}]",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Total Saved: {total_clips} / 300",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            if is_recording:
                cv2.putText(
                    frame,
                    f"RECORDING... {frames_recorded}/{NUM_FRAMES}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
            else:
                cv2.putText(
                    frame,
                    "Press 'R' to Start",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Developer Data Recorder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and not is_recording:
                if results.multi_hand_landmarks:
                    is_recording = True
                    frames_recorded = 0
                    current_sequence = []
                else:
                    print(
                        "No hand detected! Put your hand in frame before pressing R."
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
