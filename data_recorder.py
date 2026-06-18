import cv2
import os
import time
import numpy as np
import mediapipe as mp

# =========================
# CONFIGURATION
# =========================
GESTURE_NAME = "Idle"  # Change this to "Pull_Lever", "Idle", etc.
NUM_FRAMES = 30            # Frames per sequence (1 second of motion)

DATASET_DIR = os.path.join("Dataset", GESTURE_NAME)
os.makedirs(DATASET_DIR, exist_ok=True)

def main():
    # =========================
    # MEDIAPIPE SETUP
    # =========================
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1, # Only need 1 hand for training data usually
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # State variables
    is_recording = False
    frames_recorded = 0
    current_sequence = []
    
    # --- AUTOMATIC FILE COUNTER ---
    # Counts how many .npy files already exist in the folder
    total_clips = len([name for name in os.listdir(DATASET_DIR) if name.endswith('.npy')])

    cap = cv2.VideoCapture(0)

    print(f"--- DATA RECORDER LOADED ---")
    print(f"Target Gesture: {GESTURE_NAME}")
    print(f"Saving to: {DATASET_DIR}")
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
                # Just grab the first hand on screen for training data
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- RECORDING LOGIC ---
                if is_recording:
                    # Flatten the 21 landmarks into a single array of 63 values
                    frame_data = []
                    for lm in hand_landmarks.landmark:
                        frame_data.extend([lm.x, lm.y, lm.z])
                    
                    current_sequence.append(frame_data)
                    frames_recorded += 1

                    if frames_recorded == NUM_FRAMES:
                        is_recording = False
                        filename = os.path.join(DATASET_DIR, f"{int(time.time() * 1000)}.npy")
                        np.save(filename, np.array(current_sequence))
                        
                        # Increment total and print
                        total_clips += 1
                        print(f"Saved clip: {filename} (Total: {total_clips})")
                        current_sequence = []

            # --- UI FEEDBACK ---
            cv2.putText(frame, f"MODE: RECORDING [{GESTURE_NAME}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show the live total count on the screen
            cv2.putText(frame, f"Total Saved: {total_clips} / 300", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            if is_recording:
                cv2.putText(frame, f"RECORDING... {frames_recorded}/{NUM_FRAMES}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Press 'R' to Start", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Developer Data Recorder", frame)

            # --- KEYBOARD CONTROLS ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r") and not is_recording:
                # Ensure a hand is visible before starting to record
                if results.multi_hand_landmarks:
                    is_recording = True
                    frames_recorded = 0
                    current_sequence = []
                else:
                    print("No hand detected! Put your hand in frame before pressing R.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()