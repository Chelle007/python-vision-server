import cv2
import json
import socket

import mediapipe as mp

from gesture_heuristics import (
    get_hand_rotation,
    get_palm_position,
    is_fist,
    is_index_up,
    is_open_palm,
    is_peace_sign,
)
from lstm_classifier import GestureLSTM

# =========================
# UDP SETUP
# =========================
UDP_IP = "127.0.0.1"
UDP_PORT = 5052


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


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # =========================
    # MEDIAPIPE SETUP
    # =========================
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    lstm = GestureLSTM()

    # =========================
    # CAMERA START
    # =========================
    cap = cv2.VideoCapture(0)

    print(f"Combined Vision Server Running. Sending UDP to {UDP_IP}:{UDP_PORT}")
    print("Press Q in the webcam window to quit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                print("Failed to read webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            # Default data sent to Unity every frame (same keys as original server)
            data = {
                "hand_x": 0.5,
                "hand_y": 0.5,
                "hand_up": False,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "leftFist": False,
                "leftOpenPalm": False,
                "leftIndexUp": False,
                "leftPeace": False,
                "rightFist": False,
                "rightOpenPalm": False,
                "landmarks": [],
                "pinching": False,
                "openPalm": False,
                "isFist": False,
                "fistRotX": 0.0,
                "fistRotY": 0.0,
                "fistRotZ": 0.0,
                "palmX": -1.0,
                "palmY": -1.0,
                "hands": [],
                # Added for LSTM puzzle gestures (Unity can ignore until wired up)
                "lstm_gesture": "Idle",
            }

            right_hand_seen = False
            lstm_display = "Idle"

            # =========================
            # HAND TRACKING
            # =========================
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                ):
                    landmarks = hand_landmarks.landmark

                    hand_label = handedness.classification[0].label
                    side = hand_label.lower()

                    landmark_list = [
                        {
                            "x": round(lm.x, 4),
                            "y": round(lm.y, 4),
                            "z": round(lm.z, 4),
                        }
                        for lm in landmarks
                    ]

                    world_landmark_list = []

                    if hand_results.multi_hand_world_landmarks:
                        hand_index = len(data["hands"])

                        if hand_index < len(hand_results.multi_hand_world_landmarks):
                            world_landmark_list = [
                                {
                                    "x": round(lm.x, 4),
                                    "y": round(lm.y, 4),
                                    "z": round(lm.z, 4),
                                }
                                for lm in hand_results.multi_hand_world_landmarks[
                                    hand_index
                                ].landmark
                            ]

                    data["hands"].append(
                        {
                            "handedness": hand_label,
                            "landmarks": landmark_list,
                            "world_landmarks": world_landmark_list,
                        }
                    )

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    fist = is_fist(landmarks)
                    open_palm = is_open_palm(landmarks)
                    index_up = is_index_up(landmarks)
                    peace = is_peace_sign(landmarks)

                    # LEFT HAND = MOVEMENT
                    if side == "left":
                        data["leftFist"] = fist
                        data["leftOpenPalm"] = open_palm
                        data["leftIndexUp"] = index_up
                        data["leftPeace"] = peace

                    # RIGHT HAND = INTERACTION
                    else:
                        right_hand_seen = True
                        lstm.register_hand_seen()
                        data["rightFist"] = fist
                        data["rightOpenPalm"] = open_palm

                        palm_x, palm_y = get_palm_position(landmarks)

                        data["palmX"] = round(palm_x, 3)
                        data["palmY"] = round(palm_y, 3)

                        fist_rot_x, fist_rot_y, fist_rot_z = get_hand_rotation(landmarks)

                        data["fistRotX"] = round(fist_rot_x, 3)
                        data["fistRotY"] = round(fist_rot_y, 3)
                        data["fistRotZ"] = round(fist_rot_z, 3)

                        lstm_display = lstm.predict(landmarks)
                        data["lstm_gesture"] = (
                            lstm_display
                            if lstm_display in lstm.classes
                            else "Idle"
                        )

            if not right_hand_seen:
                lstm.register_hand_lost()
                lstm_display = lstm.get_overlay_label()
                data["lstm_gesture"] = (
                    lstm_display if lstm_display in lstm.classes else "Idle"
                )

            # =========================
            # FACE TRACKING
            # =========================
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

                nose = face_landmarks.landmark[1]
                left_side = face_landmarks.landmark[234]
                right_side = face_landmarks.landmark[454]
                top = face_landmarks.landmark[10]
                bottom = face_landmarks.landmark[152]

                face_width = right_side.x - left_side.x
                nose_offset_x = nose.x - left_side.x

                if face_width > 0:
                    data["head_yaw"] = round(
                        ((nose_offset_x / face_width) - 0.5) * 2, 3
                    )

                face_height = bottom.y - top.y
                nose_offset_y = nose.y - top.y

                if face_height > 0:
                    data["head_pitch"] = round(
                        ((nose_offset_y / face_height) - 0.5) * 2, 3
                    )

            # =========================
            # SEND TO UNITY
            # =========================
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

            y = 40
            for txt, color in overlay_lines:
                draw_text_with_bg(frame, txt, 10, y, color)
                y += 40

            message = json.dumps(data).encode("utf-8")
            sock.sendto(message, (UDP_IP, UDP_PORT))

            cv2.imshow("Combined Hand + Face Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()


if __name__ == "__main__":
    main()
