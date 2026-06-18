import cv2
import mediapipe as mp

from vision_server.config import UDP_IP, UDP_PORT
from vision_server.gestures.dynamic import GestureLSTM
from vision_server.gestures.hand import HAND_RULES
from vision_server.gestures.hand.geometry import get_hand_rotation, get_palm_position
from vision_server.gestures.head import HEAD_RULES
from vision_server.overlay import build_overlay_lines, draw_overlay
from vision_server.tracking import create_face_mesh, create_hands
from vision_server.udp import create_udp_socket, default_payload, send_payload


def evaluate_hand_rules(landmarks) -> dict[str, bool]:
    return {name: fn(landmarks) for name, fn in HAND_RULES}


def apply_head_rules(face_landmarks, data: dict) -> None:
    for _, fn in HEAD_RULES:
        result = fn(face_landmarks)
        if isinstance(result, dict):
            data.update(result)


def main():
    sock = create_udp_socket()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = create_hands(max_num_hands=2)
    face_mesh = create_face_mesh(max_num_faces=1)
    lstm = GestureLSTM()

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

            data = default_payload()
            right_hand_seen = False
            lstm_display = "Idle"

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

                    gestures = evaluate_hand_rules(landmarks)

                    if side == "left":
                        data["leftFist"] = gestures["fist"]
                        data["leftOpenPalm"] = gestures["open_palm"]
                        data["leftIndexUp"] = gestures["index_up"]
                        data["leftPeace"] = gestures["peace"]
                    else:
                        right_hand_seen = True
                        lstm.register_hand_seen()
                        data["rightFist"] = gestures["fist"]
                        data["rightOpenPalm"] = gestures["open_palm"]
                        data["rightIndexUp"] = gestures["index_up"]

                        palm_x, palm_y = get_palm_position(landmarks)

                        data["palmX"] = round(palm_x, 3)
                        data["palmY"] = round(palm_y, 3)

                        fist_rot_x, fist_rot_y, fist_rot_z = get_hand_rotation(landmarks)

                        data["fistRotX"] = round(fist_rot_x, 3)
                        data["fistRotY"] = round(fist_rot_y, 3)
                        data["fistRotZ"] = round(fist_rot_z, 3)

                        lstm_display = lstm.predict(landmarks)
                        data["lstm_gesture"] = (
                            lstm_display if lstm_display in lstm.classes else "Idle"
                        )

            if not right_hand_seen:
                lstm.register_hand_lost()
                lstm_display = lstm.get_overlay_label()
                data["lstm_gesture"] = (
                    lstm_display if lstm_display in lstm.classes else "Idle"
                )

            if face_results.multi_face_landmarks:
                apply_head_rules(face_results.multi_face_landmarks[0], data)

            overlay_lines = build_overlay_lines(data, lstm_display)
            draw_overlay(frame, overlay_lines)
            send_payload(sock, data)

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
