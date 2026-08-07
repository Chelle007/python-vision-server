import cv2
import mediapipe as mp

from vision_server.camera import LatestFrameCamera
from vision_server.config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MAX_NUM_FACES,
    MAX_NUM_HANDS,
    UDP_IP,
    UDP_PORT,
)
from vision_server.gestures.dynamic import GestureLSTM
from vision_server.gestures.hand import HAND_RULES
from vision_server.gestures.hand.cursor_fields import (
    apply_right_hand_cursor_fields,
    reset_last_point,
)
from vision_server.gestures.hand.geometry import get_hand_rotation
from vision_server.gestures.hand.watch_tap import apply_watch_tap_fields
from vision_server.gestures.head import HEAD_RULES
from vision_server.gestures.head.pitch_cal import PitchCalibrator
from vision_server.overlay import (
    build_overlay_lines,
    draw_lock_ring,
    draw_overlay,
    draw_pitch_indicator,
)
from vision_server.tracking import (
    PlayerLock,
    collect_faces,
    collect_hands,
    create_face_mesh,
    create_hands,
)
from vision_server.udp import create_udp_socket, default_payload, send_payload


def evaluate_hand_rules(landmarks) -> dict[str, bool]:
    return {name: fn(landmarks) for name, fn in HAND_RULES}


def apply_head_rules(face_landmarks, data: dict) -> None:
    for _, fn in HEAD_RULES:
        result = fn(face_landmarks)
        if isinstance(result, dict):
            data.update(result)


def _landmark_dicts(landmarks) -> list[dict]:
    return [
        {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
        for lm in landmarks
    ]


def _append_hand_packet(data: dict, hand) -> None:
    world = []
    if hand.world_landmarks is not None:
        world = _landmark_dicts(hand.world_landmarks.landmark)
    data["hands"].append(
        {
            "handedness": hand.handedness,
            "landmarks": _landmark_dicts(hand.landmarks),
            "world_landmarks": world,
        }
    )


def main():
    sock = create_udp_socket()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = create_hands(max_num_hands=MAX_NUM_HANDS)
    face_mesh = create_face_mesh(max_num_faces=MAX_NUM_FACES)
    lstm = GestureLSTM()
    player_lock = PlayerLock()
    pitch_cal = PitchCalibrator()

    # Threaded reader drops stale buffered frames while MediaPipe runs.
    cap = LatestFrameCamera()
    cam_w, cam_h, cam_fps = cap.negotiated_size()

    print(f"Combined Vision Server Running. Sending UDP to {UDP_IP}:{UDP_PORT}")
    print(
        f"Camera negotiated: {cam_w}x{cam_h} @ {cam_fps:.0f} fps "
        f"(requested {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS})"
    )
    print("Press Q to quit.  Press C to recalibrate look pitch neutral.")

    last_point = [-1.0, -1.0]

    try:
        while cap.isOpened():
            success, frame = cap.read()

            if not success or frame is None:
                print("Failed to read webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            lock = player_lock.update(
                collect_faces(face_results),
                collect_hands(hand_results),
            )

            data = default_payload()
            data["player_locked"] = lock.locked
            data["lock_id"] = lock.lock_id
            data["lock_status"] = lock.status

            if lock.flush_lstm:
                lstm.flush()
                reset_last_point(last_point)

            right_hand_seen = False
            lstm_display = "Idle"
            left_landmarks = None
            right_landmarks = None

            if lock.left is not None:
                left = lock.left
                left_landmarks = left.landmarks
                _append_hand_packet(data, left)
                mp_draw.draw_landmarks(
                    frame, left.mp_landmarks, mp_hands.HAND_CONNECTIONS
                )
                gestures = evaluate_hand_rules(left_landmarks)
                data["leftFist"] = gestures["fist"]
                data["leftOpenPalm"] = gestures["open_palm"]
                data["leftIndexUp"] = gestures["index_up"]
                data["leftPeace"] = gestures["peace"]

            if lock.right is not None:
                right = lock.right
                right_landmarks = right.landmarks
                right_hand_seen = True
                _append_hand_packet(data, right)
                mp_draw.draw_landmarks(
                    frame, right.mp_landmarks, mp_hands.HAND_CONNECTIONS
                )
                gestures = evaluate_hand_rules(right_landmarks)
                lstm.register_hand_seen()
                data["rightFist"] = gestures["fist"]
                data["rightOpenPalm"] = gestures["open_palm"]
                data["rightIndexUp"] = gestures["index_up"]
                data["rightPeace"] = gestures["peace"]

                apply_right_hand_cursor_fields(
                    data, right_landmarks, gestures, last_point
                )

                fist_rot_x, fist_rot_y, fist_rot_z = get_hand_rotation(
                    right_landmarks
                )
                data["fistRotX"] = round(fist_rot_x, 3)
                data["fistRotY"] = round(fist_rot_y, 3)
                data["fistRotZ"] = round(fist_rot_z, 3)

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

            if lock.face is not None:
                apply_head_rules(lock.face, data)
                pitch_cal.apply_to_payload(data, data.get("head_pitch", 0.0))
            else:
                pitch_cal.apply_to_payload(data, None)

            overlay_lines = build_overlay_lines(data, lstm_display)
            draw_overlay(frame, overlay_lines)
            draw_pitch_indicator(frame, data.get("head_pitch", 0.0))
            draw_lock_ring(
                frame,
                status=lock.status,
                center=lock.ring_center,
                ring_size=lock.ring_size,
                progress=lock.progress,
            )
            send_payload(sock, data)

            cv2.imshow("Combined Hand + Face Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                pitch_cal.request_recalibrate()
                print("Pitch recalibration requested — hold still, look at the screen.")
    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()


if __name__ == "__main__":
    main()
