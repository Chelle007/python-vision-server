import time

import cv2
import mediapipe as mp

from vision_server.camera import LatestFrameCamera
from vision_server.config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    FACE_MESH_EVERY_N,
    FRAME_SLOW_MS,
    MAX_NUM_FACES,
    MAX_NUM_HANDS,
    MOVE_GESTURE_OVERRIDES,
    SHOW_PREVIEW,
    UDP_IP,
    UDP_PORT,
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
from vision_server.gestures.head import HEAD_RULES
from vision_server.gestures.head.pitch_cal import PitchCalibrator
from vision_server.hand_roles import HandRoles
from vision_server.overlay import (
    build_overlay_lines,
    draw_lock_ring,
    draw_overlay,
    draw_pitch_indicator,
)
from vision_server.perfstats import FrameStats
from vision_server.puzzle_gate import PuzzleGate
from vision_server.runtime import apply_opencv_threads
from vision_server.tracking import (
    PlayerLock,
    collect_faces,
    collect_hands,
    create_face_mesh,
    create_hands,
)
from vision_server.udp import create_udp_socket, default_payload, send_payload


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
    apply_opencv_threads()
    sock = create_udp_socket()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = create_hands(max_num_hands=MAX_NUM_HANDS)
    face_mesh = create_face_mesh(max_num_faces=MAX_NUM_FACES)
    lstm = GestureLSTM()
    player_lock = PlayerLock()
    pitch_cal = PitchCalibrator()
    # Keyboard-driven for now; Unity becomes a second caller of set_active().
    puzzle_gate = PuzzleGate()
    # Same deal: H swaps the hands today, Unity's settings screen later.
    hand_roles = HandRoles()
    # One per role, never shared: the counters are per-hand state, and a single
    # instance would let the MOVE hand's streak commit the ACTION hand's label.
    # They also carry different thresholds — walking wants to start and stop
    # immediately, sitting does not (see MOVE_GESTURE_OVERRIDES in config).
    move_debounce = GestureDebouncer(MOVE_GESTURE_OVERRIDES)
    action_debounce = GestureDebouncer()
    frame_stats = FrameStats()

    # Threaded reader drops stale buffered frames while MediaPipe runs.
    cap = LatestFrameCamera()
    cam_w, cam_h, cam_fps = cap.negotiated_size()

    print(f"Combined Vision Server Running. Sending UDP to {UDP_IP}:{UDP_PORT}")
    print(
        f"Camera negotiated: {cam_w}x{cam_h} @ {cam_fps:.0f} fps "
        f"(requested {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS})"
    )
    if SHOW_PREVIEW:
        print("Press Q to quit.  Press C to recalibrate look pitch neutral.")
        print("Press P to toggle puzzle mode (LSTM inference) — starts OFF.")
        print(
            "Press H to swap hand roles — action (grab/cursor/LSTM) hand starts "
            f"{hand_roles.action_hand.upper()}."
        )
    else:
        print("Headless (SHOW_PREVIEW=False): no preview window, no keys.")
        print("Press Ctrl-C to quit. Unity drives the puzzle gate.")
        print(
            "Hand roles fixed at ACTION = "
            f"{hand_roles.action_hand.upper()}, "
            f"MOVE = {hand_roles.move_hand.upper()}."
        )

    last_point = [-1.0, -1.0]
    frame_index = 0
    last_faces = []

    try:
        while cap.isOpened():
            success, frame = cap.read()

            if not success or frame is None:
                # read() only fails after release/stop; keep going if still open.
                if not cap.isOpened():
                    d = cap.diagnostics()
                    print(
                        "Webcam reader stopped. "
                        f"last_event={d['exit_reason']} "
                        f"device_open={d['device_open']} "
                        f"reopen_count={d['reopen_count']} "
                        f"frame_id={d['frame_id']} "
                        f"uptime_s={d['uptime_s']}"
                    )
                    break
                print("[cam] waiting for next frame…")
                continue

            t_start = time.perf_counter()
            lstm_ms = 0.0

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_prep = time.perf_counter()

            hand_results = hands.process(rgb)
            t_hands = time.perf_counter()
            # Hand cost scales with hands actually detected, not with
            # MAX_NUM_HANDS: measured ~18ms at zero hands and ~60ms at two,
            # with no difference between max_num_hands 2 and 4. Tracked so a
            # slow `hands` stage can be told apart from a busy frame.
            hands_n = len(hand_results.multi_hand_landmarks or ())

            # Face mesh is a whole MediaPipe graph, but head pitch/tilt move far
            # slower than hands. On skipped frames PlayerLock is fed the previous
            # face so the seat is not seen as empty every other frame — its
            # timeouts run on wall clock, so a one-frame-stale face is harmless.
            if frame_index % FACE_MESH_EVERY_N == 0:
                faces = collect_faces(face_mesh.process(rgb))
                last_faces = faces
            else:
                faces = last_faces
            frame_index += 1
            t_face = time.perf_counter()

            lock = player_lock.update(faces, collect_hands(hand_results))

            data = default_payload()
            data["player_locked"] = lock.locked
            data["lock_id"] = lock.lock_id
            data["lock_status"] = lock.status
            puzzle_gate.apply_to_payload(data)
            hand_roles.apply_to_payload(data)

            if lock.flush_lstm:
                lstm.flush()
                reset_last_point(last_point)
                # The hand behind the gesture counters just changed identity —
                # a held crouch must not survive into the new player's session.
                move_debounce.reset()
                action_debounce.reset()

            lstm_display = "Idle"
            # The payload keys stay left*/right*, but they carry ROLES, not
            # physical sides: left* = MOVE hand, right* = ACTION hand. Unity
            # therefore needs no change when the player swaps hands.
            move, action = hand_roles.split(lock)
            mirror = hand_roles.mirror_action_hand
            action_hand_seen = action is not None
            move_landmarks = move.landmarks if move is not None else None
            action_landmarks = action.landmarks if action is not None else None

            if move is not None:
                _append_hand_packet(data, move)
                if SHOW_PREVIEW:
                    mp_draw.draw_landmarks(
                        frame, move.mp_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            if action is not None:
                _append_hand_packet(data, action)
                if SHOW_PREVIEW:
                    mp_draw.draw_landmarks(
                        frame, action.mp_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # Classify each hand once, then let the commit delay decide what
            # Unity actually sees. A missing hand is fed NONE rather than
            # skipped, so a MediaPipe dropout has to outlast the off-count
            # before it can end a held gesture — losing the hand for a frame
            # used to stand a crouching player straight up.
            move_label = move_debounce.update(
                classify_hand(move_landmarks) if move_landmarks is not None else NONE
            )
            action_label = action_debounce.update(
                classify_hand(action_landmarks)
                if action_landmarks is not None
                else NONE
            )

            move_gestures = rules_from_label(move_label)
            data["leftFist"] = move_gestures["fist"]
            data["leftOpenPalm"] = move_gestures["open_palm"]
            data["leftIndexUp"] = move_gestures["index_up"]
            data["leftPeace"] = move_gestures["peace"]
            data["moveGesture"] = move_label
            data["moveGestureRaw"] = move_debounce.raw

            action_gestures = rules_from_label(action_label)
            data["rightFist"] = action_gestures["fist"]
            data["rightOpenPalm"] = action_gestures["open_palm"]
            data["rightIndexUp"] = action_gestures["index_up"]
            data["rightPeace"] = action_gestures["peace"]
            data["actionGesture"] = action_label
            data["actionGestureRaw"] = action_debounce.raw

            if action is not None:
                lstm.register_hand_seen()

                # Debounced gestures, so a one-frame misread cannot warp the
                # cursor by flipping the fist/index branch.
                apply_cursor_fields(data, action_landmarks, action_gestures, last_point)

                # Keep the pitch/yaw/roll names across the call so the X/Y/Z
                # mapping is readable here instead of only in geometry.py.
                pitch, yaw, roll = get_hand_rotation(
                    action_landmarks, mirror=mirror
                )
                data["fistRotX"] = round(pitch, 3)
                data["fistRotY"] = round(yaw, 3)
                data["fistRotZ"] = round(roll, 3)

                # Buffer keeps filling either way; only inference is gated.
                t_lstm = time.perf_counter()
                lstm_display = lstm.predict(
                    action_landmarks, infer=puzzle_gate.active, mirror=mirror
                )
                lstm_ms = (time.perf_counter() - t_lstm) * 1000.0
                data["lstm_gesture"] = lstm.clamp_label(lstm_display)

            if apply_watch_tap_fields(data, move_landmarks, action_landmarks):
                reset_last_point(last_point)
                lstm_display = "Idle"
                # A tap suppresses solo gestures; clearing the counters too
                # means the player has to re-establish one afterwards rather
                # than resuming whatever was committed before the tap.
                move_debounce.reset()
                action_debounce.reset()

            if not action_hand_seen:
                reset_last_point(last_point)
                lstm.register_hand_lost()
                lstm_display = lstm.get_overlay_label()
                data["lstm_gesture"] = lstm.clamp_label(lstm_display)

            if lock.face is not None:
                apply_head_rules(lock.face, data)
                pitch_cal.apply_to_payload(data, data.get("head_pitch", 0.0))
            else:
                pitch_cal.apply_to_payload(data, None)

            # Send before drawing: Unity should not wait on the debug preview.
            send_payload(sock, data)
            t_send = time.perf_counter()

            # Everything below is the debug preview — drawing, the macOS HighGUI
            # event loop, and the keyboard. None of it is needed by Unity.
            key = 0xFF
            if SHOW_PREVIEW:
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
                cv2.imshow("Combined Hand + Face Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
            t_show = time.perf_counter()

            # Perf guard: a stalled frame is a compute problem, not a camera
            # one — the drain thread grabs independently of this loop. "other"
            # covers overlay drawing, UDP send, imshow and waitKey.
            total_ms = (t_show - t_start) * 1000.0
            prep_ms = (t_prep - t_start) * 1000.0
            hands_ms = (t_hands - t_prep) * 1000.0
            face_ms = (t_face - t_hands) * 1000.0
            # Everything between the face mesh and the display: PlayerLock,
            # gesture rules, overlay drawing and the UDP send. Measured at
            # ~1ms combined, so it is billed as one stage rather than four.
            # The LSTM runs inside this region, so bill it separately.
            logic_ms = (t_send - t_face) * 1000.0 - lstm_ms
            show_ms = (t_show - t_send) * 1000.0
            frame_stats.record(
                total_ms=total_ms,
                prep_ms=prep_ms,
                hands_ms=hands_ms,
                face_ms=face_ms,
                lstm_ms=lstm_ms,
                logic_ms=logic_ms,
                show_ms=show_ms,
                hands_n=hands_n,
                slow=total_ms >= FRAME_SLOW_MS,
            )
            frame_stats.maybe_report()

            if key == ord("q"):
                break
            if key == ord("c"):
                pitch_cal.request_recalibrate()
                print("Pitch recalibration requested — hold still, look at the screen.")
            if key == ord("p"):
                puzzle_gate.toggle(source="keyboard")
                print(
                    "Puzzle mode "
                    f"{'ON — LSTM predicting' if puzzle_gate.active else 'OFF — LSTM idle'}"
                )
            if key == ord("h"):
                hand_roles.swap(source="keyboard")
                # The buffered sequence belongs to the other hand and is now
                # mirrored differently — keeping it would blend two chiralities.
                lstm.flush()
                reset_last_point(last_point)
                # Each debouncer is now counting a different physical hand.
                move_debounce.reset()
                action_debounce.reset()
                print(
                    f"Hand roles swapped — ACTION (grab/cursor/LSTM) = "
                    f"{hand_roles.action_hand.upper()}, "
                    f"MOVE = {hand_roles.move_hand.upper()}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down.")
    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()


if __name__ == "__main__":
    main()
