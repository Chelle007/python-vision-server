import time

import cv2
import mediapipe as mp

from vision_server.camera import LatestFrameCamera
from vision_server.config import (
    ACTION_GESTURE_LATCH,
    ACTION_LATCH_HAND_LOST_FRAMES,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    FACE_MESH_EVERY_N,
    FRAME_SLOW_MS,
    MAX_NUM_FACES,
    MAX_NUM_HANDS,
    MEDIAPIPE_HAND_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MOVE_GESTURE_OVERRIDES,
    TCP_REPORT_IP,
    TCP_REPORT_PORT,
    UDP_CONTROL_IP,
    UDP_CONTROL_PORT,
    UDP_IP,
    UDP_PORT,
)
from vision_server.cli import parse_args, prepare_frozen_cwd, resolve_show_preview
from vision_server.control import (
    apply_control_messages,
    create_control_socket,
    drain_control_socket,
)
from vision_server.gesture_report import GestureDiagnostics, send_report_tcp
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
from vision_server.gestures.hand.fingers import (
    hand_frame,
    thumb_clearance,
    thumb_reach,
)
from vision_server.gestures.hand.geometry import get_hand_rotation
from vision_server.gestures.hand.watch_tap import (
    WatchTapDebouncer,
    apply_watch_tap_fields,
)
from vision_server.gestures.head import HEAD_RULES
from vision_server.gestures.head.pitch_cal import PitchCalibrator
from vision_server.hand_roles import HandRoles
from vision_server.overlay import (
    build_overlay_lines,
    draw_hand_skeleton,
    draw_lock_ring,
    draw_overlay,
    draw_pitch_indicator,
)
from vision_server.perfstats import FrameStats
from vision_server.preview import PreviewStream
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


def _thumb_metrics(landmarks) -> tuple[float | None, float | None]:
    """Diagnostic ``(clearance, reach)`` for one hand, in palm lengths.

    These two are the whole fist/thumbs-up decision, so putting them in the
    payload turns "the gesture will not fire" into a pair of readings that can
    be compared against THUMB_UP_CLEARANCE / THUMB_UP_REACH on the spot.
    """
    if landmarks is None:
        return None, None

    frame = hand_frame(landmarks)
    if frame is None:
        return None, None

    clearance = thumb_clearance(landmarks, frame)
    reach = thumb_reach(landmarks, frame)
    return (
        None if clearance is None else round(clearance, 2),
        None if reach is None else round(reach, 2),
    )


def reset_for_hand_role_change(
    *,
    lstm,
    last_point,
    move_debounce,
    action_debounce,
    watch_tap_debounce,
) -> None:
    """Clear everything that belongs to the hand that just lost its role.

    Swapping roles re-points every piece of per-hand state at a different
    physical hand, so all of it has to be dropped. The buffered LSTM sequence
    is the other hand's and is mirrored the other way, keeping it would blend
    two chiralities; each debouncer's streak was counted on the old hand, so a
    held crouch or a latched grab would survive the swap and carry over.

    Both callers — the H key and Unity's control message — must run this, which
    is why it lives here rather than inline. Every debouncer added to the loop
    needs adding here too, or the swap quietly leaks that one piece of state.
    """
    lstm.flush()
    reset_last_point(last_point)
    move_debounce.reset()
    action_debounce.reset()
    watch_tap_debounce.reset()


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


def main(argv=None):
    args = parse_args(argv)
    prepare_frozen_cwd()
    show_preview = resolve_show_preview(args)

    apply_opencv_threads()
    sock = create_udp_socket()
    # Inbound half of the link. Non-blocking, drained once per frame below.
    control_sock = create_control_socket()
    # Off until Unity opens the calibrate panel.
    preview_stream = PreviewStream()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Live A/B state for the M and N keys. Starts at the config values, so a
    # run nobody touches behaves exactly as before.
    hand_complexity = MEDIAPIPE_HAND_MODEL_COMPLEXITY
    hand_det_conf = MEDIAPIPE_MIN_DETECTION_CONFIDENCE
    hands = create_hands(
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=hand_complexity,
        min_detection_confidence=hand_det_conf,
    )
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
    # Only the ACTION hand latches: its fist is grab, which must outlast the
    # tracking blind spot that ACTION_GESTURE_LATCH describes. The MOVE hand's
    # fist is walk-forward and gets no latch, so a dropout still stops the
    # player.
    move_debounce = GestureDebouncer(MOVE_GESTURE_OVERRIDES)
    action_debounce = GestureDebouncer(latch=ACTION_GESTURE_LATCH)
    watch_tap_debounce = WatchTapDebouncer()
    # The latch's only exit when the hand simply leaves — see
    # ACTION_LATCH_HAND_LOST_FRAMES.
    action_hand_lost = 0
    frame_stats = FrameStats()
    gesture_diag = GestureDiagnostics()

    # Threaded reader drops stale buffered frames while MediaPipe runs.
    cap = LatestFrameCamera()
    cam_w, cam_h, cam_fps = cap.negotiated_size()

    print(f"Combined Vision Server Running. Sending UDP to {UDP_IP}:{UDP_PORT}")
    print(f"Listening for Unity control messages on {UDP_CONTROL_IP}:{UDP_CONTROL_PORT}")
    print(
        "Gesture reports sent over TCP "
        f"{TCP_REPORT_IP}:{TCP_REPORT_PORT} when Unity asks"
    )
    print(
        f"Camera negotiated: {cam_w}x{cam_h} @ {cam_fps:.0f} fps "
        f"(requested {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS})"
    )
    if show_preview:
        print("Press Q to quit.  Press C to recalibrate look pitch neutral.")
        print("Press P to toggle puzzle mode (LSTM inference) — starts OFF.")
        print(
            "Press H to swap hand roles — action (grab/cursor/LSTM) hand starts "
            f"{hand_roles.action_hand.upper()}."
        )
        print(
            f"Press M to toggle hand model 0/1 (now {hand_complexity}).  "
            f"Press N to toggle detection confidence 0.7/0.6 "
            f"(now {hand_det_conf:.1f})."
        )
        print(
            "  Both rebuild MediaPipe (~20-30ms); ignore the [perf] window "
            "the switch lands in."
        )
    else:
        print("Headless: no preview window, no keys.")
        print("Press Ctrl-C to quit. Pitch recalibration comes from Unity.")
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

            # Unity's controls, applied before anything reads the state they
            # touch. Billed to logic_ms below; an empty queue costs one
            # non-blocking recvfrom.
            control = apply_control_messages(
                drain_control_socket(control_sock),
                pitch_cal=pitch_cal,
                preview=preview_stream,
                hand_roles=hand_roles,
            )
            if control.recalibrate_pitch:
                print("Pitch recalibration requested by Unity — hold still.")
            if control.hand_roles_changed:
                # Exactly what the H key does — same function, so the two
                # cannot drift apart as debouncers are added to the loop.
                reset_for_hand_role_change(
                    lstm=lstm,
                    last_point=last_point,
                    move_debounce=move_debounce,
                    action_debounce=action_debounce,
                    watch_tap_debounce=watch_tap_debounce,
                )
                print(
                    "Hand roles swapped by Unity — ACTION (grab/cursor/LSTM) = "
                    f"{hand_roles.action_hand.upper()}, "
                    f"MOVE = {hand_roles.move_hand.upper()}"
                )
            if control.preview_changed_to is not None:
                print(
                    "Calibration preview stream "
                    f"{'ON' if control.preview_changed_to else 'OFF'} "
                    f"(sent={preview_stream.frames_sent} "
                    f"dropped={preview_stream.frames_dropped})"
                )
            if control.chamber is not None:
                gesture_diag.set_chamber(control.chamber)
            if control.request_gesture_report:
                send_report_tcp(gesture_diag.build_report(control.chamber))

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
                watch_tap_debounce.reset()
                gesture_diag.reset()

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

            if action is not None:
                _append_hand_packet(data, action)

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
            data["leftThumbsUp"] = move_gestures["thumbs_up"]
            data["leftRockSign"] = move_gestures["rock_sign"]
            data["leftIndexLeft"] = move_gestures["index_left"]
            data["leftIndexRight"] = move_gestures["index_right"]
            data["leftIndexDown"] = move_gestures["index_down"]
            data["moveGesture"] = move_label
            data["moveGestureRaw"] = move_debounce.raw
            data["moveThumbClear"], data["moveThumbReach"] = _thumb_metrics(
                move_landmarks
            )

            action_gestures = rules_from_label(action_label)
            data["rightFist"] = action_gestures["fist"]
            data["rightOpenPalm"] = action_gestures["open_palm"]
            data["rightIndexUp"] = action_gestures["index_up"]
            data["rightPeace"] = action_gestures["peace"]
            data["rightThumbsUp"] = action_gestures["thumbs_up"]
            data["rightRockSign"] = action_gestures["rock_sign"]
            data["rightIndexLeft"] = action_gestures["index_left"]
            data["rightIndexRight"] = action_gestures["index_right"]
            data["rightIndexDown"] = action_gestures["index_down"]
            data["actionGesture"] = action_label
            data["actionGestureRaw"] = action_debounce.raw
            data["actionThumbClear"], data["actionThumbReach"] = _thumb_metrics(
                action_landmarks
            )

            if show_preview:
                if move is not None:
                    draw_hand_skeleton(
                        frame, mp_draw, mp_hands, move.mp_landmarks, True
                    )
                if action is not None:
                    draw_hand_skeleton(
                        frame,
                        mp_draw,
                        mp_hands,
                        action.mp_landmarks,
                        gesture_diag.track_confident,
                    )

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

            if apply_watch_tap_fields(
                data, move_landmarks, action_landmarks, watch_tap_debounce
            ):
                reset_last_point(last_point)
                lstm_display = "Idle"
                # A tap suppresses solo gestures; clearing the counters too
                # means the player has to re-establish one afterwards rather
                # than resuming whatever was committed before the tap.
                move_debounce.reset()
                action_debounce.reset()

            if action_hand_seen:
                action_hand_lost = 0
            else:
                action_hand_lost += 1
                if action_hand_lost >= ACTION_LATCH_HAND_LOST_FRAMES:
                    # The hand is gone, not merely unreadable. Nothing else
                    # clears a latched hold, so without this the player keeps
                    # gripping an object they walked away from.
                    action_debounce.reset()

                reset_last_point(last_point)
                lstm.register_hand_lost()
                lstm_display = lstm.get_overlay_label()
                data["lstm_gesture"] = lstm.clamp_label(lstm_display)

            if lock.face is not None:
                apply_head_rules(lock.face, data)
                pitch_cal.apply_to_payload(data, data.get("head_pitch", 0.0))
            else:
                pitch_cal.apply_to_payload(data, None)

            gesture_diag.update(
                move_landmarks=move_landmarks,
                action_landmarks=action_landmarks,
                move_raw=move_debounce.raw,
                move_committed=move_label,
                move_candidate=move_debounce.candidate,
                action_raw=action_debounce.raw,
                action_committed=action_label,
                action_candidate=action_debounce.candidate,
                watch_raw=bool(data.get("watchTapRaw")),
                watch_committed=bool(data.get("watchTap")),
                lstm=str(data.get("lstm_gesture") or "Idle"),
                puzzle=bool(data.get("puzzle_active")),
                frame=frame,
            )
            data["track_confident"] = gesture_diag.track_confident

            # Send before drawing: Unity should not wait on the debug preview.
            send_payload(sock, data)

            # After the payload, never before: the tracking packet must not
            # wait on a JPEG encode. No-op unless the calibrate panel is open,
            # and rate-limited to PREVIEW_FPS while it is.
            if preview_stream.should_send():
                # Copy first. The debug preview draws its own overlay on
                # `frame` further down, and the pitch meter would then be
                # drawn twice on the same array.
                preview_frame = frame.copy()
                draw_pitch_indicator(preview_frame, data.get("head_pitch", 0.0))
                # The ring is the answer to "move into camera view": the panel's
                # text says detection failed, this shows where the server is
                # actually looking and whether it has locked on.
                draw_lock_ring(
                    preview_frame,
                    status=lock.status,
                    center=lock.ring_center,
                    ring_size=lock.ring_size,
                    progress=lock.progress,
                )
                preview_stream.send(preview_frame)
            t_send = time.perf_counter()

            # Everything below is the debug preview — drawing, the macOS HighGUI
            # event loop, and the keyboard. None of it is needed by Unity.
            key = 0xFF
            if show_preview:
                overlay_lines = build_overlay_lines(data, lstm_display)
                # Burned into the preview on purpose: these runs get recorded
                # and compared later, and a clip that does not say which
                # settings produced it is not evidence of anything.
                overlay_lines.append(
                    (
                        f"HAND MODEL: {hand_complexity} (M)   "
                        f"DET CONF: {hand_det_conf:.1f} (N)",
                        (0, 255, 255),
                    )
                )
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
            # The preview encode also lands here, so expect logic to jump by a
            # few ms on streamed frames — only while the calibrate panel is up.
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
            if key in (ord("m"), ord("n")):
                # Both settings are baked into the graph at construction, so a
                # switch means rebuilding it. Measured at 17-32ms, and it also
                # resets MediaPipe's internal hand tracking, so the frame it
                # happens on is a stall and the [perf] window containing it is
                # not representative — read the NEXT one when comparing.
                if key == ord("m"):
                    hand_complexity = 1 - hand_complexity
                else:
                    hand_det_conf = 0.6 if hand_det_conf == 0.7 else 0.7
                hands.close()
                hands = create_hands(
                    max_num_hands=MAX_NUM_HANDS,
                    model_complexity=hand_complexity,
                    min_detection_confidence=hand_det_conf,
                )
                # The rebuilt graph starts detection from scratch; a buffered
                # sequence spanning the switch would blend two models' output.
                lstm.flush()
                move_debounce.reset()
                action_debounce.reset()
                watch_tap_debounce.reset()
                print(
                    f"[tune] hand model complexity={hand_complexity} "
                    f"detection confidence={hand_det_conf:.1f} "
                    f"(M = model, N = confidence)"
                )
            if key == ord("h"):
                hand_roles.swap(source="keyboard")
                reset_for_hand_role_change(
                    lstm=lstm,
                    last_point=last_point,
                    move_debounce=move_debounce,
                    action_debounce=action_debounce,
                    watch_tap_debounce=watch_tap_debounce,
                )
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
        control_sock.close()
        preview_stream.close()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()


if __name__ == "__main__":
    main()
