import json
import socket

from vision_server.config import UDP_IP, UDP_PORT


def create_udp_socket() -> socket.socket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def default_payload() -> dict:
    """Default data sent to Unity every frame.

    The v1 single-hand fields (``hand_x``/``hand_y``/``hand_up``/``openPalm``/
    ``isFist`` and the always-empty top-level ``landmarks``) were removed once
    the left/right split landed — nothing on either side read them any more.
    Cursor position now travels as ``palmX``/``palmY``.
    """
    return {
        "head_yaw": 0.0,
        "head_pitch": 0.0,
        "head_pitch_raw": 0.0,
        "pitch_cal_status": "idle",
        "pitch_cal_neutral": 0.0,
        "pitch_calibrated": False,
        "tilt_left": False,
        "tilt_right": False,
        "leftFist": False,
        "leftOpenPalm": False,
        "leftIndexUp": False,
        "leftPeace": False,
        "rightFist": False,
        "rightOpenPalm": False,
        "rightIndexUp": False,
        "rightPeace": False,
        "watchTap": False,
        "watchTapDistance": None,
        "fistRotX": 0.0,
        "fistRotY": 0.0,
        "fistRotZ": 0.0,
        "palmX": -1.0,
        "palmY": -1.0,
        "indexTipX": -1.0,
        "indexTipY": -1.0,
        "hands": [],
        "lstm_gesture": "Idle",
        "puzzle_active": False,
        "puzzle_gate_source": "default",
        # left*/right* fields above are ROLE fields (left = MOVE hand,
        # right = ACTION hand). These say which physical hand fills each.
        "action_hand": "right",
        "move_hand": "left",
        "hand_roles_source": "default",
        "player_locked": False,
        "lock_id": 0,
        "lock_status": "unlocked",
    }


def send_payload(sock: socket.socket, data: dict) -> None:
    message = json.dumps(data).encode("utf-8")
    sock.sendto(message, (UDP_IP, UDP_PORT))
