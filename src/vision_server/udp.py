import json
import socket

from vision_server.config import UDP_IP, UDP_PORT


def create_udp_socket() -> socket.socket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def default_payload() -> dict:
    """Default data sent to Unity every frame (same keys as original server)."""
    return {
        "hand_x": 0.5,
        "hand_y": 0.5,
        "hand_up": False,
        "head_yaw": 0.0,
        "head_pitch": 0.0,
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
        "landmarks": [],
        "openPalm": False,
        "isFist": False,
        "fistRotX": 0.0,
        "fistRotY": 0.0,
        "fistRotZ": 0.0,
        "palmX": -1.0,
        "palmY": -1.0,
        "indexTipX": -1.0,
        "indexTipY": -1.0,
        "hands": [],
        "lstm_gesture": "Idle",
        "player_locked": False,
        "lock_id": 0,
        "lock_status": "unlocked",
    }


def send_payload(sock: socket.socket, data: dict) -> None:
    message = json.dumps(data).encode("utf-8")
    sock.sendto(message, (UDP_IP, UDP_PORT))
