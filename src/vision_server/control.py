"""Inbound UDP control channel (Unity -> server).

``udp.py`` is send-only: the server pushes a tracking payload at Unity every
frame and never listens. This module is the other direction — a second socket,
on its own port, that lets Unity drive server state which until now was
reachable only from the debug preview's keyboard.

That keyboard is the whole reason this exists. ``SHOW_PREVIEW = False`` is the
right setting for the demo machine (no landmark drawing, no HighGUI event
loop), but it also disables ``cv2.waitKey`` and with it C (recalibrate pitch),
P (puzzle gate) and H (swap hand roles). Before this module, turning the
preview off meant losing those three controls with no replacement — so the
demo either paid for a debug window it never looked at, or shipped without
calibration.

Two properties matter and both are load-bearing:

*Non-blocking.* The socket is drained from the same loop that runs MediaPipe,
so a blocking read would stall the whole pipeline waiting on a packet that may
never come.

*Drained empty, not one-per-frame.* Reading a single datagram per frame would
let a burst build a backlog that takes seconds to clear, and every message in
it would be applied late.

Adding a message type is deliberately small: extend :class:`ControlResult` and
add a branch in :func:`apply_control_messages`. ``HandRoles.set_action_hand``
and ``PuzzleGate.set_active`` are the next two callers, and both are already
shaped for it — idempotent setters that report whether they actually changed
anything, so Unity can resend freely.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass

from vision_server.config import (
    UDP_CONTROL_IP,
    UDP_CONTROL_MAX_DATAGRAMS,
    UDP_CONTROL_PORT,
    UDP_CONTROL_RECV_BYTES,
)


@dataclass(frozen=True)
class ControlResult:
    """What a frame's worth of control messages actually asked for.

    Returned rather than logged so the caller can run side effects the message
    itself cannot know about — a hand-role change has to flush the LSTM and
    reset every debouncer, none of which this module can reach.
    """

    recalibrate_pitch: bool = False
    # True/False only when the stream actually changed state, so a resend of
    # the same value does not re-log or restart anything. None means the
    # messages this frame said nothing about it.
    preview_changed_to: bool | None = None
    # True only when the action hand actually ended the frame somewhere new.
    # A resend of the current side, or a swap and swap-back inside one frame,
    # leaves this False so the caller does not flush a live gesture for nothing.
    hand_roles_changed: bool = False


def create_control_socket(
    ip: str = UDP_CONTROL_IP,
    port: int = UDP_CONTROL_PORT,
) -> socket.socket:
    """Bind the control port, non-blocking.

    Binding failure is left to raise. A port already in use means a second
    server instance is running, which is broken anyway — two processes cannot
    share the webcam — and a server that silently accepts no control is exactly
    the kind of quiet failure this channel was added to remove.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((ip, port))
    except OSError as exc:
        sock.close()
        raise OSError(
            f"could not bind control socket on {ip}:{port} — is another "
            f"vision server already running? ({exc})"
        ) from exc
    sock.setblocking(False)
    return sock


def drain_control_socket(
    sock: socket.socket,
    *,
    max_datagrams: int = UDP_CONTROL_MAX_DATAGRAMS,
) -> list[dict]:
    """Read every queued datagram, returning the ones that parsed as objects.

    Junk is dropped silently rather than raised. Anything at all can be sent to
    an open UDP port, and a stray packet must not be able to kill a frame loop
    that is also holding the webcam.
    """
    messages: list[dict] = []

    for _ in range(max_datagrams):
        try:
            raw, _addr = sock.recvfrom(UDP_CONTROL_RECV_BYTES)
        except BlockingIOError:
            break  # Queue empty — the normal exit, once per frame.
        except OSError:
            break  # Socket closed or errored; nothing useful left to read.

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

        if isinstance(parsed, dict):
            messages.append(parsed)

    return messages


def _apply_hand_roles(messages: list[dict], hand_roles) -> bool:
    """Apply hand-role messages; return True if the action hand actually moved.

    Two shapes are accepted. ``{"action_hand": "left"}`` is the one to prefer:
    absolute, so a duplicated or dropped datagram cannot desync Unity's display
    from the server, and safe to resend every frame. ``{"cmd": "swap_hands"}``
    is a relative toggle kept for the existing Unity button — it works, but a
    lost datagram leaves the two sides disagreeing about which hand is which.

    The verdict compares the side before and after the whole batch rather than
    trusting the setters, so a swap and a swap-back inside one frame correctly
    reports no change and spares a live gesture the flush.
    """
    before = hand_roles.action_hand

    for msg in messages:
        if "action_hand" in msg:
            try:
                hand_roles.set_action_hand(msg["action_hand"], source="unity")
            except ValueError:
                # Not a side. Anything can arrive on an open UDP port, and a
                # typo from Unity must not take the frame loop down.
                continue
        elif msg.get("cmd") == "swap_hands":
            hand_roles.swap(source="unity")

    return hand_roles.action_hand != before


def apply_control_messages(
    messages: list[dict],
    *,
    pitch_cal,
    preview=None,
    hand_roles=None,
) -> ControlResult:
    """Apply a frame's control messages to server state.

    Repeats within one frame collapse to a single action: Unity may resend, and
    two recalibrate requests in the same frame are one request. Where messages
    disagree the last one wins, since they arrived in that order.
    """
    recalibrate = any(bool(msg.get("recalibrate_pitch")) for msg in messages)

    if recalibrate:
        pitch_cal.request_recalibrate()

    preview_changed_to = None
    if preview is not None:
        wanted = None
        for msg in messages:
            if "preview" in msg:
                wanted = bool(msg["preview"])
        if wanted is not None and preview.set_active(wanted, source="unity"):
            preview_changed_to = wanted

    hand_roles_changed = (
        _apply_hand_roles(messages, hand_roles) if hand_roles is not None else False
    )

    return ControlResult(
        recalibrate_pitch=recalibrate,
        preview_changed_to=preview_changed_to,
        hand_roles_changed=hand_roles_changed,
    )
