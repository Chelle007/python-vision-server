"""Inbound control channel: the socket must never stall or crash the frame loop."""

import json
import socket

import pytest

from vision_server.control import (
    apply_control_messages,
    create_control_socket,
    drain_control_socket,
)


class FakePitchCal:
    def __init__(self):
        self.requests = 0

    def request_recalibrate(self):
        self.requests += 1


@pytest.fixture
def control_pair():
    """A bound control socket plus a sender aimed at it, on an ephemeral port."""
    sock = create_control_socket("127.0.0.1", 0)
    _ip, port = sock.getsockname()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    yield sock, sender, port
    sender.close()
    sock.close()


def send(sender, port, payload):
    if not isinstance(payload, (bytes, bytearray)):
        payload = json.dumps(payload).encode("utf-8")
    sender.sendto(payload, ("127.0.0.1", port))


def drain_until(sock, expected, tries=50):
    """Drain repeatedly until `expected` messages arrive.

    Loopback delivery is not instant, and the point of the test is what the
    drain returns once the datagrams land — not how fast the kernel is.
    """
    collected = []
    for _ in range(tries):
        collected.extend(drain_control_socket(sock))
        if len(collected) >= expected:
            break
    return collected


# --- Socket behaviour ------------------------------------------------------


def test_drain_on_empty_socket_returns_empty_without_blocking(control_pair):
    """The common case: no Unity traffic, every frame, forever."""
    sock, _sender, _port = control_pair
    assert drain_control_socket(sock) == []


def test_drain_empties_the_whole_queue(control_pair):
    """One datagram per frame would build a backlog that applies late."""
    sock, sender, port = control_pair
    for i in range(5):
        send(sender, port, {"seq": i})

    received = drain_until(sock, 5)

    assert [m["seq"] for m in received] == [0, 1, 2, 3, 4]
    assert drain_control_socket(sock) == []


def test_drain_stops_at_max_datagrams(control_pair):
    """A flood must not hold the frame loop past its budget."""
    sock, sender, port = control_pair
    for i in range(10):
        send(sender, port, {"seq": i})

    first = []
    for _ in range(50):
        first = drain_control_socket(sock, max_datagrams=3)
        if first:
            break

    assert len(first) <= 3


def test_socket_is_non_blocking(control_pair):
    sock, _sender, _port = control_pair
    assert sock.gettimeout() == 0.0


def test_bind_conflict_names_the_cause():
    """A second server must fail loudly, not run without controls."""
    first = create_control_socket("127.0.0.1", 0)
    _ip, port = first.getsockname()
    try:
        with pytest.raises(OSError, match="already running"):
            create_control_socket("127.0.0.1", port)
    finally:
        first.close()


# --- Malformed input -------------------------------------------------------
# Anything at all can be sent to an open UDP port. None of it may raise into a
# loop that is holding the webcam.


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\xfe not utf-8",
        b"{not json at all",
        b"",
        b"[1, 2, 3]",  # valid JSON, wrong shape
        b'"a string"',
        b"null",
    ],
    ids=["bad-utf8", "bad-json", "empty", "json-array", "json-string", "json-null"],
)
def test_junk_is_dropped_not_raised(control_pair, payload):
    sock, sender, port = control_pair
    send(sender, port, payload)
    send(sender, port, {"recalibrate_pitch": True})

    received = drain_until(sock, 1)

    assert received == [{"recalibrate_pitch": True}]


def test_junk_does_not_hide_later_messages(control_pair):
    """A bad datagram must be skipped, not abort the drain behind it."""
    sock, sender, port = control_pair
    send(sender, port, b"{broken")
    send(sender, port, {"seq": 1})
    send(sender, port, b"also broken")
    send(sender, port, {"seq": 2})

    received = drain_until(sock, 2)

    assert [m["seq"] for m in received] == [1, 2]


# --- Message application ---------------------------------------------------


def test_recalibrate_request_reaches_the_calibrator():
    pitch_cal = FakePitchCal()

    result = apply_control_messages(
        [{"recalibrate_pitch": True}], pitch_cal=pitch_cal
    )

    assert result.recalibrate_pitch is True
    assert pitch_cal.requests == 1


def test_no_messages_changes_nothing():
    pitch_cal = FakePitchCal()

    result = apply_control_messages([], pitch_cal=pitch_cal)

    assert result.recalibrate_pitch is False
    assert pitch_cal.requests == 0


def test_false_is_not_a_request():
    """Only a truthy flag fires; a state blob carrying `false` must not."""
    pitch_cal = FakePitchCal()

    result = apply_control_messages(
        [{"recalibrate_pitch": False}], pitch_cal=pitch_cal
    )

    assert result.recalibrate_pitch is False
    assert pitch_cal.requests == 0


def test_unknown_keys_are_ignored():
    """Forward compatibility: a newer Unity build must not break the server."""
    pitch_cal = FakePitchCal()

    result = apply_control_messages(
        [{"some_future_setting": "value"}], pitch_cal=pitch_cal
    )

    assert result.recalibrate_pitch is False
    assert pitch_cal.requests == 0


def test_repeats_in_one_frame_collapse_to_one_request():
    """Two presses inside a single frame are one recalibration."""
    pitch_cal = FakePitchCal()

    result = apply_control_messages(
        [{"recalibrate_pitch": True}, {"recalibrate_pitch": True}],
        pitch_cal=pitch_cal,
    )

    assert result.recalibrate_pitch is True
    assert pitch_cal.requests == 1
