"""Calibration preview stream: off by default, rate-limited, never fatal."""

import socket

import cv2
import numpy as np
import pytest

from vision_server.control import apply_control_messages
from vision_server.preview import PreviewStream


class FakePitchCal:
    def request_recalibrate(self):
        pass


def make_frame(width=640, height=480):
    """Pure noise: the worst case for JPEG, used where size is the point."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def natural_frame(width=640, height=480):
    """Blurred noise — close enough to real webcam image statistics.

    Pure noise does not compress and encodes an order of magnitude larger than
    anything a camera produces, so sizing decisions must not be made against it.
    """
    return cv2.GaussianBlur(make_frame(width, height), (0, 0), 6)


@pytest.fixture
def receiver():
    """A bound UDP socket standing in for Unity's preview listener."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Mirrors what Unity has to do: the default receive buffer is smaller than
    # a JPEG frame, and the datagram would arrive truncated or not at all.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.5)
    yield sock
    sock.close()


@pytest.fixture
def stream(receiver):
    _ip, port = receiver.getsockname()
    stream = PreviewStream(port=port, fps=1000.0)
    yield stream
    stream.close()


# --- Gating ----------------------------------------------------------------


def test_inactive_by_default_sends_nothing(stream):
    """The panel is shut almost all the time; the default must cost nothing."""
    assert stream.active is False
    assert stream.maybe_send(natural_frame()) is False
    assert stream.frames_sent == 0


def test_set_active_reports_only_real_changes(stream):
    assert stream.set_active(True, source="unity") is True
    assert stream.set_active(True, source="unity") is False
    assert stream.set_active(False, source="unity") is False or not stream.active
    assert stream.active is False


def test_sends_once_activated(stream, receiver):
    stream.set_active(True, source="unity")

    assert stream.maybe_send(natural_frame()) is True

    payload = receiver.recv(65535)
    assert payload[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert stream.frames_sent == 1


def test_stops_when_deactivated(stream):
    stream.set_active(True, source="unity")
    stream.maybe_send(natural_frame())
    stream.set_active(False, source="unity")

    assert stream.maybe_send(natural_frame()) is False
    assert stream.frames_sent == 1


def test_none_frame_is_ignored(stream):
    stream.set_active(True, source="unity")
    assert stream.maybe_send(None) is False


# --- Rate limiting ---------------------------------------------------------


def test_rate_limit_skips_frames_inside_the_interval(receiver):
    """MediaPipe shares this thread — the stream must not run every frame."""
    _ip, port = receiver.getsockname()
    stream = PreviewStream(port=port, fps=10.0)
    try:
        stream.set_active(True, source="unity")
        first = stream.maybe_send(natural_frame())
        second = stream.maybe_send(natural_frame())

        assert first is True
        assert second is False
        assert stream.frames_sent == 1
    finally:
        stream.close()


def test_reopening_sends_immediately(receiver):
    """A reopened panel should not wait out the previous interval."""
    _ip, port = receiver.getsockname()
    stream = PreviewStream(port=port, fps=10.0)
    try:
        stream.set_active(True, source="unity")
        stream.maybe_send(natural_frame())
        stream.set_active(False, source="unity")
        stream.set_active(True, source="unity")

        assert stream.maybe_send(natural_frame()) is True
    finally:
        stream.close()


# --- Encoding --------------------------------------------------------------


def test_oversized_capture_is_downscaled(stream):
    payload = stream.encode(natural_frame(1280, 720))

    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == stream.width
    assert decoded.shape[0] == 720 * stream.width // 1280  # aspect preserved


def test_capture_sized_frame_is_sent_untouched(stream):
    """PREVIEW_WIDTH tracks CAMERA_WIDTH, so the usual case is no resize."""
    payload = stream.encode(natural_frame(stream.width, 480))

    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == stream.width
    assert decoded.shape[0] == 480


def test_small_frame_is_not_upscaled(stream):
    payload = stream.encode(natural_frame(160, 120))

    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 160


def test_encoded_frame_fits_one_datagram(stream):
    payload = stream.encode(natural_frame())

    assert payload is not None
    assert len(payload) < 65507


def test_incompressible_frame_is_dropped_not_raised(stream):
    """At full capture width, pure noise exceeds the datagram budget.

    No camera produces this, but a dropped frame must stay a dropped frame
    rather than an EMSGSIZE out of the middle of the frame loop.
    """
    stream.set_active(True, source="unity")

    assert stream.encode(make_frame()) is None
    assert stream.maybe_send(make_frame()) is False
    assert stream.frames_dropped == 1


def test_oversized_frame_is_dropped_not_raised(receiver):
    _ip, port = receiver.getsockname()
    stream = PreviewStream(port=port, fps=1000.0, max_bytes=10)
    try:
        stream.set_active(True, source="unity")

        assert stream.maybe_send(natural_frame()) is False
        assert stream.frames_dropped == 1
        assert stream.frames_sent == 0
    finally:
        stream.close()


def test_send_failure_does_not_raise():
    """Nothing listens until the panel opens; that must not kill the loop."""

    class ExplodingSocket:
        def setsockopt(self, *args):
            pass

        def sendto(self, *args):
            raise OSError("no listener")

    stream = PreviewStream(ExplodingSocket())
    stream.set_active(True, source="unity")

    assert stream.maybe_send(natural_frame()) is False
    assert stream.frames_dropped == 1


# --- Control channel integration -------------------------------------------


def test_preview_message_toggles_the_stream(stream):
    result = apply_control_messages(
        [{"preview": True}], pitch_cal=FakePitchCal(), preview=stream
    )

    assert result.preview_changed_to is True
    assert stream.active is True


def test_repeated_preview_message_reports_no_change(stream):
    """Unity may resend; only a real flip should be reported or logged."""
    apply_control_messages([{"preview": True}], pitch_cal=FakePitchCal(), preview=stream)

    result = apply_control_messages(
        [{"preview": True}], pitch_cal=FakePitchCal(), preview=stream
    )

    assert result.preview_changed_to is None
    assert stream.active is True


def test_last_message_in_a_frame_wins(stream):
    """Open-then-close inside one frame must end closed, not open."""
    result = apply_control_messages(
        [{"preview": True}, {"preview": False}],
        pitch_cal=FakePitchCal(),
        preview=stream,
    )

    assert result.preview_changed_to is None  # started off, ended off
    assert stream.active is False


def test_absent_preview_key_leaves_stream_alone(stream):
    stream.set_active(True, source="unity")

    result = apply_control_messages(
        [{"recalibrate_pitch": True}], pitch_cal=FakePitchCal(), preview=stream
    )

    assert result.preview_changed_to is None
    assert stream.active is True
