"""Calibration preview: JPEG frames pushed to Unity while the panel is open.

Unity's calibrate panel used to show a `WebCamTexture` of its own. That is two
processes opening one webcam — tolerated on macOS, refused on Windows, which is
where the demo runs. Since the server has already decoded the frame for
MediaPipe, the cheaper and portable answer is to send Unity a picture of it
rather than let Unity open the device.

Same shape as :class:`~vision_server.puzzle_gate.PuzzleGate` and
:class:`~vision_server.hand_roles.HandRoles`: one piece of state behind one
idempotent setter, so Unity can resend freely and the caller learns whether
anything actually changed.

Off by default and off the moment the panel closes. While active it costs a
resize plus a JPEG encode at :data:`PREVIEW_FPS`, paid on the same thread as
MediaPipe — acceptable only because it happens in a paused menu.
"""

from __future__ import annotations

import socket
import time

import cv2

from vision_server.config import (
    PREVIEW_FPS,
    PREVIEW_JPEG_QUALITY,
    PREVIEW_MAX_BYTES,
    PREVIEW_SOCKET_BUFFER_BYTES,
    PREVIEW_WIDTH,
    UDP_IP,
    UDP_PREVIEW_PORT,
)


class PreviewStream:
    """Sends downscaled JPEG frames to Unity while calibration is on screen."""

    def __init__(
        self,
        sock: socket.socket | None = None,
        *,
        ip: str = UDP_IP,
        port: int = UDP_PREVIEW_PORT,
        fps: float = PREVIEW_FPS,
        width: int = PREVIEW_WIDTH,
        quality: int = PREVIEW_JPEG_QUALITY,
        max_bytes: int = PREVIEW_MAX_BYTES,
    ):
        self._sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._owns_sock = sock is None
        # Without this a JPEG frame is simply unsendable on macOS: the default
        # datagram ceiling is 9216 bytes and sendto raises EMSGSIZE above it.
        try:
            self._sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, PREVIEW_SOCKET_BUFFER_BYTES
            )
        except OSError:
            pass  # Best effort; the size guard in encode() still applies.
        self.ip = ip
        self.port = port
        self.fps = float(fps)
        self.width = int(width)
        self.quality = int(quality)
        self.max_bytes = int(max_bytes)

        self.active = False
        self.source = "default"
        self.frames_sent = 0
        self.frames_dropped = 0
        self._last_sent: float | None = None

    def set_active(self, value: bool, *, source: str = "unknown") -> bool:
        """Set stream state; return True only if this call actually flipped it."""
        value = bool(value)
        self.source = source
        if value == self.active:
            return False

        self.active = value
        # Restart the clock so reopening the panel sends immediately rather
        # than waiting out the interval left over from last time.
        self._last_sent = None
        return True

    def _due(self, now: float) -> bool:
        if self._last_sent is None:
            return True
        return (now - self._last_sent) >= (1.0 / self.fps)

    def should_send(self) -> bool:
        """Whether this frame is due to go out.

        Split from :meth:`send` so the caller can skip work that only matters
        for a frame that is actually leaving — the copy and overlay drawing
        cost nothing on the nine frames in ten that are rate-limited away.
        """
        return self.active and self._due(time.monotonic())

    def encode(self, frame) -> bytes | None:
        """Downscale and JPEG-encode one frame. None if it will not fit."""
        height, width = frame.shape[:2]
        if width > self.width:
            scale = self.width / float(width)
            frame = cv2.resize(
                frame,
                (self.width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        )
        if not ok:
            return None

        payload = buf.tobytes()
        if len(payload) > self.max_bytes:
            return None
        return payload

    def maybe_send(self, frame) -> bool:
        """Send one frame if the stream is on and the interval has elapsed."""
        if not self.should_send():
            return False

        return self.send(frame)

    def send(self, frame) -> bool:
        """Encode and send one frame now, without checking the rate limit.

        Callers that used :meth:`should_send` to decide whether to prepare the
        frame come here; everyone else wants :meth:`maybe_send`.
        """
        if frame is None:
            return False

        now = time.monotonic()
        payload = self.encode(frame)
        if payload is None:
            self.frames_dropped += 1
            # Still stamp the clock: a frame too big to send will very likely
            # repeat, and retrying it every frame would cost an encode each
            # time for nothing.
            self._last_sent = now
            return False

        try:
            self._sock.sendto(payload, (self.ip, self.port))
        except OSError:
            # Nothing listens until the panel opens, and on some platforms that
            # surfaces as ICMP-driven errors here. Never break the frame loop.
            self.frames_dropped += 1
            self._last_sent = now
            return False

        self.frames_sent += 1
        self._last_sent = now
        return True

    def close(self) -> None:
        if self._owns_sock:
            self._sock.close()
