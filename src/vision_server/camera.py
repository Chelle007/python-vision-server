"""Low-latency webcam reader: always process the newest frame.

While MediaPipe runs, OpenCV's capture queue fills with older frames. A
background thread keeps draining the device so ``read()`` returns only the
latest image (stale frames are dropped). The reader blocks until a *new*
frame arrives so the main loop does not spin on a duplicate and re-fill the
LSTM buffer with identical samples.

Eval still uses plain ``VideoCapture`` on MP4 files and is unchanged.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from vision_server.config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
)


class LatestFrameCamera:
    """Threaded ``VideoCapture`` that surfaces only the most recent frame."""

    def __init__(
        self,
        src: int = CAMERA_INDEX,
        *,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ):
        self._cap = cv2.VideoCapture(src)
        self._configure_capture(width=width, height=height, fps=fps)

        self._lock = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._grabbed = False
        self._frame_id = 0
        self._last_served_id = -1
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

        if self._cap.isOpened():
            grabbed, frame = self._cap.read()
            with self._lock:
                self._grabbed = grabbed
                if grabbed:
                    self._frame = frame
                    self._frame_id = 1
            self._thread = threading.Thread(
                target=self._update, name="LatestFrameCamera", daemon=True
            )
            self._thread.start()

    def _configure_capture(self, *, width: int, height: int, fps: int) -> None:
        """Request low-res / low-latency capture (best-effort per backend)."""
        # Best-effort on macOS; thread drain is the real fix when this is ignored.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def negotiated_size(self) -> tuple[int, int, float]:
        """Actual width, height, fps reported by the driver after open."""
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0),
        )

    def _update(self) -> None:
        while not self._stopped:
            grabbed, frame = self._cap.read()
            with self._lock:
                self._grabbed = grabbed
                if grabbed:
                    self._frame = frame
                    self._frame_id += 1
                    self._lock.notify_all()
                else:
                    self._lock.notify_all()
                    break

    def isOpened(self) -> bool:
        """Match OpenCV ``VideoCapture.isOpened`` naming."""
        return self._cap.isOpened() and not self._stopped

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Block until a new frame, then return a copy (safe for drawing)."""
        with self._lock:
            while (
                self._grabbed
                and self._frame is not None
                and self._frame_id == self._last_served_id
                and not self._stopped
            ):
                self._lock.wait(timeout=0.5)

            if self._stopped or not self._grabbed or self._frame is None:
                return False, None

            self._last_served_id = self._frame_id
            return True, self._frame.copy()

    def release(self) -> None:
        with self._lock:
            self._stopped = True
            self._lock.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        self._cap.release()
