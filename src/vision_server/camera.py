"""Low-latency webcam reader: always process the newest frame.

While MediaPipe runs, OpenCV's capture queue fills with older frames. A
background thread keeps draining the device so ``read()`` returns only the
latest image (stale frames are dropped). The reader blocks until a *new*
frame arrives so the main loop does not spin on a duplicate and re-fill the
LSTM buffer with identical samples.

Transient grab failures (common on macOS AVFoundation) are retried in-place
and, if needed, the capture device is reopened. The drain thread only exits
when ``release()`` is called.

Eval still uses plain ``VideoCapture`` on MP4 files and is unchanged.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from vision_server.config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_REOPEN_SLEEP_S,
    CAMERA_SOFT_RETRIES,
    CAMERA_SOFT_RETRY_SLEEP_S,
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
        self._src = src
        self._width = width
        self._height = height
        self._fps = fps

        self._lock = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._grabbed = False
        self._frame_id = 0
        self._last_served_id = -1
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._started_at = time.monotonic()
        self._last_event: Optional[str] = None
        self._fail_grab_at: Optional[float] = None
        self._recovering = False
        self._reopen_count = 0

        self._cap = self._open_capture()

        if self._cap is not None and self._cap.isOpened():
            grabbed, frame = self._cap.read()
            with self._lock:
                if grabbed:
                    self._frame = frame
                    self._grabbed = True
                    self._frame_id = 1
                else:
                    self._last_event = "initial_read_failed"
                    self._fail_grab_at = time.monotonic()
            self._thread = threading.Thread(
                target=self._update, name="LatestFrameCamera", daemon=True
            )
            self._thread.start()
        else:
            self._last_event = "open_failed"
            # Still start the thread so we keep reopening until stop.
            self._thread = threading.Thread(
                target=self._update, name="LatestFrameCamera", daemon=True
            )
            self._thread.start()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self._src)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        self._configure_capture(cap)
        return cap

    def _configure_capture(self, cap: cv2.VideoCapture) -> None:
        """Request low-res / low-latency capture (best-effort per backend)."""
        # Best-effort on macOS; thread drain is the real fix when this is ignored.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)

    def negotiated_size(self) -> tuple[int, int, float]:
        """Actual width, height, fps reported by the driver after open."""
        with self._lock:
            cap = self._cap
        if cap is None or not cap.isOpened():
            return 0, 0, 0.0
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        )

    def _device_open(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    def _release_capture(self) -> None:
        with self._lock:
            cap = self._cap
            self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _reopen(self) -> bool:
        """Drop and reacquire the device. Caller must not hold ``_lock``."""
        self._reopen_count += 1
        self._release_capture()
        time.sleep(CAMERA_REOPEN_SLEEP_S)
        if self._stopped:
            return False
        cap = self._open_capture()
        with self._lock:
            self._cap = cap
            if cap is not None:
                self._last_event = "reopened"
                self._recovering = False
            else:
                self._last_event = "reopen_failed"
                self._recovering = True
        opened = cap is not None
        print(
            f"[cam] reopen #{self._reopen_count}: "
            f"{'ok' if opened else 'failed'} (src={self._src})"
        )
        return opened

    def _update(self) -> None:
        soft_fails = 0
        while not self._stopped:
            with self._lock:
                cap = self._cap
            if cap is None or not cap.isOpened():
                with self._lock:
                    self._recovering = True
                    self._last_event = "device_not_open"
                    self._fail_grab_at = time.monotonic()
                    # Keep last good frame if any; do not mark grab dead forever.
                    self._lock.notify_all()
                if not self._reopen():
                    time.sleep(CAMERA_REOPEN_SLEEP_S)
                soft_fails = 0
                continue

            grabbed, frame = cap.read()
            if self._stopped:
                break

            if grabbed and frame is not None:
                soft_fails = 0
                with self._lock:
                    self._frame = frame
                    self._grabbed = True
                    self._frame_id += 1
                    self._recovering = False
                    if self._last_event not in (None, "ok"):
                        print(
                            f"[cam] stream recovered "
                            f"(frame_id={self._frame_id}, "
                            f"reopens={self._reopen_count})"
                        )
                    self._last_event = "ok"
                    self._lock.notify_all()
                continue

            # Soft failure: device may still report isOpened=True.
            soft_fails += 1
            now = time.monotonic()
            with self._lock:
                device_open = self._device_open()
                self._fail_grab_at = now
                self._recovering = True
                self._last_event = (
                    "grab_failed_device_still_open"
                    if device_open
                    else "grab_failed_device_closed"
                )
                # Leave last good frame available; only wait for a *new* frame.
                self._lock.notify_all()

            print(
                "[cam] grab failed (soft): "
                f"src={self._src} "
                f"frame_id={self._frame_id} "
                f"device_open={device_open} "
                f"soft_fail={soft_fails}/{CAMERA_SOFT_RETRIES} "
                f"uptime_s={now - self._started_at:.1f}"
            )

            if soft_fails < CAMERA_SOFT_RETRIES:
                time.sleep(CAMERA_SOFT_RETRY_SLEEP_S)
                continue

            soft_fails = 0
            self._reopen()

        with self._lock:
            if self._last_event is None or self._last_event == "ok":
                self._last_event = "stopped"
            self._recovering = False
            self._lock.notify_all()

    def isOpened(self) -> bool:
        """True while the reader is active (even mid-reconnect)."""
        return (
            not self._stopped
            and self._thread is not None
            and self._thread.is_alive()
        )

    def diagnostics(self) -> dict[str, Any]:
        """Snapshot for crash/audit logs when a read fails."""
        with self._lock:
            frame_id = self._frame_id
            last_served = self._last_served_id
            grabbed = self._grabbed
            stopped = self._stopped
            last_event = self._last_event
            fail_at = self._fail_grab_at
            recovering = self._recovering
            reopens = self._reopen_count
            device_open = self._device_open()
        thread_alive = (
            self._thread is not None and self._thread.is_alive()
        )
        now = time.monotonic()
        return {
            "src": self._src,
            "uptime_s": round(now - self._started_at, 1),
            "frame_id": frame_id,
            "last_served_id": last_served,
            "grabbed": grabbed,
            "stopped": stopped,
            "device_open": device_open,
            "thread_alive": thread_alive,
            "recovering": recovering,
            "reopen_count": reopens,
            "exit_reason": last_event or "unknown",
            "seconds_since_fail": (
                round(now - fail_at, 2) if fail_at is not None else None
            ),
        }

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Block until a new frame, then return a copy (safe for drawing).

        Returns ``(False, None)`` only after ``release()`` / stop. While the
        camera is recovering, this waits for the next good frame instead of
        treating a transient glitch as fatal.
        """
        with self._lock:
            while not self._stopped:
                if (
                    self._grabbed
                    and self._frame is not None
                    and self._frame_id != self._last_served_id
                ):
                    self._last_served_id = self._frame_id
                    return True, self._frame.copy()
                self._lock.wait(timeout=0.5)
            return False, None

    def release(self) -> None:
        with self._lock:
            self._stopped = True
            self._lock.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            self._thread = None
        self._release_capture()
