"""Pitch neutral auto-calibration after player lock.

Subtracts resting head_pitch so look-up/down are measured relative to a
comfortable sit-and-look-at-screen pose (rest ≈ 0 after cal).
"""

from __future__ import annotations

import time
from collections import deque

from vision_server.config import (
    PITCH_CAL_MAX_STD,
    PITCH_CAL_MIN_SAMPLES,
    PITCH_CAL_SAMPLE_S,
)


class PitchCalibrator:
    """Stateful pitch zeroing; update every frame after raw head_pitch is known."""

    def __init__(
        self,
        sample_s: float = PITCH_CAL_SAMPLE_S,
        max_std: float = PITCH_CAL_MAX_STD,
        min_samples: int = PITCH_CAL_MIN_SAMPLES,
    ):
        self.sample_s = sample_s
        self.max_std = max_std
        self.min_samples = min_samples
        self.neutral = 0.0
        self.calibrated = False
        self.status = "idle"  # idle | calibrating | hold_still | calibrated
        self._lock_id = -1
        self._samples: deque[tuple[float, float]] = deque()
        self._window_start: float | None = None
        self._manual_pending = False

    def request_recalibrate(self) -> None:
        """Hotkey / UI: re-zero using a new stable sample window."""
        self._manual_pending = True
        self._reset_window(keep_lock_id=True)
        self.calibrated = False
        self.neutral = 0.0
        self.status = "calibrating"

    def _reset_window(self, *, keep_lock_id: bool) -> None:
        self._samples.clear()
        self._window_start = None
        if not keep_lock_id:
            self._lock_id = -1

    def _clear_for_unlock(self) -> None:
        self._reset_window(keep_lock_id=False)
        self.calibrated = False
        self.neutral = 0.0
        self.status = "idle"
        self._manual_pending = False

    def _std(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        vals = [p for _, p in self._samples]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return var**0.5

    @staticmethod
    def _median(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        mid = len(s) // 2
        if len(s) % 2:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    def _finish(self, raw: float) -> float:
        vals = [p for _, p in self._samples]
        self.neutral = self._median(vals)
        self.calibrated = True
        self.status = "calibrated"
        self._samples.clear()
        return raw - self.neutral

    def update(
        self,
        *,
        locked: bool,
        lock_id: int,
        raw_pitch: float | None,
    ) -> float:
        """Return calibrated pitch. raw_pitch None if no face this frame."""
        if not locked:
            self._clear_for_unlock()
            return 0.0 if raw_pitch is None else float(raw_pitch)

        if lock_id != self._lock_id or self._manual_pending:
            # New player / first lock / user asked for C.
            self._lock_id = lock_id
            self._manual_pending = False
            self.calibrated = False
            self.neutral = 0.0
            self._reset_window(keep_lock_id=True)
            self.status = "calibrating"

        if raw_pitch is None:
            # Face blip during cal: keep waiting, don't corrupt window with fake zeros.
            if not self.calibrated:
                self.status = "calibrating"
            # Keep last calibrated offset when face briefly drops after cal.
            return 0.0

        raw = float(raw_pitch)

        if self.calibrated:
            self.status = "calibrated"
            return raw - self.neutral

        now = time.monotonic()
        if self._window_start is None:
            self._window_start = now

        self._samples.append((now, raw))
        # Keep recent samples for median (window slightly longer than target).
        cutoff = now - (self.sample_s * 1.25)
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        n = len(self._samples)
        std = self._std()
        # Wall-clock since sampling started — NOT span of the rolling deque.
        # A full sliding window of width sample_s always has span < sample_s, which
        # previously made calibration never finish on a live camera.
        # Avoid "or" with window_start: 0.0 is a valid clock and is falsy in Python.
        assert self._window_start is not None
        elapsed = now - self._window_start

        # Always finish after wall sample_s + enough frames (median handles noise).
        if elapsed >= self.sample_s and n >= self.min_samples:
            return self._finish(raw)

        # Soft hint only — never blocks finish above.
        if n >= max(5, self.min_samples // 3) and std > self.max_std:
            self.status = "hold_still"
        else:
            self.status = "calibrating"

        # While calibrating, report 0 so Unity won't false-arm look-up/down.
        return 0.0

    def apply_to_payload(self, data: dict, raw_pitch: float | None) -> None:
        """Write raw + calibrated pitch and cal status onto the UDP dict."""
        locked = bool(data.get("player_locked"))
        lock_id = int(data.get("lock_id") or 0)
        calibrated = self.update(
            locked=locked,
            lock_id=lock_id,
            raw_pitch=raw_pitch,
        )
        data["head_pitch_raw"] = (
            0.0 if raw_pitch is None else round(float(raw_pitch), 3)
        )
        data["head_pitch"] = round(calibrated, 3)
        data["pitch_cal_status"] = self.status
        data["pitch_cal_neutral"] = round(self.neutral, 3)
        data["pitch_calibrated"] = self.calibrated
