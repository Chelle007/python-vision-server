"""PitchCalibrator: lock-gated sampling, stable mean as neutral, C / lock switch."""

from vision_server.gestures.head.pitch_cal import PitchCalibrator


class _Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _patch_clock(monkeypatch, clock: _Clock) -> None:
    monkeypatch.setattr(
        "vision_server.gestures.head.pitch_cal.time.monotonic", clock
    )


def test_unlock_is_idle_and_passthrough():
    cal = PitchCalibrator(sample_s=0.5, max_std=0.05, min_samples=5)
    out = cal.update(locked=False, lock_id=0, raw_pitch=0.2)
    assert out == 0.2
    assert cal.status == "idle"
    assert not cal.calibrated


def test_stable_window_sets_neutral(monkeypatch):
    cal = PitchCalibrator(sample_s=1.0, max_std=0.05, min_samples=5)
    clock = _Clock(0.0)
    _patch_clock(monkeypatch, clock)
    base = 0.18
    last = 0.0
    mid = None
    for i in range(15):
        last = cal.update(locked=True, lock_id=1, raw_pitch=base)
        if i == 3:
            mid = last  # during cal should report 0 (not raw rest as look-down)
        clock.advance(0.1)

    assert mid == 0.0
    assert cal.calibrated
    assert cal.status == "calibrated"
    assert abs(cal.neutral - base) < 0.02
    assert abs(last) < 0.02


def test_noisy_window_still_finishes(monkeypatch):
    """Live face jitter used to exceed max_std and block forever."""
    cal = PitchCalibrator(sample_s=0.5, max_std=0.02, min_samples=5)
    clock = _Clock(0.0)
    _patch_clock(monkeypatch, clock)
    for i in range(12):
        # Alternate so std >> max_std
        raw = 0.15 + (0.05 if i % 2 else -0.05)
        cal.update(locked=True, lock_id=1, raw_pitch=raw)
        clock.advance(0.1)
    assert cal.calibrated
    assert 0.05 < cal.neutral < 0.25


def test_look_up_is_negative_after_cal(monkeypatch):
    cal = PitchCalibrator(sample_s=0.5, max_std=0.05, min_samples=5)
    clock = _Clock(0.0)
    _patch_clock(monkeypatch, clock)
    for _ in range(12):
        cal.update(locked=True, lock_id=2, raw_pitch=0.2)
        clock.advance(0.1)

    assert cal.calibrated
    cal_pitch = cal.update(locked=True, lock_id=2, raw_pitch=0.0)
    assert cal_pitch < -0.15


def test_lock_id_change_restarts_cal(monkeypatch):
    cal = PitchCalibrator(sample_s=0.5, max_std=0.05, min_samples=5)
    clock = _Clock(0.0)
    _patch_clock(monkeypatch, clock)
    for _ in range(12):
        cal.update(locked=True, lock_id=1, raw_pitch=0.1)
        clock.advance(0.1)

    assert cal.calibrated
    cal.update(locked=True, lock_id=2, raw_pitch=0.1)
    assert not cal.calibrated
    assert cal.status in ("calibrating", "hold_still")


def test_request_recalibrate(monkeypatch):
    cal = PitchCalibrator(sample_s=0.5, max_std=0.05, min_samples=5)
    clock = _Clock(0.0)
    _patch_clock(monkeypatch, clock)
    for _ in range(12):
        cal.update(locked=True, lock_id=1, raw_pitch=0.1)
        clock.advance(0.1)

    assert cal.calibrated
    cal.request_recalibrate()
    assert not cal.calibrated
    cal.update(locked=True, lock_id=1, raw_pitch=0.1)
    assert cal.status in ("calibrating", "hold_still")
    assert not cal.calibrated


def test_dense_camera_frames_still_finish(monkeypatch):
    """Rolling sample span is always < sample_s on a live camera; must use wall time."""
    cal = PitchCalibrator(sample_s=1.0, max_std=0.05, min_samples=20)
    clock = _Clock(0.0)
    _patch_clock(monkeypatch, clock)
    # ~30 fps over 1.2s — sample deque span stays under 1.0 after trim, wall time hits 1.0.
    for _ in range(40):
        cal.update(locked=True, lock_id=1, raw_pitch=0.16)
        clock.advance(0.033)
    assert cal.calibrated
    assert abs(cal.neutral - 0.16) < 0.02


def test_apply_to_payload_fields():
    cal = PitchCalibrator(sample_s=0.5, max_std=0.05, min_samples=3)
    data = {
        "player_locked": False,
        "lock_id": 0,
        "head_pitch": 0.0,
    }
    cal.apply_to_payload(data, 0.12)
    assert data["head_pitch_raw"] == 0.12
    assert data["pitch_cal_status"] == "idle"
    assert data["pitch_calibrated"] is False
