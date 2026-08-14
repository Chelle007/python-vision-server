"""Pain-list retroactive labelling and camera chips."""

from types import SimpleNamespace

from vision_server.control import apply_control_messages
from vision_server.gesture_report import (
    ISSUE_DWELL_TOO_SHORT,
    ISSUE_HAND_TOO_SMALL,
    ISSUE_POOR_LIGHTING,
    GestureDiagnostics,
)


def _hand(palm=0.2):
    lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lms[0] = SimpleNamespace(x=0.5, y=0.7, z=0.0)
    lms[9] = SimpleNamespace(x=0.5, y=0.7 - palm, z=0.0)
    return lms


def _tick(d, t, **kwargs):
    defaults = dict(
        move_landmarks=_hand(),
        action_landmarks=_hand(),
        move_raw="none",
        move_committed="none",
        move_candidate="none",
        action_raw="none",
        action_committed="none",
        action_candidate="none",
        watch_raw=False,
        watch_committed=False,
        lstm="Idle",
        puzzle=False,
        now=t,
    )
    defaults.update(kwargs)
    d.update(**defaults)


def test_idle_never_enters_rows():
    d = GestureDiagnostics()
    for i in range(40):
        _tick(d, i * 0.05, move_landmarks=None, action_landmarks=None)
    report = d.build_report(chamber=1)
    assert report["type"] == "gesture_report"
    for row in report["gestures"]:
        assert row["successes"] == 0
        assert row["attempts"] == 0


def test_jump_retries_from_failed_index_up():
    d = GestureDiagnostics()
    t = 0.0
    for _ in range(3):
        _tick(d, t, move_raw="index_up", move_candidate="index_up")
        t += 0.05
    for _ in range(3):
        _tick(d, t)
        t += 0.05
    _tick(
        d,
        t,
        move_raw="index_up",
        move_committed="index_up",
        move_candidate="index_up",
    )
    jump = d.stats["jump"]
    assert jump.successes == 1
    assert jump.attempts == 2
    assert jump.retries() == 1.0
    assert jump.dominant_issue() == ISSUE_DWELL_TOO_SHORT


def test_grab_is_action_fist_not_move():
    d = GestureDiagnostics()
    t = 0.0
    _tick(
        d,
        t,
        move_raw="fist",
        move_committed="fist",
        move_candidate="fist",
    )
    assert d.stats["grab"].successes == 0
    t += 0.05
    _tick(
        d,
        t,
        action_raw="fist",
        action_committed="fist",
        action_candidate="fist",
    )
    assert d.stats["grab"].successes == 0


def test_watch_tap_and_pull_lever():
    d = GestureDiagnostics()
    t = 0.0
    for _ in range(4):
        _tick(d, t, watch_raw=True)
        t += 0.05
    _tick(d, t, watch_raw=True, watch_committed=True)
    assert d.stats["watch_tap"].successes == 1
    t += 0.05
    _tick(d, t, puzzle=True, lstm="Pull_Lever", action_landmarks=_hand())
    assert d.stats["pull_lever"].successes == 1
    t += 0.05
    _tick(d, t, puzzle=False, lstm="Pull_Lever")
    assert d.stats["pull_lever"].successes == 1


def test_tiny_hand_camera_distance():
    d = GestureDiagnostics()
    tiny = _hand(palm=0.03)
    for i in range(20):
        _tick(d, i * 0.05, move_landmarks=tiny, action_landmarks=tiny)
    report = d.build_report()
    assert report["camera"]["distance"] == "far"
    assert report["advice_issue"] == ISSUE_HAND_TOO_SMALL


class _Plane:
    def __init__(self, value):
        self.value = float(value)

    def astype(self, _dtype):
        return self

    def __mul__(self, other):
        return _Plane(self.value * float(other))

    __rmul__ = __mul__

    def __add__(self, other):
        return _Plane(self.value + other.value)

    def mean(self):
        return self.value


class _FakeFrame:
    """BGR-looking crop without importing numpy in tests."""

    def __init__(self, luma, height=48, width=64):
        self.shape = (height, width, 3)
        self._luma = float(luma)

    @property
    def size(self):
        return self.shape[0] * self.shape[1] * 3

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 3:
            return _Plane(self._luma)
        return _FakeFrame(self._luma, max(2, self.shape[0] // 4), max(2, self.shape[1] // 4))


def test_dark_frame_sets_lighting_bad():
    d = GestureDiagnostics()
    dark = _FakeFrame(10)
    for i in range(20):
        _tick(d, i * 0.05, frame=dark)
    report = d.build_report()
    assert report["camera"]["lighting"] == "bad"
    assert report["advice_issue"] == ISSUE_POOR_LIGHTING


def test_bright_frame_keeps_lighting_ok():
    d = GestureDiagnostics()
    bright = _FakeFrame(200)
    for i in range(20):
        _tick(d, i * 0.05, frame=bright)
    report = d.build_report()
    assert report["camera"]["lighting"] == "ok"


def test_control_requests_report():
    class FakePitch:
        def request_recalibrate(self):
            pass

    result = apply_control_messages(
        [{"cmd": "gesture_report", "chamber": 2}],
        pitch_cal=FakePitch(),
    )
    assert result.request_gesture_report is True
    assert result.chamber == 2
