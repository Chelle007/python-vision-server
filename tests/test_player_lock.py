"""Player lock: reserved seat + auto challenger."""

from types import SimpleNamespace

from vision_server.tracking.player_lock import (
    FaceCandidate,
    HandCandidate,
    PlayerLock,
    _dist,
)


def _hand(
    cx,
    cy,
    length=0.12,
    handedness="Right",
) -> HandCandidate:
    landmarks = [SimpleNamespace(x=cx, y=cy, z=0.0) for _ in range(21)]
    landmarks[0] = SimpleNamespace(x=cx, y=cy + length / 2, z=0.0)
    landmarks[12] = SimpleNamespace(x=cx, y=cy - length / 2, z=0.0)
    for i in (5, 9, 13, 17):
        landmarks[i] = SimpleNamespace(x=cx, y=cy, z=0.0)
    return HandCandidate(
        landmarks=landmarks,
        mp_landmarks=None,
        handedness=handedness,
        world_landmarks=None,
        center=(cx, cy),
        length=length,
    )


def _face(cx=0.5, cy=0.45, width=0.2, height=0.25) -> FaceCandidate:
    return FaceCandidate(
        landmarks=SimpleNamespace(),
        center=(cx, cy),
        width=width,
        height=height,
    )


def test_dist():
    assert abs(_dist((0.0, 0.0), (3.0, 4.0)) - 5.0) < 1e-6


def test_acquire_centered_face():
    lock = PlayerLock()
    face = _face(0.5, 0.45)
    r0 = lock.update([face], [], now=0.0)
    assert r0.locked is False
    assert r0.status == "challenger"
    r1 = lock.update([face], [], now=0.4)
    assert r1.locked is True
    assert r1.status == "locked"
    assert r1.lock_id == 1
    assert r1.flush_lstm is True


def test_rejects_far_hand():
    lock = PlayerLock()
    face = _face(0.5, 0.4, width=0.08, height=0.2)
    lock.update([face], [], now=0.0)
    # Far corner relative to a small face — bystander-scale reach.
    result = lock.update(
        [face], [_hand(0.05, 0.95, length=0.12)], now=0.4
    )
    assert result.locked is True
    assert result.left is None
    assert result.right is None


def test_accepts_player_hand_away_from_face():
    lock = PlayerLock()
    face = _face(0.5, 0.4, width=0.2, height=0.25)
    # Typical play pose: hand lower / toward screen edge, still same person.
    hand = _hand(0.72, 0.75, length=0.12, handedness="Right")
    lock.update([face], [], now=0.0)
    result = lock.update([face], [hand], now=0.4)
    assert result.locked is True
    assert result.right is not None or result.left is not None


def test_accepts_small_fist_near_face():
    lock = PlayerLock()
    face = _face(0.5, 0.45, width=0.2, height=0.25)
    fist = _hand(0.55, 0.55, length=0.04, handedness="Right")
    lock.update([face], [], now=0.0)
    result = lock.update([face], [fist], now=0.4)
    assert result.locked is True
    assert result.right is not None or result.left is not None


def test_hand_assignment_prefers_lower_cost():
    lock = PlayerLock()
    face = _face(0.5, 0.4, width=0.2, height=0.25)
    left = _hand(0.35, 0.55, length=0.12, handedness="Left")
    right = _hand(0.65, 0.55, length=0.12, handedness="Right")
    lock.update([face], [], now=0.0)
    acquired = lock.update([face], [left, right], now=0.4)
    assert acquired.locked is True
    assert acquired.left is not None and acquired.right is not None

    swapped = lock.update([face], [right, left], now=0.5)
    assert abs(swapped.left.center[0] - 0.35) < 1e-6
    assert abs(swapped.right.center[0] - 0.65) < 1e-6


def test_seat_resume_same_lock_id():
    lock = PlayerLock()
    face = _face(0.5, 0.45)
    lock.update([face], [], now=0.0)
    lock.update([face], [], now=0.4)
    assert lock.lock_id == 1

    miss = lock.update([], [], now=1.0)
    assert miss.locked is True
    assert miss.face is None
    assert miss.lock_id == 1

    back = lock.update([_face(0.52, 0.46)], [], now=1.2)
    assert back.locked is True
    assert back.face is not None
    assert back.lock_id == 1
    assert back.flush_lstm is False


def test_challenger_takeover_after_seat_empty():
    lock = PlayerLock()
    player = _face(0.5, 0.45, width=0.2)
    lock.update([player], [], now=0.0)
    lock.update([player], [], now=0.4)
    assert lock.lock_id == 1

    # Player leaves; seat reserved
    lock.update([], [], now=1.0)

    # Bystander far/small should not be eligible; use a close large challenger
    # Outside seat radius but still in central play zone
    challenger = _face(0.85, 0.45, width=0.24)
    # Still within seat empty window — no takeover
    early = lock.update([challenger], [], now=2.0)
    assert early.lock_id == 1
    assert early.status in ("locked", "challenger")

    # Seat empty long enough; challenger starts hold
    lock.update([challenger], [], now=4.2)
    # Hold completes
    taken = lock.update([challenger], [], now=5.3)
    assert taken.locked is True
    assert taken.lock_id == 2
    assert taken.flush_lstm is True


def test_tiny_background_face_not_eligible():
    lock = PlayerLock()
    tiny = _face(0.5, 0.45, width=0.03)
    result = lock.update([tiny], [], now=0.0)
    assert result.locked is False
    assert result.status == "unlocked"
