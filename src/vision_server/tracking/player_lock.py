"""Lock onto one player's face/hands via reserved seat + auto challenger."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from vision_server.config import (
    CHALLENGER_HOLD_S,
    CHALLENGER_MAX_CENTER_DIST,
    CHALLENGER_MIN_FACE_WIDTH,
    CHALLENGER_SIZE_REF,
    FACE_MATCH_GATE,
    HAND_MATCH_GATE,
    HAND_RESEED_GATE,
    HAND_RESEED_S,
    HAND_SIZE_REF_RATIO,
    HAND_SIZE_SCORE_WEIGHT,
    HAND_TO_FACE_REACH,
    LOCK_CONFIRM_S,
    SEAT_EMPTY_S,
    SEAT_RADIUS,
)

_IMAGE_CENTER = (0.5, 0.45)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _face_metrics(face_landmarks) -> tuple[tuple[float, float], float, float]:
    lm = face_landmarks.landmark
    left, right = lm[234], lm[454]
    top, bottom = lm[10], lm[152]
    nose = lm[1]
    width = max(abs(right.x - left.x), 1e-6)
    height = max(abs(bottom.y - top.y), 1e-6)
    return (nose.x, nose.y), width, height


def _hand_center(landmarks) -> tuple[float, float]:
    xs = [landmarks[i].x for i in (0, 5, 9, 13, 17)]
    ys = [landmarks[i].y for i in (0, 5, 9, 13, 17)]
    return sum(xs) / 5, sum(ys) / 5


def _hand_length(landmarks) -> float:
    wrist, tip = landmarks[0], landmarks[12]
    return max(math.hypot(tip.x - wrist.x, tip.y - wrist.y), 1e-6)


@dataclass
class FaceCandidate:
    landmarks: object
    center: tuple[float, float]
    width: float
    height: float


@dataclass
class HandCandidate:
    landmarks: object
    mp_landmarks: object
    handedness: str
    world_landmarks: object | None
    center: tuple[float, float]
    length: float


@dataclass
class LockResult:
    locked: bool
    lock_id: int
    face: object | None
    left: HandCandidate | None
    right: HandCandidate | None
    flush_lstm: bool
    status: str  # unlocked | locked | challenger
    ring_center: tuple[float, float] | None
    ring_size: tuple[float, float]  # (width, height), normalized 0-1
    progress: float  # 0..1 for ring fill


def collect_faces(face_results) -> list[FaceCandidate]:
    if not face_results.multi_face_landmarks:
        return []
    faces = []
    for face in face_results.multi_face_landmarks:
        center, width, height = _face_metrics(face)
        faces.append(
            FaceCandidate(
                landmarks=face, center=center, width=width, height=height
            )
        )
    return faces


def collect_hands(hand_results) -> list[HandCandidate]:
    if not (
        hand_results.multi_hand_landmarks and hand_results.multi_handedness
    ):
        return []

    world_list = hand_results.multi_hand_world_landmarks or []
    hands = []
    for index, (mp_hand, handedness) in enumerate(
        zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)
    ):
        landmarks = mp_hand.landmark
        world = world_list[index] if index < len(world_list) else None
        hands.append(
            HandCandidate(
                landmarks=landmarks,
                mp_landmarks=mp_hand,
                handedness=handedness.classification[0].label,
                world_landmarks=world,
                center=_hand_center(landmarks),
                length=_hand_length(landmarks),
            )
        )
    return hands


def _log(msg: str) -> None:
    print(f"[lock] {msg}")


def _challenger_eligible(face: FaceCandidate) -> bool:
    if face.width < CHALLENGER_MIN_FACE_WIDTH:
        return False
    return _dist(face.center, _IMAGE_CENTER) <= CHALLENGER_MAX_CENTER_DIST


def _challenger_score(face: FaceCandidate) -> float:
    center_dist = _dist(face.center, _IMAGE_CENTER)
    centeredness = 1.0 - min(1.0, center_dist / CHALLENGER_MAX_CENTER_DIST)
    size = min(1.0, face.width / CHALLENGER_SIZE_REF)
    return centeredness + size


class PlayerLock:
    def __init__(self):
        self.lock_id = 0
        self.locked = False
        self._face_center: tuple[float, float] | None = None
        self._face_width = 0.1
        self._face_height = 0.1
        self._left_center: tuple[float, float] | None = None
        self._right_center: tuple[float, float] | None = None
        self._left_length: float | None = None
        self._right_length: float | None = None
        self._left_missing_since: float | None = None
        self._right_missing_since: float | None = None
        self._seat_center: tuple[float, float] | None = None
        self._seat_width = 0.1
        self._seat_empty_since: float | None = None
        self._cold_face: FaceCandidate | None = None
        self._cold_since: float | None = None
        self._challenger: FaceCandidate | None = None
        self._challenger_since: float | None = None

    def update(
        self,
        faces: list[FaceCandidate],
        hands: list[HandCandidate],
        now: float | None = None,
    ) -> LockResult:
        now = time.monotonic() if now is None else now

        if not self.locked:
            return self._cold_start(faces, hands, now)

        in_seat = self._face_in_seat(faces)
        if in_seat is not None:
            return self._resume(in_seat, hands, now)

        # While seat is reserved, only seat-resume or challenger — not track-gate
        if self._seat_empty_since is None:
            tracked = self._match_tracked_face(faces)
            if tracked is not None:
                self._challenger = None
                self._challenger_since = None
                self._set_face(tracked)
                left, right, flush = self._match_hands(hands, now)
                return LockResult(
                    locked=True,
                    lock_id=self.lock_id,
                    face=tracked.landmarks,
                    left=left,
                    right=right,
                    flush_lstm=flush,
                    status="locked",
                    ring_center=tracked.center,
                    ring_size=(tracked.width, tracked.height),
                    progress=1.0,
                )

            # Just lost the face — reserve seat
            self._seat_empty_since = now
            _log("seat reserved")

        challenger = self._update_challenger(faces, now)
        if challenger is not None:
            return self._takeover(challenger, hands)

        left, right, flush = self._match_hands(hands, now)
        progress = 0.0
        status = "locked"
        ring = self._seat_center
        ring_size = (self._seat_width, self._face_height)
        if (
            self._challenger is not None
            and self._challenger_since is not None
            and now - self._seat_empty_since >= SEAT_EMPTY_S
        ):
            status = "challenger"
            ring = self._challenger.center
            ring_size = (self._challenger.width, self._challenger.height)
            progress = min(
                1.0, (now - self._challenger_since) / CHALLENGER_HOLD_S
            )

        return LockResult(
            locked=True,
            lock_id=self.lock_id,
            face=None,
            left=left,
            right=right,
            flush_lstm=flush,
            status=status,
            ring_center=ring,
            ring_size=ring_size,
            progress=progress,
        )

    def _set_face(self, face: FaceCandidate) -> None:
        self._face_center = face.center
        self._face_width = face.width
        self._face_height = face.height
        self._seat_center = face.center
        self._seat_width = face.width

    def _face_in_seat(
        self, faces: list[FaceCandidate]
    ) -> FaceCandidate | None:
        if self._seat_center is None:
            return None
        scale = max(self._seat_width, 1e-6)
        best = None
        best_cost = float("inf")
        for face in faces:
            cost = _dist(face.center, self._seat_center) / scale
            if cost < best_cost:
                best_cost = cost
                best = face
        if best is None or best_cost > SEAT_RADIUS:
            return None
        return best

    def _match_tracked_face(
        self, faces: list[FaceCandidate]
    ) -> FaceCandidate | None:
        if not faces or self._face_center is None:
            return None
        scale = max(self._face_width, 1e-6)
        best = min(
            faces,
            key=lambda f: _dist(f.center, self._face_center) / scale,
        )
        if _dist(best.center, self._face_center) / scale > FACE_MATCH_GATE:
            return None
        return best

    def _resume(
        self,
        face: FaceCandidate,
        hands: list[HandCandidate],
        now: float,
    ) -> LockResult:
        if self._seat_empty_since is not None:
            _log("seat resume")
        self._seat_empty_since = None
        self._challenger = None
        self._challenger_since = None
        self._set_face(face)
        left, right, flush = self._match_hands(hands, now)
        return LockResult(
            locked=True,
            lock_id=self.lock_id,
            face=face.landmarks,
            left=left,
            right=right,
            flush_lstm=flush,
            status="locked",
            ring_center=face.center,
            ring_size=(face.width, face.height),
            progress=1.0,
        )

    def _update_challenger(
        self, faces: list[FaceCandidate], now: float
    ) -> FaceCandidate | None:
        if (
            self._seat_empty_since is None
            or now - self._seat_empty_since < SEAT_EMPTY_S
        ):
            self._challenger = None
            self._challenger_since = None
            return None

        outside = []
        for face in faces:
            if not _challenger_eligible(face):
                continue
            if self._seat_center is not None:
                cost = _dist(face.center, self._seat_center) / max(
                    self._seat_width, 1e-6
                )
                if cost <= SEAT_RADIUS:
                    continue
            outside.append(face)

        if not outside:
            self._challenger = None
            self._challenger_since = None
            return None

        best = max(outside, key=_challenger_score)

        if self._challenger is None:
            self._challenger = best
            self._challenger_since = now
            _log(
                f"challenger start score={_challenger_score(best):.2f}"
            )
            return None

        # Same person if still near previous challenger center
        same = (
            _dist(best.center, self._challenger.center)
            / max(best.width, 1e-6)
            <= FACE_MATCH_GATE
        )
        if not same or _challenger_score(best) + 0.05 < _challenger_score(
            self._challenger
        ):
            self._challenger = best
            self._challenger_since = now
            _log("challenger reset")
            return None

        self._challenger = best
        if (
            self._challenger_since is None
            or now - self._challenger_since < CHALLENGER_HOLD_S
        ):
            return None

        return best

    def _takeover(
        self, face: FaceCandidate, hands: list[HandCandidate]
    ) -> LockResult:
        self.lock_id += 1
        self.locked = True
        self._reset_hand_tracks()
        self._seat_empty_since = None
        self._challenger = None
        self._challenger_since = None
        self._set_face(face)
        left, right = self._seed_hands(hands)
        self._commit_seeded_hands(left, right)
        _log(f"challenger takeover id={self.lock_id}")
        return LockResult(
            locked=True,
            lock_id=self.lock_id,
            face=face.landmarks,
            left=left,
            right=right,
            flush_lstm=True,
            status="locked",
            ring_center=face.center,
            ring_size=(face.width, face.height),
            progress=1.0,
        )

    def _cold_start(
        self,
        faces: list[FaceCandidate],
        hands: list[HandCandidate],
        now: float,
    ) -> LockResult:
        eligible = [f for f in faces if _challenger_eligible(f)]
        if not eligible:
            self._cold_face = None
            self._cold_since = None
            return LockResult(
                locked=False,
                lock_id=self.lock_id,
                face=None,
                left=None,
                right=None,
                flush_lstm=False,
                status="unlocked",
                ring_center=None,
                ring_size=(0.15, 0.2),
                progress=0.0,
            )

        best = max(eligible, key=_challenger_score)

        if self._cold_face is None:
            self._cold_face = best
            self._cold_since = now
            _log("cold start confirm")
            return LockResult(
                locked=False,
                lock_id=self.lock_id,
                face=None,
                left=None,
                right=None,
                flush_lstm=False,
                status="challenger",
                ring_center=best.center,
                ring_size=(best.width, best.height),
                progress=0.0,
            )

        same = (
            _dist(best.center, self._cold_face.center)
            / max(best.width, 1e-6)
            <= FACE_MATCH_GATE
        )
        if not same:
            self._cold_face = best
            self._cold_since = now
            _log("cold start reset")
            return LockResult(
                locked=False,
                lock_id=self.lock_id,
                face=None,
                left=None,
                right=None,
                flush_lstm=False,
                status="challenger",
                ring_center=best.center,
                ring_size=(best.width, best.height),
                progress=0.0,
            )

        self._cold_face = best
        elapsed = now - (self._cold_since if self._cold_since is not None else now)
        progress = min(1.0, elapsed / LOCK_CONFIRM_S)
        if elapsed < LOCK_CONFIRM_S:
            return LockResult(
                locked=False,
                lock_id=self.lock_id,
                face=None,
                left=None,
                right=None,
                flush_lstm=False,
                status="challenger",
                ring_center=best.center,
                ring_size=(best.width, best.height),
                progress=progress,
            )

        self.locked = True
        self.lock_id += 1
        self._cold_face = None
        self._cold_since = None
        self._set_face(best)
        left, right = self._seed_hands(hands)
        self._commit_seeded_hands(left, right)
        _log(f"acquired id={self.lock_id}")
        return LockResult(
            locked=True,
            lock_id=self.lock_id,
            face=best.landmarks,
            left=left,
            right=right,
            flush_lstm=True,
            status="locked",
            ring_center=best.center,
            ring_size=(best.width, best.height),
            progress=1.0,
        )

    def _hand_ok(self, hand: HandCandidate) -> bool:
        # Deliberately silent: this runs per hand per frame, and a bystander
        # standing in shot made it a continuous print() inside the capture loop.
        if self._face_center is None:
            return False
        scale = max(self._face_width, 1e-6)
        return _dist(hand.center, self._face_center) / scale <= HAND_TO_FACE_REACH

    def _reset_hand_tracks(self) -> None:
        self._left_center = None
        self._right_center = None
        self._left_length = None
        self._right_length = None
        self._left_missing_since = None
        self._right_missing_since = None

    def _commit_seeded_hands(
        self,
        left: HandCandidate | None,
        right: HandCandidate | None,
    ) -> None:
        self._left_center = left.center if left else None
        self._right_center = right.center if right else None
        self._left_length = left.length if left else None
        self._right_length = right.length if right else None
        self._left_missing_since = None
        self._right_missing_since = None

    def _expected_hand_length(self) -> float:
        return max(self._face_height * HAND_SIZE_REF_RATIO, 1e-6)

    def _seed_hands(
        self, hands: list[HandCandidate]
    ) -> tuple[HandCandidate | None, HandCandidate | None]:
        ok = [h for h in hands if self._hand_ok(h)]
        face = self._face_center or (0.5, 0.5)
        scale = max(self._face_width, 1e-6)
        expected = self._expected_hand_length()
        left = self._pick_side(
            ok, face, scale, "left", gate=None, preferred_length=expected
        )
        right = self._pick_side(
            ok, face, scale, "right", gate=None, preferred_length=expected
        )
        return left, right

    def _match_hands(
        self, hands: list[HandCandidate], now: float
    ) -> tuple[HandCandidate | None, HandCandidate | None, bool]:
        # Seeding filters through _hand_ok itself, so do not pay for it twice.
        if self._left_center is None and self._right_center is None:
            left, right = self._seed_hands(hands)
            self._commit_seeded_hands(left, right)
            return left, right, False

        ok = [h for h in hands if self._hand_ok(h)]
        face = self._face_center or (0.5, 0.5)
        scale = max(self._face_width, 1e-6)
        expected = self._expected_hand_length()
        flush = False

        left = right = None

        if self._left_center is not None:
            left = self._pick_side(
                ok,
                self._left_center,
                scale,
                "left",
                gate=HAND_MATCH_GATE,
                preferred_length=self._left_length or expected,
            )
        if self._right_center is not None:
            right = self._pick_side(
                ok,
                self._right_center,
                scale,
                "right",
                gate=HAND_MATCH_GATE,
                preferred_length=self._right_length or expected,
            )

        if self._left_center is not None:
            left, flush_l = self._hold_or_reseed_side(
                side="left",
                matched=left,
                candidates=ok,
                scale=scale,
                now=now,
            )
            flush = flush or flush_l
        if self._right_center is not None:
            right, flush_r = self._hold_or_reseed_side(
                side="right",
                matched=right,
                candidates=ok,
                scale=scale,
                now=now,
            )
            flush = flush or flush_r

        # Empty / abandoned sides: soft-acquire near the locked face.
        if self._left_center is None and left is None:
            left = self._pick_side(
                ok, face, scale, "left", gate=None, preferred_length=expected
            )
        if self._right_center is None and right is None:
            right = self._pick_side(
                ok, face, scale, "right", gate=None, preferred_length=expected
            )

        if left is not None:
            self._left_center = left.center
            self._left_length = left.length
            self._left_missing_since = None
        if right is not None:
            self._right_center = right.center
            self._right_length = right.length
            self._right_missing_since = None

        return left, right, flush

    def _clear_side_track(self, side: str) -> None:
        if side == "left":
            self._left_center = None
            self._left_length = None
            self._left_missing_since = None
        else:
            self._right_center = None
            self._right_length = None
            self._right_missing_since = None

    def _hold_or_reseed_side(
        self,
        *,
        side: str,
        matched: HandCandidate | None,
        candidates: list[HandCandidate],
        scale: float,
        now: float,
    ) -> tuple[HandCandidate | None, bool]:
        if side == "left":
            track = self._left_center
            missing_attr = "_left_missing_since"
            preferred = self._left_length or self._expected_hand_length()
        else:
            track = self._right_center
            missing_attr = "_right_missing_since"
            preferred = self._right_length or self._expected_hand_length()

        if track is None:
            return matched, False

        if matched is not None:
            setattr(self, missing_attr, None)
            return matched, False

        missing_since = getattr(self, missing_attr)
        if missing_since is None:
            setattr(self, missing_attr, now)
            return None, False

        if now - missing_since < HAND_RESEED_S:
            # Hold last position; do not steal a hand nearer the face.
            return None, False

        # Reseed only near the previous hand location — never "nearest to face".
        reseed = self._pick_side(
            candidates,
            track,
            scale,
            side,
            gate=HAND_RESEED_GATE,
            preferred_length=preferred,
        )
        if reseed is not None:
            _log(f"{side} hand reseed near previous")
            return reseed, True

        # Bystander left / wrong lock: drop stale track so face-near
        # soft-acquire can pick up the locked player's hands.
        _log(f"{side} hand track abandoned")
        self._clear_side_track(side)
        return None, True

    @staticmethod
    def _pick_side(
        hands: list[HandCandidate],
        target: tuple[float, float],
        scale: float,
        side: str,
        *,
        gate: float | None,
        preferred_length: float | None,
    ) -> HandCandidate | None:
        pool = [h for h in hands if h.handedness.lower() == side]
        best = None
        best_score = float("inf")
        for hand in pool:
            dist_cost = _dist(hand.center, target) / scale
            if gate is not None and dist_cost > gate:
                continue
            size_cost = 0.0
            if preferred_length is not None and preferred_length > 0:
                size_cost = abs(hand.length - preferred_length) / preferred_length
            score = dist_cost + HAND_SIZE_SCORE_WEIGHT * size_cost
            if score < best_score:
                best_score = score
                best = hand
        return best
