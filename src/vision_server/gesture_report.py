"""Session diagnostics for the in-game gesture coach.

Pain rows (Python): jump, crouch, inventory, watch_tap, pull_lever.
Grab and click are counted in Unity.

Intent is unknowable until something commits. When it does, look back
``GESTURE_REPORT_LOOKBACK_S`` and treat failed flashes in that window as
tries. Idle players never commit, so they never enter the data.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from vision_server.config import (
    GESTURE_REPORT_DARK_MEAN,
    GESTURE_REPORT_JITTER,
    GESTURE_REPORT_LOOKBACK_S,
    GESTURE_REPORT_MIN_PALM,
    GESTURE_REPORT_ROWS,
    TCP_REPORT_CONNECT_S,
    TCP_REPORT_IP,
    TCP_REPORT_PORT,
)
from vision_server.gestures.hand.fingers import hand_frame

ISSUE_HAND_OUT_OF_FRAME = "HAND_OUT_OF_FRAME"
ISSUE_HAND_TOO_SMALL = "HAND_TOO_SMALL"
ISSUE_LANDMARK_JITTER = "LANDMARK_JITTER"
ISSUE_POOR_LIGHTING = "POOR_LIGHTING"
ISSUE_DWELL_TOO_SHORT = "DWELL_TOO_SHORT"

# MOVE / ACTION classifier label -> pain row.
_MOVE_ROWS = {"index_up": "jump", "peace": "crouch", "thumbs_up": "inventory"}
_ACTION_ROWS = {"fist": "grab"}
_LEVER = "Pull_Lever"

HEALTH_RETRY_WARN = 0.8
HEALTH_RETRY_BAD = 1.5
CAMERA_SHARE = 0.20


def _palm_length(landmarks) -> float | None:
    if landmarks is None:
        return None
    frame = hand_frame(landmarks)
    if frame is None:
        return None
    return frame[1]


def _xy(landmarks) -> list[tuple[float, float]] | None:
    if landmarks is None:
        return None
    try:
        return [(float(lm.x), float(lm.y)) for lm in landmarks]
    except (TypeError, AttributeError):
        return None


def _jitter(
    prev_xy: list[tuple[float, float]] | None,
    xy: list[tuple[float, float]] | None,
    palm: float | None,
) -> float | None:
    if prev_xy is None or xy is None or palm is None or palm <= 1e-6:
        return None
    if len(prev_xy) != len(xy):
        return None
    total = 0.0
    for (ax, ay), (bx, by) in zip(prev_xy, xy):
        total += math.hypot(bx - ax, by - ay)
    return (total / len(xy)) / palm


def _hand_luma(frame, move_xy, action_xy) -> float | None:
    """Mean Rec. 601 luma of the hand crop, else the whole frame."""
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
    except AttributeError:
        return None
    if h < 2 or w < 2:
        return None

    pts: list[tuple[float, float]] = []
    if move_xy:
        pts.extend(move_xy)
    if action_xy:
        pts.extend(action_xy)

    if pts:
        pad = 0.08
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0, int((min(xs) - pad) * w))
        x1 = min(w, int((max(xs) + pad) * w))
        y0 = max(0, int((min(ys) - pad) * h))
        y1 = min(h, int((max(ys) + pad) * h))
        crop = frame[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else frame
    else:
        crop = frame

    sample = crop[::4, ::4]
    if sample.size == 0:
        return None
    b = sample[:, :, 0].astype("float32")
    g = sample[:, :, 1].astype("float32")
    r = sample[:, :, 2].astype("float32")
    return float((0.114 * b + 0.587 * g + 0.299 * r).mean())


def _health_from_retries(retries: float, successes: int) -> str:
    if successes <= 0:
        return "ok"
    if retries >= HEALTH_RETRY_BAD:
        return "bad"
    if retries >= HEALTH_RETRY_WARN:
        return "warn"
    return "ok"


@dataclass
class _FrameObs:
    t: float
    move_seen: bool
    action_seen: bool
    palm: float | None
    jitter: float | None
    move_raw: str
    move_committed: str
    move_candidate: str
    action_raw: str
    action_committed: str
    action_candidate: str
    watch_raw: bool
    watch_committed: bool
    lstm: str
    puzzle: bool


@dataclass
class GestureStats:
    row_id: str
    attempts: int = 0
    successes: int = 0
    trigger_ms_sum: float = 0.0
    retry_sum: float = 0.0
    issue_counts: dict[str, int] = field(default_factory=dict)

    def note_issue(self, code: str) -> None:
        if not code:
            return
        self.issue_counts[code] = self.issue_counts.get(code, 0) + 1

    def dominant_issue(self) -> str:
        if not self.issue_counts:
            return ""
        return max(self.issue_counts.items(), key=lambda kv: kv[1])[0]

    def retries(self) -> float:
        if not self.successes:
            return 0.0
        return round(self.retry_sum / self.successes, 1)

    def as_row(self) -> dict:
        retries = self.retries()
        return {
            "id": self.row_id,
            "attempts": self.attempts,
            "successes": self.successes,
            "avg_trigger_ms": (
                int(round(self.trigger_ms_sum / self.successes))
                if self.successes
                else 0
            ),
            "retries": retries,
            "issue": self.dominant_issue(),
            "health": _health_from_retries(retries, self.successes),
        }


class GestureDiagnostics:
    def __init__(self, lookback_s: float = GESTURE_REPORT_LOOKBACK_S):
        self.lookback_s = lookback_s
        self._frames: deque[_FrameObs] = deque()
        self._prev_move_xy: list[tuple[float, float]] | None = None
        self._prev_action_xy: list[tuple[float, float]] | None = None
        self._prev_move = "none"
        self._prev_action = "none"
        self._prev_watch = False
        self._prev_lstm = "Idle"
        self._chamber = 0
        self.stats: dict[str, GestureStats] = {
            name: GestureStats(row_id=name) for name in GESTURE_REPORT_ROWS
        }
        self.cam_frames = 0
        self.cam_out = 0
        self.cam_small = 0
        self.cam_jitter = 0
        self.cam_dark = 0
        self.track_confident = False

    def reset(self) -> None:
        self._frames.clear()
        self._prev_move_xy = None
        self._prev_action_xy = None
        self._prev_move = "none"
        self._prev_action = "none"
        self._prev_watch = False
        self._prev_lstm = "Idle"
        self.stats = {name: GestureStats(row_id=name) for name in GESTURE_REPORT_ROWS}
        self.cam_frames = 0
        self.cam_out = 0
        self.cam_small = 0
        self.cam_jitter = 0
        self.cam_dark = 0
        self.track_confident = False

    def set_chamber(self, chamber: int | None) -> None:
        if chamber is None:
            return
        self._chamber = int(chamber)

    def update(
        self,
        *,
        move_landmarks,
        action_landmarks,
        move_raw: str,
        move_committed: str,
        move_candidate: str,
        action_raw: str,
        action_committed: str,
        action_candidate: str,
        watch_raw: bool,
        watch_committed: bool,
        lstm: str,
        puzzle: bool,
        frame=None,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        move_xy = _xy(move_landmarks)
        action_xy = _xy(action_landmarks)
        # Camera chips use whichever hand is larger / present.
        palms = [_palm_length(move_landmarks), _palm_length(action_landmarks)]
        palm = max((p for p in palms if p is not None), default=None)
        seen = move_xy is not None or action_xy is not None
        jitter_m = _jitter(self._prev_move_xy, move_xy, _palm_length(move_landmarks))
        jitter_a = _jitter(
            self._prev_action_xy, action_xy, _palm_length(action_landmarks)
        )
        jitter_vals = [j for j in (jitter_m, jitter_a) if j is not None]
        jitter = max(jitter_vals) if jitter_vals else None
        self._prev_move_xy = move_xy
        self._prev_action_xy = action_xy

        too_small = seen and (palm is None or palm < GESTURE_REPORT_MIN_PALM)
        shaky = jitter is not None and jitter >= GESTURE_REPORT_JITTER
        luma = _hand_luma(frame, move_xy, action_xy)
        dark = luma is not None and luma < GESTURE_REPORT_DARK_MEAN
        self.track_confident = seen and not too_small and not shaky

        self.cam_frames += 1
        if not seen:
            self.cam_out += 1
        if too_small:
            self.cam_small += 1
        if shaky:
            self.cam_jitter += 1
        if dark:
            self.cam_dark += 1

        obs = _FrameObs(
            t=now,
            move_seen=move_xy is not None,
            action_seen=action_xy is not None,
            palm=palm,
            jitter=jitter,
            move_raw=move_raw,
            move_committed=move_committed,
            move_candidate=move_candidate,
            action_raw=action_raw,
            action_committed=action_committed,
            action_candidate=action_candidate,
            watch_raw=bool(watch_raw),
            watch_committed=bool(watch_committed),
            lstm=lstm or "Idle",
            puzzle=bool(puzzle),
        )
        self._frames.append(obs)
        cutoff = now - self.lookback_s
        while self._frames and self._frames[0].t < cutoff:
            self._frames.popleft()

        for label, row in _MOVE_ROWS.items():
            if move_committed == label and self._prev_move != label:
                self._label_success(row, now, kind="move", label=label)
        self._prev_move = move_committed

        # Grab is Unity-only (holding a world object). ACTION fist is also
        # the pause-menu click, so counting it here as grab poisons the row.
        self._prev_action = action_committed

        if watch_committed and not self._prev_watch:
            self._label_success("watch_tap", now, kind="watch", label="")
        self._prev_watch = watch_committed

        if (
            puzzle
            and lstm == _LEVER
            and self._prev_lstm != _LEVER
        ):
            self._label_success("pull_lever", now, kind="lever", label=_LEVER)
        self._prev_lstm = lstm

    def _window(self, now: float) -> list[_FrameObs]:
        cutoff = now - self.lookback_s
        return [f for f in self._frames if f.t >= cutoff]

    def _label_success(self, row_id: str, now: float, *, kind: str, label: str) -> None:
        window = self._window(now)
        row = self.stats[row_id]
        fail_runs = _failed_runs(window, kind, label)
        row.attempts += fail_runs + 1
        row.successes += 1
        row.retry_sum += fail_runs
        first = _first_try_time(window, kind, label)
        if first is None:
            first = window[0].t if window else now
        row.trigger_ms_sum += max(0.0, (now - first) * 1000.0)
        row.note_issue(_diagnose_window(window, kind, label))

    def camera_strip(self) -> dict:
        n = max(1, self.cam_frames)
        distance = "far" if self.cam_small / n >= CAMERA_SHARE else "ok"
        lighting = "bad" if self.cam_dark / n >= CAMERA_SHARE else "ok"
        in_frame = "out" if self.cam_out / n >= CAMERA_SHARE else "ok"
        return {
            "distance": distance,
            "lighting": lighting,
            "in_frame": in_frame,
        }

    def build_report(self, chamber: int | None = None) -> dict:
        if chamber is not None:
            self._chamber = chamber
        cam = self.camera_strip()
        rows = [self.stats[name].as_row() for name in GESTURE_REPORT_ROWS]
        advice_issue, advice_gesture = _advice(cam, rows)
        return {
            "type": "gesture_report",
            "chamber": self._chamber,
            "camera": cam,
            "advice_issue": advice_issue,
            "advice_gesture": advice_gesture,
            "gestures": rows,
        }


def _is_try(obs: _FrameObs, kind: str, label: str) -> bool:
    if kind == "move":
        return obs.move_raw == label or obs.move_candidate == label
    if kind == "action":
        return obs.action_raw == label or obs.action_candidate == label
    if kind == "watch":
        return obs.watch_raw
    if kind == "lever":
        return obs.puzzle and obs.action_seen and obs.lstm != "Idle"
    return False


def _is_committed(obs: _FrameObs, kind: str, label: str) -> bool:
    if kind == "move":
        return obs.move_committed == label
    if kind == "action":
        return obs.action_committed == label
    if kind == "watch":
        return obs.watch_committed
    if kind == "lever":
        return obs.lstm == _LEVER
    return False


def _first_try_time(window: list[_FrameObs], kind: str, label: str) -> float | None:
    for obs in window:
        if _is_try(obs, kind, label):
            return obs.t
    return None


def _failed_runs(window: list[_FrameObs], kind: str, label: str) -> int:
    runs = 0
    in_run = False
    committed_in_run = False
    for obs in window:
        trying = _is_try(obs, kind, label)
        if trying:
            in_run = True
            if _is_committed(obs, kind, label):
                committed_in_run = True
        elif in_run:
            if not committed_in_run:
                runs += 1
            in_run = False
            committed_in_run = False
    return runs


def _diagnose_window(window: list[_FrameObs], kind: str, label: str) -> str:
    if not window:
        return ""
    n = len(window)
    if kind == "move":
        out = sum(1 for f in window if not f.move_seen)
        dwell = sum(
            1
            for f in window
            if f.move_candidate == label and f.move_committed != label
        )
    elif kind == "action":
        out = sum(1 for f in window if not f.action_seen)
        dwell = sum(
            1
            for f in window
            if f.action_candidate == label and f.action_committed != label
        )
    elif kind == "watch":
        out = sum(1 for f in window if not (f.move_seen and f.action_seen))
        dwell = sum(1 for f in window if f.watch_raw and not f.watch_committed)
    else:
        out = sum(1 for f in window if not f.action_seen)
        dwell = sum(
            1
            for f in window
            if f.puzzle and f.action_seen and f.lstm != _LEVER
        )
    small = sum(
        1
        for f in window
        if (f.move_seen or f.action_seen)
        and (f.palm is None or f.palm < GESTURE_REPORT_MIN_PALM)
    )
    jitter = sum(
        1
        for f in window
        if f.jitter is not None and f.jitter >= GESTURE_REPORT_JITTER
    )
    floor = max(1, int(n * 0.15))
    ranked = [
        (out, ISSUE_HAND_OUT_OF_FRAME),
        (small, ISSUE_HAND_TOO_SMALL),
        (jitter, ISSUE_LANDMARK_JITTER),
        (dwell, ISSUE_DWELL_TOO_SHORT),
    ]
    ranked.sort(key=lambda x: x[0], reverse=True)
    count, code = ranked[0]
    if count < floor:
        return ""
    return code


def _advice(cam: dict, rows: list[dict]) -> tuple[str, str]:
    if cam["in_frame"] == "out":
        return ISSUE_HAND_OUT_OF_FRAME, ""
    if cam["distance"] == "far":
        return ISSUE_HAND_TOO_SMALL, ""
    if cam["lighting"] == "bad":
        return ISSUE_POOR_LIGHTING, ""
    worst = None
    for row in rows:
        if row["health"] == "ok":
            continue
        rank = 2 if row["health"] == "bad" else 1
        score = (rank, row["retries"])
        if worst is None or score > worst[0]:
            worst = (score, row)
    if worst is None:
        return "", ""
    row = worst[1]
    return row.get("issue") or ISSUE_DWELL_TOO_SHORT, row["id"]


def send_report_tcp(
    payload: dict,
    *,
    host: str = TCP_REPORT_IP,
    port: int = TCP_REPORT_PORT,
    timeout: float = TCP_REPORT_CONNECT_S,
) -> None:
    body = (json.dumps(payload) + "\n").encode("utf-8")

    def _run() -> None:
        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.sendall(body)
        except OSError:
            return

    threading.Thread(target=_run, daemon=True).start()
