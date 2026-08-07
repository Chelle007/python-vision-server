"""Rolling frame-time percentiles for the capture loop.

Accumulates per-stage timings and flushes a summary on an interval, so a run
always produces a comparable line rather than only reporting bad frames.

Median says what the loop normally costs; p95 and max say how bad the stalls
get. Contention shows up as a large max against a healthy median, whereas a
genuinely expensive pipeline lifts the median too — the same read as the
per-stage rule in :mod:`vision_server.runtime`, applied over time instead of
within one frame.

``nhands`` is tracked alongside because hand cost scales with the number of
hands actually in frame: measured at ~18ms for none and ~60ms for two, and
independent of ``MAX_NUM_HANDS``.
"""

from __future__ import annotations

import time

from vision_server.config import FRAME_STATS_INTERVAL_S

# Stage order as printed; must match the keyword arguments to `record`.
_STAGES = ("total", "prep", "hands", "face", "lstm", "logic", "show")


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not values:
        return 0.0
    index = int(round(fraction * (len(values) - 1)))
    return values[index]


class FrameStats:
    """Collects per-frame stage timings and prints a summary on an interval."""

    def __init__(self, interval_s: float = FRAME_STATS_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self._samples: dict[str, list[float]] = {s: [] for s in _STAGES}
        self._slow_frames = 0
        self._hands_n: list[int] = []
        self._window_start = time.perf_counter()

    def record(
        self,
        *,
        total_ms: float,
        prep_ms: float,
        hands_ms: float,
        face_ms: float,
        lstm_ms: float,
        logic_ms: float,
        show_ms: float,
        hands_n: int,
        slow: bool,
    ) -> None:
        if self.interval_s <= 0:
            return
        self._samples["total"].append(total_ms)
        self._samples["prep"].append(prep_ms)
        self._samples["hands"].append(hands_ms)
        self._samples["face"].append(face_ms)
        self._samples["lstm"].append(lstm_ms)
        self._samples["logic"].append(logic_ms)
        self._samples["show"].append(show_ms)
        self._hands_n.append(hands_n)
        if slow:
            self._slow_frames += 1

    def maybe_report(self) -> None:
        """Print and reset if the interval has elapsed. Cheap to call per frame."""
        if self.interval_s <= 0:
            return
        elapsed = time.perf_counter() - self._window_start
        if elapsed < self.interval_s:
            return

        frames = len(self._samples["total"])
        if frames == 0:
            self._reset()
            return

        for values in self._samples.values():
            values.sort()

        total = self._samples["total"]
        fps = frames / elapsed
        hands_avg = sum(self._hands_n) / len(self._hands_n)
        print(
            f"[perf] {elapsed:.0f}s frames={frames} fps={fps:.1f} "
            f"slow={self._slow_frames} "
            f"nhands avg={hands_avg:.1f}/max={max(self._hands_n)} | "
            f"total med={_percentile(total, 0.5):.0f} "
            f"p95={_percentile(total, 0.95):.0f} "
            f"max={total[-1]:.0f} | "
            + " ".join(
                f"{stage} med={_percentile(self._samples[stage], 0.5):.0f}"
                f"/max={self._samples[stage][-1]:.0f}"
                for stage in _STAGES
                if stage != "total"
            )
        )
        self._reset()

    def _reset(self) -> None:
        for values in self._samples.values():
            values.clear()
        self._slow_frames = 0
        self._hands_n.clear()
        self._window_start = time.perf_counter()
