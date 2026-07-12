#!/usr/bin/env python3
"""Layer B — landmark jitter score on a fixed test video.

Usage (from python-vision-server/):
  python eval/measure_jitter.py eval/videos/test_video_1.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from vision_server.evaluation.jitter import measure_jitter


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer B jitter measurement")
    parser.add_argument("video", help="Path to test video (.mp4)")
    parser.add_argument(
        "--still-threshold",
        type=float,
        default=None,
        help="Max normalized motion to count as still (default: package default)",
    )
    args = parser.parse_args()

    kwargs = {}
    if args.still_threshold is not None:
        kwargs["still_threshold"] = args.still_threshold

    result = measure_jitter(args.video, **kwargs)
    print()
    print(result.format_report())
    print("--- Spreadsheet paste (jitter) ---")
    print(f"jitter={result.overall_jitter:.6f}  detection={result.detection_rate:.4f}")


if __name__ == "__main__":
    main()
