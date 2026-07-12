#!/usr/bin/env python3
"""Layer B — replay a fixed test video for detection / recovery / LSTM timeline.

Usage (from python-vision-server/):
  python eval/run_layer_b.py eval/videos/test_video_1.mp4
  python eval/run_layer_b.py eval/videos/test_video_1.mp4 --no-lstm
  python eval/run_layer_b.py eval/videos/test_video_1.mp4 --labels eval/videos/test_video_1_labels.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from vision_server.config import MODEL_PATH
from vision_server.evaluation.segment_scoring import (
    default_labels_path_for_video,
    score_video_segments,
)
from vision_server.evaluation.video_metrics import evaluate_test_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer B fixed test-video evaluation")
    parser.add_argument("video", help="Path to test video (.mp4)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .keras model")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM inference")
    parser.add_argument("--no-jitter", action="store_true", help="Skip jitter pass")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not selfie-flip frames (default matches live server)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Segment labels JSON (default: <video_stem>_labels.json if present)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip segment scoring even if a labels file exists",
    )
    args = parser.parse_args()

    result = evaluate_test_video(
        args.video,
        run_lstm=not args.no_lstm,
        run_jitter=not args.no_jitter,
        model_path=args.model,
        mirror=not args.no_mirror,
    )
    print()
    print(result.format_report())
    print("--- Spreadsheet paste (Layer B auto) ---")
    recovery = (
        f"{result.mean_recovery_frames:.1f}"
        if result.mean_recovery_frames == result.mean_recovery_frames
        else "n/a"
    )
    jitter = (
        f"{result.jitter_score:.6f}"
        if result.jitter_score == result.jitter_score
        else "n/a"
    )
    print(
        f"detection={result.detection_rate:.4f}  "
        f"recovery_frames={recovery}  "
        f"jitter={jitter}"
    )

    labels_path = None
    if not args.no_labels:
        labels_path = (
            Path(args.labels)
            if args.labels
            else default_labels_path_for_video(args.video)
        )
    if labels_path is not None:
        seg = score_video_segments(
            args.video,
            labels_path,
            model_path=args.model,
            mirror=not args.no_mirror,
        )
        print()
        print(seg.format_report())
        print("--- Spreadsheet paste (Layer B labels) ---")
        print(
            f"lr={seg.lr_assignment_accuracy:.4f}  "
            f"static_incl_watchTap={seg.combined_static_including_watch_tap:.4f}  "
            f"lstm_Pull_Lever={seg.lstm_video_accuracy:.4f}  "
            f"false_trigger={seg.false_trigger_rate:.4f}"
        )


if __name__ == "__main__":
    main()
