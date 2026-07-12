#!/usr/bin/env python3
"""Export an annotated Layer B replay video (HUD + GT labels).

Usage (from python-vision-server/):
  python eval/export_annotated_video.py eval/videos/test_video_1.mp4 --version v1

Writes: eval/exports/v1/test_video_1_annotated.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from vision_server.config import MODEL_PATH
from vision_server.evaluation.export_annotated import export_annotated_video
from vision_server.evaluation.segment_scoring import default_labels_path_for_video

EXPORTS_ROOT = _ROOT / "eval" / "exports"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export annotated test-video replay for visual label checking"
    )
    parser.add_argument("video", help="Path to test video")
    parser.add_argument(
        "--version",
        default="v1",
        help="Eval version folder under eval/exports/ (default: v1)",
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .keras model")
    parser.add_argument(
        "--labels",
        default=None,
        help="Segment labels JSON (default: <video_stem>_labels.json if present)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output path (default: eval/exports/<version>/<stem>_annotated.mp4)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not selfie-flip frames (default matches live server)",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Skip drawing MediaPipe hand skeletons",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            EXPORTS_ROOT / args.version / f"{video_path.stem}_annotated.mp4"
        )

    labels_path = (
        Path(args.labels) if args.labels else default_labels_path_for_video(video_path)
    )

    print(f"Video:   {video_path}")
    print(f"Labels:  {labels_path if labels_path else '(none)'}")
    print(f"Output:  {output_path}")
    print("Exporting… (this can take several minutes)")

    written = export_annotated_video(
        video_path,
        output_path,
        labels_path=labels_path,
        model_path=args.model,
        mirror=not args.no_mirror,
        draw_landmarks=not args.no_landmarks,
    )
    print(f"Done: {written}")


if __name__ == "__main__":
    main()
