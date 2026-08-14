"""Launch flags for the live server.

Dev runs keep the OpenCV window (SHOW_PREVIEW). The Windows player build is
frozen by PyInstaller; that path defaults to headless so players never see a
desktop webcam or a console. Unity still gets the calibrate-panel JPEG stream.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vision_server.config import SHOW_PREVIEW


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def prepare_frozen_cwd() -> None:
    """Make relative paths like models/escape_gestures.keras work when frozen."""
    if is_frozen():
        os.chdir(Path(sys.executable).resolve().parent)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hand/face tracking server for Gaming with Bare Hands."
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "No OpenCV window and no keyboard shortcuts. "
            "Default: on when frozen (player build), otherwise SHOW_PREVIEW."
        ),
    )
    return parser.parse_args(argv)


def resolve_show_preview(args: argparse.Namespace) -> bool:
    """True means the debug HighGUI window. Independent of Unity calibrate preview."""
    if args.headless is True:
        return False
    if args.headless is False:
        return True
    if is_frozen():
        return False
    return bool(SHOW_PREVIEW)
