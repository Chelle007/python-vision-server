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

from vision_server.config import CAMERA_INDEX, SHOW_PREVIEW

# Player-editable override for the webcam OpenCV index. Lives next to the
# game so a player whose working camera is not index 0 (very common: a virtual
# camera or a laptop's built-in cam grabs index 0) can point the server at the
# right device WITHOUT rebuilding Unity. Unity's WebCamTexture dropdown order
# can also disagree with OpenCV's DirectShow order (see camera.py), so this file
# is the reliable escape hatch and deliberately wins over the --camera-index arg
# Unity passes.
CAMERA_INDEX_FILE = "camera_index.txt"
CAMERA_INDEX_ENV = "VISION_CAMERA_INDEX"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def prepare_frozen_cwd() -> None:
    """Make relative paths like models/escape_gestures.keras work when frozen."""
    if is_frozen():
        os.chdir(Path(sys.executable).resolve().parent)


def has_console() -> bool:
    """False when launched via pythonw.exe (Unity's player path): no window
    is allocated, so every print() has nowhere to go and a bug report from
    that run carries no diagnostics at all.
    """
    if sys.platform != "win32":
        return True
    import ctypes

    return bool(ctypes.windll.kernel32.GetConsoleWindow())


def redirect_output_to_log(log_path: Path) -> None:
    """Send stdout/stderr to a file when there is no console to print to.

    Player runs (pythonw.exe, no console) would otherwise lose every
    diagnostic line this module prints — camera negotiated size, [lock]
    state changes, [perf] stats, [cam] recovery events — which is exactly
    the evidence needed to tell a bad camera feed from a genuine bug when a
    player reports "gestures don't work". Overwritten each run: only the
    most recent session matters for that.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file


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
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help=(
            "OpenCV webcam index (0, 1, 2, …). "
            f"Overridden by the {CAMERA_INDEX_ENV} env var or a "
            f"{CAMERA_INDEX_FILE} file next to the server. "
            "Default: CAMERA_INDEX in config.py."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help=(
            "Probe webcam indices 0-9 (DirectShow), print which ones open, "
            "then exit. Run this to find the right --camera-index."
        ),
    )
    return parser.parse_args(argv)


def _camera_index_search_dirs() -> list[Path]:
    """Where to look for the camera_index.txt override, most specific first.

    The Unity launcher runs the server with its working directory set to
    ``VisionServer/app`` (or the exe's own dir when frozen), so both that folder
    and its parent (``VisionServer/``) are natural places for a player to drop
    the file.
    """
    dirs: list[Path] = [Path.cwd(), Path.cwd().parent]
    exe_dir = Path(sys.executable).resolve().parent
    dirs += [exe_dir, exe_dir.parent]
    # Preserve order while removing duplicates.
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _read_camera_index_file() -> tuple[int, Path] | None:
    """First readable camera_index.txt holding an int, or None.

    Blank lines and ``#`` comments are ignored so the file can carry a note.
    """
    for directory in _camera_index_search_dirs():
        path = directory / CAMERA_INDEX_FILE
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                return int(line), path
            except ValueError:
                break  # File exists but is garbled; fall through to next source.
    return None


def resolve_camera_index(args: argparse.Namespace) -> tuple[int, str]:
    """Pick the webcam index and report where it came from.

    Precedence (highest first): the VISION_CAMERA_INDEX env var, a
    camera_index.txt file, the --camera-index arg, then the config default.
    The env var and file deliberately beat the arg because the arg is whatever
    Unity's dropdown produced, and that dropdown's device order can disagree
    with OpenCV's — so these two are the player's manual override.
    """
    env_raw = os.environ.get(CAMERA_INDEX_ENV)
    if env_raw is not None and env_raw.strip():
        try:
            return int(env_raw.strip()), f"{CAMERA_INDEX_ENV} env var"
        except ValueError:
            print(
                f"[camera] Ignoring {CAMERA_INDEX_ENV}={env_raw!r} "
                "(not an integer)."
            )

    from_file = _read_camera_index_file()
    if from_file is not None:
        index, path = from_file
        return index, f"{path}"

    if args.camera_index is not None:
        return args.camera_index, "--camera-index arg"

    return CAMERA_INDEX, "config.py default"


def list_cameras(max_index: int = 10) -> None:
    """Print which DirectShow webcam indices open and deliver a frame.

    Imported lazily so --list-cameras is the only path that pulls in cv2 here.
    """
    import cv2

    print(f"Probing webcam indices 0-{max_index - 1} (DirectShow)...")
    found = 0
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        opened = cap.isOpened()
        grabbed = False
        if opened:
            grabbed, _ = cap.read()
            found += 1
        cap.release()
        if opened:
            status = "opens + delivers frames" if grabbed else "opens but NO frame"
            print(f"  index {index}: {status}")
        else:
            print(f"  index {index}: (no device)")
    if found == 0:
        print("No cameras opened. Check Windows camera privacy settings.")
    else:
        print(
            "Pick an index that 'delivers frames', then put it in "
            f"{CAMERA_INDEX_FILE} (a single number) next to the server."
        )


def _camera_opens(index: int) -> bool:
    """True if the DirectShow camera at ``index`` opens and delivers one frame.

    A missing device fails to open; a real camera both opens and reads a frame.
    Released immediately so the server's own capture can grab it right after.
    """
    if index < 0:
        return False
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            return False
        grabbed, _ = cap.read()
        return bool(grabbed)
    finally:
        cap.release()


def resolve_working_camera_index(
    preferred: int, source: str, max_index: int = 10
) -> tuple[int, str]:
    """Use ``preferred`` if a camera is actually there; otherwise auto-fall back.

    A hardcoded index is fragile across machines: index 1 is the real webcam on
    one PC and absent on a single-camera laptop (where 0 is the webcam). So when
    the preferred index has no device, probe 0..max_index-1 and take the first
    that delivers frames. This lets ONE build work on both the developer's
    machine and a tester's without editing camera_index.txt. The manual override
    still wins whenever its index genuinely exists.
    """
    if _camera_opens(preferred):
        return preferred, source

    for candidate in range(max_index):
        if candidate == preferred:
            continue
        if _camera_opens(candidate):
            return candidate, (
                f"auto-detected (index {preferred} from {source} "
                "had no camera)"
            )

    # Nothing responded. Keep the preferred index so the normal camera-reopen
    # and retry path (camera.py) handles it — the device may appear late.
    return preferred, f"{source} (no camera responded yet; will keep retrying)"


def resolve_show_preview(args: argparse.Namespace) -> bool:
    """True means the debug HighGUI window. Independent of Unity calibrate preview."""
    if args.headless is True:
        return False
    if args.headless is False:
        return True
    if is_frozen():
        return False
    return bool(SHOW_PREVIEW)
