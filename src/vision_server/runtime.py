"""CPU thread budgeting so the vision server can coexist with Unity.

MediaPipe, OpenCV and TensorFlow each size their thread pools to the *logical*
core count. On an Apple M-series chip that is ~10 threads per library fighting
over only 4 performance cores, while Unity — a foreground GUI app — holds those
cores at higher QoS and pushes this process onto the efficiency cores.

The symptom is distinctive: every pipeline stage inflates *together* (hand
tracking, face mesh and render all slow at once) rather than one stage becoming
expensive. That is starvation, not an algorithmic cost, so the fix is to stop
oversubscribing rather than to make any single stage cheaper.

Env vars must be set before the native libraries initialise, which is why
:func:`apply_thread_env` is called from ``vision_server/__init__.py`` — that
runs before any submodule import. ``cv2.setNumThreads`` is a runtime call and
lives in :func:`apply_opencv_threads` instead.
"""

from __future__ import annotations

import os

from vision_server.config import CPU_THREADS, TF_INTER_OP_THREADS

# Read by OpenMP, TensorFlow and Accelerate at library init time.
_THREAD_ENV = {
    "OMP_NUM_THREADS": str(CPU_THREADS),
    "OPENBLAS_NUM_THREADS": str(CPU_THREADS),
    "MKL_NUM_THREADS": str(CPU_THREADS),
    "VECLIB_MAXIMUM_THREADS": str(CPU_THREADS),
    "NUMEXPR_NUM_THREADS": str(CPU_THREADS),
    "TF_NUM_INTRAOP_THREADS": str(CPU_THREADS),
    "TF_NUM_INTEROP_THREADS": str(TF_INTER_OP_THREADS),
}


def apply_thread_env() -> None:
    """Cap native thread pools. Must run before cv2/mediapipe/tensorflow load.

    Existing values win, so a caller can still override from the shell:
    ``OMP_NUM_THREADS=8 python -m vision_server.app``
    """
    for key, value in _THREAD_ENV.items():
        os.environ.setdefault(key, value)


def apply_opencv_threads() -> None:
    """Cap OpenCV's internal pool. Safe to call any time after ``import cv2``.

    macOS builds use GCD as the parallel framework, and GCD ignores any
    positive thread count — ``setNumThreads(2)`` leaves the count at the core
    count. Only 0, meaning "run sequentially", is honoured. Falling back to
    sequential is the right trade here: OpenCV's share of a frame is the flip
    and colour convert, ~1-2ms, so running those inline costs almost nothing
    and takes a pool of threads out of contention with Unity.
    """
    import cv2

    cv2.setNumThreads(CPU_THREADS)
    if cv2.getNumThreads() > CPU_THREADS:
        cv2.setNumThreads(0)
