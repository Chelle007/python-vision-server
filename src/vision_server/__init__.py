"""Python vision server for hand/face tracking and gesture classification."""

from vision_server.runtime import apply_thread_env

# Must run before any submodule pulls in cv2/mediapipe/tensorflow, since those
# read their thread-pool size from the environment at native-library init.
apply_thread_env()

__version__ = "0.2.0"
