"""Shared dataset loading for training and Layer A evaluation."""

from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from vision_server.config import (
    DATA_DIR,
    FOLDER_MAPPING,
    NUM_FEATURES,
    NUM_FRAMES,
    TRAIN_RANDOM_STATE,
    TRAIN_TEST_SPLIT,
)
from vision_server.gestures.dynamic.labels import CLASSES


def load_sequences(
    data_dir: str = DATA_DIR,
    folder_mapping: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load (.npy) clips into X (N, frames, features) and integer label ids."""
    mapping = folder_mapping or FOLDER_MAPPING
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for folder_name, target_class in mapping.items():
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Skipping: {folder_path} (Folder not found)")
            continue

        filenames = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
        print(
            f"Loading {len(filenames)} files from '{folder_name}' "
            f"into class '{target_class}'..."
        )
        label_id = CLASSES.index(target_class)

        for filename in filenames:
            file_path = os.path.join(folder_path, filename)
            res = np.load(file_path)
            if res.shape == (NUM_FRAMES, NUM_FEATURES):
                sequences.append(res)
                labels.append(label_id)

    if not sequences:
        raise FileNotFoundError(
            f"No valid .npy clips found under {data_dir!r}. "
            "Record data first (scripts/record_data.py)."
        )

    X = np.array(sequences)
    y = np.array(labels, dtype=int)
    print(f"TOTAL SAMPLES LOADED: {X.shape[0]}")
    print(f"INPUT DATA SHAPE: {X.shape} (Samples, Frames, Features)")
    return X, y


def train_test_arrays(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = TRAIN_TEST_SPLIT,
    random_state: int = TRAIN_RANDOM_STATE,
    categorical: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed split used by train + Layer A (never train on the test portion)."""
    y_split = to_categorical(y, num_classes=len(CLASSES)).astype(int) if categorical else y
    return train_test_split(
        X,
        y_split,
        test_size=test_size,
        random_state=random_state,
    )
