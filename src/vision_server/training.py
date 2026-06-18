import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from vision_server.config import (
    DATA_DIR,
    FOLDER_MAPPING,
    MODEL_PATH,
    NUM_FEATURES,
    NUM_FRAMES,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_RANDOM_STATE,
    TRAIN_TEST_SPLIT,
)
from vision_server.gestures.dynamic.labels import CLASSES


def main():
    print("--- STARTING DATA LOADING ---")
    sequences, labels = [], []

    for folder_name, target_class in FOLDER_MAPPING.items():
        folder_path = os.path.join(DATA_DIR, folder_name)

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

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(CLASSES)).astype(int)

    print(f"\nTOTAL SAMPLES LOADED: {X.shape[0]}")
    print(f"INPUT DATA SHAPE: {X.shape} (Samples, Frames, Features)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SPLIT,
        random_state=TRAIN_RANDOM_STATE,
    )

    print("\n--- BUILDING MODEL ---")
    model = Sequential()
    model.add(
        LSTM(
            64,
            return_sequences=True,
            activation="relu",
            input_shape=(NUM_FRAMES, NUM_FEATURES),
        )
    )
    model.add(LSTM(32, return_sequences=False, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(len(CLASSES), activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    print("\n--- TRAINING STARTED ---")
    model.fit(
        X_train,
        y_train,
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        validation_data=(X_test, y_test),
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nSUCCESS! Model saved as {MODEL_PATH}")
