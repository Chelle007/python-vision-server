import os

from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

from vision_server.config import (
    MODEL_PATH,
    NUM_FEATURES,
    NUM_FRAMES,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
)
from vision_server.evaluation.dataset import load_sequences, train_test_arrays
from vision_server.gestures.dynamic.labels import CLASSES


def main():
    print("--- STARTING DATA LOADING ---")
    X, y = load_sequences()
    X_train, X_test, y_train, y_test = train_test_arrays(X, y, categorical=True)

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
