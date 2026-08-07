"""Layer A — offline LSTM metrics on the fixed held-out test split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from vision_server.config import MODEL_PATH
from vision_server.evaluation.dataset import load_sequences, train_test_arrays
from vision_server.gestures.dynamic.labels import CLASSES


@dataclass
class LayerAResult:
    overall_accuracy: float
    per_class_accuracy: dict[str, float]
    idle_to_action_fp_rate: float
    confusion: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    n_test: int

    def format_report(self) -> str:
        lines = [
            "=== Layer A — Offline LSTM ===",
            f"Test samples: {self.n_test}",
            f"Overall accuracy: {self.overall_accuracy:.4f}  (target ≥ 0.85)",
            "",
            "Per-class accuracy (target ≥ 0.80 each):",
        ]
        for name in CLASSES:
            acc = self.per_class_accuracy.get(name, float("nan"))
            lines.append(f"  {name:12s}  {acc:.4f}")

        lines.extend(
            [
                "",
                f"FP rate (Idle → action): {self.idle_to_action_fp_rate:.4f}",
                "",
                "Confusion matrix (rows=true, cols=pred):",
                f"  labels: {CLASSES}",
                str(self.confusion),
                "",
                classification_report(
                    self.y_true,
                    self.y_pred,
                    labels=list(range(len(CLASSES))),
                    target_names=CLASSES,
                    digits=4,
                    zero_division=0,
                ),
            ]
        )
        return "\n".join(lines)


def _load_model(model_path: str):
    from tensorflow.keras.models import load_model

    return load_model(model_path)


def evaluate_lstm_on_test_set(
    model_path: str = MODEL_PATH,
    data_dir: str | None = None,
) -> LayerAResult:
    """Run saved model on the fixed 20% test split (random_state=42)."""
    kwargs = {"data_dir": data_dir} if data_dir is not None else {}
    X, y = load_sequences(**kwargs)
    _, X_test, _, y_test = train_test_arrays(X, y, categorical=False)

    model = _load_model(model_path)
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = y_test.astype(int)

    overall = float(accuracy_score(y_true, y_pred))
    per_class: dict[str, float] = {}
    for idx, name in enumerate(CLASSES):
        mask = y_true == idx
        if not np.any(mask):
            per_class[name] = float("nan")
        else:
            per_class[name] = float(np.mean(y_pred[mask] == idx))

    idle_id = CLASSES.index("Idle")
    idle_mask = y_true == idle_id
    if np.any(idle_mask):
        fp_rate = float(np.mean(y_pred[idle_mask] != idle_id))
    else:
        fp_rate = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    return LayerAResult(
        overall_accuracy=overall,
        per_class_accuracy=per_class,
        idle_to_action_fp_rate=fp_rate,
        confusion=cm,
        y_true=y_true,
        y_pred=y_pred,
        n_test=int(len(y_true)),
    )
