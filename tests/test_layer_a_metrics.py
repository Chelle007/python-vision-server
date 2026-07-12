"""Unit tests for Layer A metric helpers (no model/data required)."""

import numpy as np

from vision_server.gestures.dynamic.labels import CLASSES


def test_idle_to_action_fp_rate_logic():
    idle_id = CLASSES.index("Idle")
    action_id = CLASSES.index("Turn_Key")
    y_true = np.array([idle_id, idle_id, idle_id, action_id])
    y_pred = np.array([idle_id, action_id, idle_id, action_id])
    idle_mask = y_true == idle_id
    fp_rate = float(np.mean(y_pred[idle_mask] != idle_id))
    assert abs(fp_rate - (1 / 3)) < 1e-9
