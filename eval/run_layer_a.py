#!/usr/bin/env python3
"""Layer A — offline LSTM eval on the fixed held-out test split.

Usage (from python-vision-server/):
  python eval/run_layer_a.py
  python eval/run_layer_a.py --model models/escape_gestures.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python eval/run_layer_a.py` without install quirks
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from vision_server.config import MODEL_PATH
from vision_server.evaluation.lstm_metrics import evaluate_lstm_on_test_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer A offline LSTM evaluation")
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=f"Path to .keras model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data/ directory (default: config DATA_DIR)",
    )
    args = parser.parse_args()

    result = evaluate_lstm_on_test_set(model_path=args.model, data_dir=args.data_dir)
    print()
    print(result.format_report())

    # One-line summary for spreadsheet paste
    pc = result.per_class_accuracy
    print("--- Spreadsheet paste (Layer A) ---")
    print(
        f"overall={result.overall_accuracy:.4f}  "
        f"Idle={pc.get('Idle', float('nan')):.4f}  "
        f"Turn_Key={pc.get('Turn_Key', float('nan')):.4f}  "
        f"Pull_Lever={pc.get('Pull_Lever', float('nan')):.4f}  "
        f"FP_Idle_to_action={result.idle_to_action_fp_rate:.4f}"
    )


if __name__ == "__main__":
    main()
