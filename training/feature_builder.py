"""
Read game_logs/*.json → build (X, y) training arrays.

Each (game_state, action) decision point becomes one row.
Label: 1 if that player won, 0 if lost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.encoder import encode, FEATURE_DIM


def build_dataset(log_dir: str = "data/game_logs/") -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y):
        X shape: (N, FEATURE_DIM)
        y shape: (N,) — binary int labels
    """
    log_dir_path = Path(log_dir)
    log_files = sorted(log_dir_path.glob("game_*.json"))

    if not log_files:
        raise FileNotFoundError(f"No game logs found in {log_dir}")

    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    skipped = 0

    for log_file in log_files:
        try:
            log = json.loads(log_file.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue

        for turn in log.get("turns", []):
            action = turn.get("action_taken")
            outcome = turn.get("outcome")
            gs = turn.get("game_state")

            if action is None or outcome is None or gs is None:
                continue

            try:
                vec = encode(gs, action)
                rows_X.append(vec)
                rows_y.append(int(outcome))
            except Exception:
                skipped += 1
                continue

    if not rows_X:
        raise ValueError("No valid training samples found in game logs.")

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int32)

    pos = y.sum()
    print(f"Dataset: {len(y)} samples from {len(log_files)} games ({skipped} skipped)")
    print(f"  Positive (win): {pos} ({100*pos/len(y):.1f}%)")
    print(f"  Feature dim: {X.shape[1]}")

    return X, y


if __name__ == "__main__":
    X, y = build_dataset()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
