"""
Train XGBoost policy model on self-play game logs.

Run from project root:
    python training/train.py
    python training/train.py --log_dir data/game_logs/ --out models/model.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.encoder import FEATURE_NAMES, FEATURE_DIM
from training.feature_builder import build_dataset


def train(log_dir: str = "data/game_logs/", out_path: str = "models/model.pkl"):
    print("=== PTCG Agent — XGBoost Training ===\n")

    # 1. Load data
    X, y = build_dataset(log_dir)

    if len(y) < 100:
        print(f"WARNING: Only {len(y)} samples — run more self-play games for better results.")

    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y if y.mean() not in (0.0, 1.0) else None
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}\n")

    # 3. Train
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric="logloss",
        early_stopping_rounds=30,
        verbosity=1,
        tree_method="hist",
        device="cpu",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # 4. Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")

    # 5. Feature importance
    importances = model.feature_importances_
    importance_pairs = sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: -x[1]
    )
    print("\nTop 15 features:")
    for name, score in importance_pairs[:15]:
        print(f"  {name:<30} {score:.4f}")

    # 6. Save model
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"\nSaved model → {out_path}")

    # Save feature importance JSON alongside model
    imp_path = Path(out_path).with_suffix(".feature_importance.json")
    imp_data = [{"name": n, "score": float(s)} for n, s in importance_pairs]
    imp_path.write_text(json.dumps(imp_data, indent=2))
    print(f"Saved feature importance → {imp_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="data/game_logs/")
    parser.add_argument("--out", default="models/model.pkl")
    args = parser.parse_args()
    train(log_dir=args.log_dir, out_path=args.out)
