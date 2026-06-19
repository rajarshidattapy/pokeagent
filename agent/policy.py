"""Phase 2: XGBoost policy. Loads model.pkl and scores legal actions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from encoder import encode


class XGBoostPolicy:
    def __init__(self, model_path: str):
        import joblib
        self.model = joblib.load(model_path)

    def select_action(self, game_state: dict, legal_actions: list[str]) -> str:
        if not legal_actions:
            return "PASS"
        try:
            features = np.array([encode(game_state, a) for a in legal_actions])
            win_probs = self.model.predict_proba(features)[:, 1]
            return legal_actions[int(np.argmax(win_probs))]
        except Exception:
            return legal_actions[0]

    def score_actions(self, game_state: dict, legal_actions: list[str]) -> dict[str, float]:
        if not legal_actions:
            return {}
        try:
            features = np.array([encode(game_state, a) for a in legal_actions])
            win_probs = self.model.predict_proba(features)[:, 1]
            return {a: float(p) for a, p in zip(legal_actions, win_probs)}
        except Exception:
            return {a: 0.0 for a in legal_actions}


class HeuristicPolicy:
    """Wraps heuristic agent to match the policy interface."""

    def select_action(self, game_state: dict, legal_actions: list[str]) -> str:
        from heuristic import select_action
        return select_action(legal_actions, game_state)

    def score_actions(self, game_state: dict, legal_actions: list[str]) -> dict[str, float]:
        chosen = self.select_action(game_state, legal_actions)
        return {a: (1.0 if a == chosen else 0.0) for a in legal_actions}


class RandomPolicy:
    """Picks a random legal action."""

    def select_action(self, game_state: dict, legal_actions: list[str]) -> str:
        import random
        return random.choice(legal_actions) if legal_actions else "PASS"

    def score_actions(self, game_state: dict, legal_actions: list[str]) -> dict[str, float]:
        n = len(legal_actions)
        return {a: 1.0 / n for a in legal_actions} if n else {}


def load_policy(name: str, model_path: str = "models/model.pkl"):
    if name == "xgboost":
        return XGBoostPolicy(model_path)
    elif name == "heuristic":
        return HeuristicPolicy()
    elif name == "random":
        return RandomPolicy()
    else:
        raise ValueError(f"Unknown policy: {name!r}")
