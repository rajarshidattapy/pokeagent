"""Shared helpers for the FastAPI server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_card_lookup(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_model(path: str):
    """Load the XGBoost policy. Returns None if file doesn't exist yet."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    p = Path(path)
    if not p.exists():
        return None
    try:
        from agent.policy import XGBoostPolicy
        return XGBoostPolicy(str(p))
    except Exception as e:
        print(f"Warning: could not load model at {path}: {e}")
        return None


def list_game_logs(log_dir: str, full: bool = False) -> list[Any]:
    """
    full=False: returns list of {id, winner, turns} dicts.
    full=True: returns list of full game log dicts.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    result = []
    for f in sorted(log_path.glob("game_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if full:
                result.append(data)
            else:
                result.append({
                    "id": data.get("game_id", f.stem),
                    "winner": data.get("winner"),
                    "total_turns": data.get("total_turns", len(data.get("turns", []))),
                })
        except Exception:
            continue
    return result
