from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class PredictRequest(BaseModel):
    game_state: dict[str, Any]
    legal_actions: list[str]


class PredictResponse(BaseModel):
    best_action: str
    action_scores: dict[str, float]


class SimulateRequest(BaseModel):
    num_games: int = 10
    agent_a: str = "heuristic"   # "xgboost" | "heuristic" | "random"
    agent_b: str = "random"
    deck_a: list[int] | None = None   # flat 60-card list of card_ids; None = default
    deck_b: list[int] | None = None


class SimulateResponse(BaseModel):
    games_played: int
    wins_a: int
    wins_b: int
    draws: int
    win_rate_a: float
    log_ids: list[str]


class CardResponse(BaseModel):
    card_id: int
    name: str
    expansion: str | None
    collection_no: str
    stage: str
    rule: str
    hp: int | None
    type: str | None
    retreat: int | None
    is_ex: bool
    is_supporter: bool
    is_item: bool
    is_energy: bool
    moves: list[dict]


class DeckEntry(BaseModel):
    card_id: int
    count: int


class DeckRequest(BaseModel):
    cards: list[DeckEntry]
    name: str = "default"


class FeatureImportanceEntry(BaseModel):
    name: str
    score: float


class StatsResponse(BaseModel):
    total_games: int
    win_rate: float
    avg_turns: float
    top_winning_actions: list[dict]
    feature_importance: list[FeatureImportanceEntry]
