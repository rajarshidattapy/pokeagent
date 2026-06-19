"""
FastAPI server — PTCG Agent API.

Run from project root:
    uvicorn server.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.schemas import (
    CardResponse, DeckRequest, PredictRequest, PredictResponse,
    SimulateRequest, SimulateResponse, StatsResponse,
)
from server.utils import list_game_logs, load_card_lookup, load_model

from agent.card_constants import ACE_SPEC_IDS

app = FastAPI(title="PTCG Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve UI at /ui
ui_dir = Path(__file__).parent.parent / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

# ── Globals ──────────────────────────────────────────────────────────────────

MODEL = None
CARDS: dict = {}

LOG_DIR   = str(Path(__file__).parent.parent / "data" / "game_logs")
DECK_PATH = Path(__file__).parent.parent / "deck" / "deck.json"


@app.on_event("startup")
def startup():
    global MODEL, CARDS
    root = Path(__file__).parent.parent
    MODEL = load_model(str(root / "models" / "model.pkl"))
    CARDS = load_card_lookup(str(root / "data" / "card_lookup.json"))
    print(f"Startup: model_loaded={MODEL is not None}, cards={len(CARDS)}")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "cards_loaded": len(CARDS),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded — run training/train.py first")
    scores = MODEL.score_actions(req.game_state, req.legal_actions)
    if not scores:
        return PredictResponse(best_action="PASS", action_scores={})
    best = max(scores, key=lambda k: scores[k])
    return PredictResponse(best_action=best, action_scores=scores)


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    try:
        from training.self_play import run_games
        result = run_games(
            num_games=req.num_games,
            agent_a=req.agent_a,
            agent_b=req.agent_b,
            deck_a=req.deck_a,
            deck_b=req.deck_b,
            log_dir=LOG_DIR,
        )
        return SimulateResponse(**result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/cards", response_model=list[CardResponse])
def get_cards(
    expansion: str | None = None,
    category: str | None = None,
    type: str | None = None,
    q: str | None = None,
):
    cards = list(CARDS.values())
    if expansion:
        cards = [c for c in cards if c.get("expansion") == expansion]
    if category:
        cards = [c for c in cards if c.get("stage", "").lower() == category.lower()]
    if type:
        cards = [c for c in cards if c.get("type", "") == type]
    if q:
        q_lower = q.lower()
        cards = [c for c in cards if q_lower in c["name"].lower()]
    return cards


@app.get("/cards/{card_id}", response_model=CardResponse)
def get_card(card_id: int):
    card = CARDS.get(str(card_id))
    if not card:
        raise HTTPException(404, f"Card {card_id} not found")
    return card


@app.get("/deck")
def get_deck():
    if DECK_PATH.exists():
        return json.loads(DECK_PATH.read_text(encoding="utf-8"))
    return {"cards": [], "name": "empty"}


@app.post("/deck")
def save_deck(req: DeckRequest):
    total = sum(e.count for e in req.cards)
    if total != 60:
        raise HTTPException(400, f"Deck must have exactly 60 cards, got {total}")

    ace_count = sum(e.count for e in req.cards if e.card_id in ACE_SPEC_IDS)
    if ace_count > 1:
        raise HTTPException(400, f"Deck has {ace_count} ACE SPEC cards — max 1 allowed")

    DECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECK_PATH.write_text(
        json.dumps(req.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"saved": True, "total_cards": total}


@app.get("/logs")
def list_logs():
    logs = list_game_logs(LOG_DIR, full=False)
    return {"logs": logs, "total": len(logs)}


@app.get("/logs/{log_id}")
def get_log(log_id: str):
    path = Path(LOG_DIR) / f"{log_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Log {log_id} not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    logs = list_game_logs(LOG_DIR, full=True)
    if not logs:
        return StatsResponse(
            total_games=0, win_rate=0.0, avg_turns=0.0,
            top_winning_actions=[], feature_importance=[],
        )

    wins_a = sum(1 for g in logs if g.get("winner") == "agent_a")
    avg_turns = sum(g.get("total_turns", 0) for g in logs) / len(logs)

    feat_imp = []
    imp_path = Path(__file__).parent.parent / "models" / "model.feature_importance.json"
    if imp_path.exists():
        feat_imp = json.loads(imp_path.read_text(encoding="utf-8"))[:15]

    return StatsResponse(
        total_games=len(logs),
        win_rate=round(wins_a / len(logs), 3),
        avg_turns=round(avg_turns, 1),
        top_winning_actions=[],
        feature_importance=feat_imp,
    )
