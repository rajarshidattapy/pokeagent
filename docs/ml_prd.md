# PRD: PTCG Agent — Implementation Stack
### Paramarsh Labs | ML Model + FastAPI Backend + HTML UI
**Version:** 1.0 | **Date:** June 2026 | **Depends on:** PRD v1.0 (Agent Strategy)

---

## 1. Overview

This PRD covers the **concrete build** — the actual files, code structure, and interfaces that make the agent work end-to-end. It spans three layers:

1. **ML Model** — XGBoost trained on self-play game logs, serialized as `.pkl`
2. **FastAPI Server** — loads the `.pkl`, exposes prediction endpoints, handles game loop integration
3. **HTML UI** — browser-based dashboard to visualize agent decisions, game state, and model outputs during development and for the Strategy Report

This is not the Kaggle submission format (which uses the simulator's own runner). The stack described here is the **local development + analysis environment** that lets you train, debug, and demonstrate the agent.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Browser (UI)                          │
│                                                              │
│  index.html ──── deck_builder.html ──── match_viewer.html    │
│       │                │                      │              │
│       └────────────────┴──────────────────────┘             │
│                         fetch() / REST                       │
└─────────────────────────────┬────────────────────────────────┘
                              │ HTTP
┌─────────────────────────────▼────────────────────────────────┐
│                    FastAPI Server (main.py)                   │
│                                                              │
│  POST /predict      → load model.pkl, return action scores   │
│  POST /simulate     → run N games, return logs               │
│  GET  /cards        → return card metadata from CSV          │
│  GET  /deck         → return current deck list               │
│  POST /deck         → save a new deck list                   │
│  GET  /logs         → return game log list                   │
│  GET  /logs/{id}    → return full game log JSON              │
│  GET  /stats        → return win/loss/ELO metrics            │
│                                                              │
└─────────────────────────────┬────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
┌─────────▼──────┐  ┌─────────▼──────┐  ┌────────▼───────┐
│  model.pkl     │  │ EN_Card_Data   │  │ game_logs/     │
│  (XGBoost)     │  │ .csv           │  │ *.json         │
└────────────────┘  └────────────────┘  └────────────────┘
```

---

## 3. File Structure

```
ptcg-agent/
│
├── data/
│   ├── EN_Card_Data.csv            # Competition dataset (card metadata)
│   ├── card_lookup.json            # Pre-built: id → {name, type, hp, moves, ...}
│   └── game_logs/                  # Self-play logs; one .json per game
│       ├── game_0001.json
│       └── ...
│
├── agent/
│   ├── __init__.py
│   ├── encoder.py                  # GameState → feature vector (numpy array)
│   ├── action_space.py             # Enumerate legal actions from game state
│   ├── heuristic.py                # Phase 1: rule-based fallback policy
│   └── policy.py                   # Phase 2: loads model.pkl, scores actions
│
├── training/
│   ├── self_play.py                # Runs N games, writes game_logs/
│   ├── feature_builder.py          # Reads logs → builds X, y for training
│   └── train.py                    # Trains XGBoost, writes models/model.pkl
│
├── models/
│   └── model.pkl                   # Trained XGBoost model (joblib serialized)
│
├── server/
│   ├── main.py                     # FastAPI app — all routes
│   ├── schemas.py                  # Pydantic request/response models
│   └── utils.py                    # Shared helpers (load model, load CSV)
│
├── ui/
│   ├── index.html                  # Dashboard / home
│   ├── deck_builder.html           # Browse cards, build 60-card deck
│   ├── match_viewer.html           # Replay a game log turn by turn
│   └── assets/
│       ├── style.css               # Shared stylesheet
│       └── app.js                  # Shared JS utilities (fetch wrappers, etc.)
│
├── deck/
│   └── deck.json                   # Current 60-card deck [{card_id, count}, ...]
│
├── tests/
│   ├── test_encoder.py
│   ├── test_policy.py
│   └── test_api.py
│
├── requirements.txt
└── README.md
```

---

## 4. ML Model

### 4.1 Training Data Format

Each game log (`game_logs/game_XXXX.json`) stores every decision point in the game:

```json
{
  "game_id": "game_0001",
  "winner": "agent_a",
  "turns": [
    {
      "turn": 1,
      "player": "agent_a",
      "game_state": { ... },
      "legal_actions": ["ATTACK_0", "PLAY_ENERGY_3", "PLAY_SUPPORTER_7"],
      "action_taken": "PLAY_SUPPORTER_7",
      "outcome": 1
    }
  ]
}
```

`outcome` is the game result from the perspective of the acting player: `1` = won, `0` = lost.

### 4.2 Feature Vector (`encoder.py`)

`encode(game_state, action) → np.array of shape (N,)`

| Feature Group | Dims | Notes |
|---|---|---|
| Active Pokémon HP ratio (mine) | 1 | current_hp / max_hp |
| Active Pokémon type (one-hot, 9 types) | 9 | Grass/Fire/Water/Lightning/Psychic/Fighting/Darkness/Metal/Colorless |
| Active energy attached (per type) | 9 | count per type |
| Active retreat cost | 1 | raw value |
| Bench size (mine) | 1 | 0–5 |
| Bench HP ratios (mine, 5 slots) | 5 | 0 if empty |
| Opponent active HP ratio | 1 | |
| Opponent active type (one-hot) | 9 | |
| Opponent energy attached | 9 | |
| Opponent bench size | 1 | |
| My hand size | 1 | |
| My hand: # energy cards | 1 | |
| My hand: # Trainer cards | 1 | |
| My hand: # Supporter cards | 1 | |
| My hand: # Pokémon cards | 1 | |
| My deck remaining | 1 | |
| My prizes remaining | 1 | |
| Opponent prizes remaining | 1 | |
| Turn number (normalized) | 1 | turn / 30 |
| Active stadium ID (one-hot, top 10 stadiums) | 10 | 0s if none |
| Active Pokémon status (burned/poisoned/confused/paralyzed/asleep) | 5 | binary each |
| **Action features** | | |
| Action type (one-hot: ATTACK/ENERGY/TRAINER/SUPPORTER/EVOLVE/RETREAT/PASS) | 7 | |
| If ATTACK: expected damage (normalized) | 1 | dmg / 300 |
| If ATTACK: would KO opponent flag | 1 | binary |
| If ATTACK: opponent can KO me next turn (risk flag) | 1 | binary |
| If PLAY_ENERGY: enables attack next turn | 1 | binary |
| If EVOLVE: HP gain from evolution | 1 | normalized |
| If PLAY_SUPPORTER: draw count (e.g., Prof Research = 7) | 1 | normalized |
| **Total** | **~80** | |

```python
# encoder.py — core function signature
def encode(game_state: dict, action: str) -> np.ndarray:
    """
    Returns a 1D numpy array of floats representing
    (game_state, action) as input to the policy model.
    Returns zeros for irrelevant action-specific features.
    """
```

### 4.3 Training Script (`training/train.py`)

```python
# training/train.py
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from feature_builder import build_dataset

# 1. Load training data
X, y = build_dataset("../data/game_logs/")

# 2. Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

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
    verbosity=1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# 4. Evaluate
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_val, y_prob):.4f}")

# 5. Feature importance
feat_names = [f"feat_{i}" for i in range(X.shape[1])]  # replace with real names
importance = dict(zip(feat_names, model.feature_importances_))
top_features = sorted(importance.items(), key=lambda x: -x[1])[:15]
print("\nTop 15 features:")
for name, score in top_features:
    print(f"  {name}: {score:.4f}")

# 6. Save
joblib.dump(model, "../models/model.pkl")
print("\nSaved: models/model.pkl")
```

### 4.4 Model Inference (`agent/policy.py`)

```python
# agent/policy.py
import joblib
import numpy as np
from encoder import encode

class XGBoostPolicy:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def select_action(self, game_state: dict, legal_actions: list[str]) -> str:
        if not legal_actions:
            return "PASS"

        features = np.array([encode(game_state, a) for a in legal_actions])
        win_probs = self.model.predict_proba(features)[:, 1]
        best_idx = int(np.argmax(win_probs))
        return legal_actions[best_idx]

    def score_actions(self, game_state: dict, legal_actions: list[str]) -> dict:
        """Returns {action: win_probability} for all legal actions. Used by UI."""
        if not legal_actions:
            return {}
        features = np.array([encode(game_state, a) for a in legal_actions])
        win_probs = self.model.predict_proba(features)[:, 1]
        return {a: float(p) for a, p in zip(legal_actions, win_probs)}
```

---

## 5. FastAPI Server

### 5.1 Pydantic Schemas (`server/schemas.py`)

```python
# server/schemas.py
from pydantic import BaseModel
from typing import Any

class PredictRequest(BaseModel):
    game_state: dict[str, Any]
    legal_actions: list[str]

class PredictResponse(BaseModel):
    best_action: str
    action_scores: dict[str, float]   # action → win probability

class SimulateRequest(BaseModel):
    num_games: int = 10
    agent_a: str = "xgboost"          # "xgboost" | "heuristic" | "random"
    agent_b: str = "heuristic"
    deck_a: list[int]                  # list of card IDs (60 cards)
    deck_b: list[int]

class SimulateResponse(BaseModel):
    games_played: int
    wins_a: int
    wins_b: int
    draws: int
    win_rate_a: float
    log_ids: list[str]                 # game log filenames written

class CardResponse(BaseModel):
    card_id: int
    name: str
    expansion: str
    collection_no: int
    category: str
    hp: int | None
    type: str | None
    stage: str | None
    moves: list[dict] | None

class DeckRequest(BaseModel):
    cards: list[dict]                  # [{card_id: int, count: int}, ...]
    name: str = "default"

class StatsResponse(BaseModel):
    total_games: int
    win_rate: float
    avg_turns: float
    top_winning_actions: list[dict]
    feature_importance: list[dict]     # [{name, score}, ...]
```

### 5.2 FastAPI App (`server/main.py`)

```python
# server/main.py
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib

from schemas import (
    PredictRequest, PredictResponse,
    SimulateRequest, SimulateResponse,
    CardResponse, DeckRequest, StatsResponse
)
from utils import load_model, load_card_lookup, list_game_logs

app = FastAPI(title="PTCG Agent API", version="1.0.0")

# CORS — allow UI pages served as local files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve UI files at /ui
app.mount("/ui", StaticFiles(directory="../ui"), name="ui")

# ── Startup ─────────────────────────────────────────────────────────────

MODEL = None
CARDS = None

@app.on_event("startup")
def startup():
    global MODEL, CARDS
    MODEL = load_model("../models/model.pkl")
    CARDS = load_card_lookup("../data/card_lookup.json")

# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Given a game state and list of legal actions,
    return the best action and win-probability scores for all actions.
    """
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    scores = MODEL.score_actions(req.game_state, req.legal_actions)
    if not scores:
        return PredictResponse(best_action="PASS", action_scores={})
    best = max(scores, key=scores.get)
    return PredictResponse(best_action=best, action_scores=scores)

@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    """
    Run N self-play games between two agent types.
    Writes game logs to data/game_logs/.
    """
    from training.self_play import run_games
    result = run_games(
        num_games=req.num_games,
        agent_a=req.agent_a,
        agent_b=req.agent_b,
        deck_a=req.deck_a,
        deck_b=req.deck_b,
        log_dir="../data/game_logs/"
    )
    return SimulateResponse(**result)

@app.get("/cards", response_model=list[CardResponse])
def get_cards(
    expansion: str | None = None,
    category: str | None = None,
    type: str | None = None,
    q: str | None = None
):
    """Return card list with optional filters."""
    cards = list(CARDS.values())
    if expansion:
        cards = [c for c in cards if c["expansion"] == expansion]
    if category:
        cards = [c for c in cards if c["category"].lower() == category.lower()]
    if type:
        cards = [c for c in cards if c.get("type", "").lower() == type.lower()]
    if q:
        cards = [c for c in cards if q.lower() in c["name"].lower()]
    return cards

@app.get("/cards/{card_id}", response_model=CardResponse)
def get_card(card_id: int):
    card = CARDS.get(str(card_id))
    if not card:
        raise HTTPException(404, f"Card {card_id} not found")
    return card

@app.get("/deck")
def get_deck():
    deck_path = Path("../deck/deck.json")
    if not deck_path.exists():
        return {"cards": [], "name": "empty"}
    return json.loads(deck_path.read_text())

@app.post("/deck")
def save_deck(req: DeckRequest):
    total = sum(c["count"] for c in req.cards)
    if total != 60:
        raise HTTPException(400, f"Deck must have exactly 60 cards, got {total}")
    deck_path = Path("../deck/deck.json")
    deck_path.write_text(json.dumps(req.dict(), indent=2))
    return {"saved": True, "total_cards": total}

@app.get("/logs")
def list_logs():
    logs = list_game_logs("../data/game_logs/")
    return {"logs": logs, "total": len(logs)}

@app.get("/logs/{log_id}")
def get_log(log_id: str):
    log_path = Path(f"../data/game_logs/{log_id}.json")
    if not log_path.exists():
        raise HTTPException(404, f"Log {log_id} not found")
    return json.loads(log_path.read_text())

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    logs = list_game_logs("../data/game_logs/", full=True)
    if not logs:
        return StatsResponse(
            total_games=0, win_rate=0.0, avg_turns=0.0,
            top_winning_actions=[], feature_importance=[]
        )
    wins = sum(1 for g in logs if g.get("winner") == "agent_a")
    avg_turns = sum(len(g.get("turns", [])) for g in logs) / len(logs)
    feat_imp = []
    if MODEL is not None:
        names = MODEL.model.get_booster().feature_names or []
        scores = MODEL.model.feature_importances_
        feat_imp = sorted(
            [{"name": n, "score": float(s)} for n, s in zip(names, scores)],
            key=lambda x: -x["score"]
        )[:15]
    return StatsResponse(
        total_games=len(logs),
        win_rate=round(wins / len(logs), 3),
        avg_turns=round(avg_turns, 1),
        top_winning_actions=[],
        feature_importance=feat_imp
    )
```

### 5.3 Startup

```bash
# Install
pip install fastapi uvicorn xgboost scikit-learn pandas joblib python-multipart

# Run (from project root)
uvicorn server.main:app --reload --port 8000

# API docs auto-generated at:
# http://localhost:8000/docs   (Swagger UI)
# http://localhost:8000/redoc  (ReDoc)
```

---

## 6. HTML UI

Three pages. All use plain HTML + CSS + vanilla JS — no framework, no build step. All data from FastAPI via `fetch()`.

---

### 6.1 `index.html` — Dashboard

**Purpose:** The home page. Shows live model status, recent win-rate, stats, and links to the other tools.

**Sections:**

| Section | Content |
|---|---|
| Header | Logo, nav links to Deck Builder / Match Viewer |
| Status Bar | API health, model loaded (green/red), cards loaded |
| Stats Cards | Total games, Win rate %, Avg turns/game |
| Feature Importance Chart | Horizontal bar chart of top 10 XGBoost features |
| Recent Logs Table | Last 10 game logs with outcome, turns, agents |
| Quick Simulate | Form: num_games, agent_a vs agent_b → calls `/simulate` |

**Key interactions:**
- On load: `GET /` → show model status; `GET /stats` → fill all stat cards and chart
- "Run Simulation" button → `POST /simulate` → refresh stats + log table
- Each log row links to `match_viewer.html?log=game_XXXX`

---

### 6.2 `deck_builder.html` — Deck Builder

**Purpose:** Browse all 1,266 cards, filter them, and assemble a 60-card deck. Save to server.

**Layout:**

```
┌────────────────────────┬──────────────────────────────┐
│  Filters (left panel)  │  Card Grid (right panel)     │
│                        │                              │
│  Search: [________]    │  ┌──────┐ ┌──────┐ ┌──────┐ │
│  Expansion: [v]        │  │Card  │ │Card  │ │Card  │ │
│  Category: [v]         │  │Image │ │Image │ │Image │ │
│  Type: [v]             │  │Name  │ │Name  │ │Name  │ │
│                        │  │Exp   │ │Exp   │ │Exp   │ │
│  ─────────────────     │  │[+][-]│ │[+][-]│ │[+][-]│ │
│  Deck (60 cards)       │  └──────┘ └──────┘ └──────┘ │
│                        │                              │
│  [Card name x2]        │  (grid continues...)         │
│  [Card name x4]        │                              │
│  ...                   │                              │
│  Total: 38/60          │                              │
│                        │                              │
│  [Save Deck]           │                              │
└────────────────────────┴──────────────────────────────┘
```

**Key interactions:**
- On load: `GET /cards` → render all cards in grid
- Filter inputs debounce → `GET /cards?expansion=X&type=Y&q=Z` → re-render grid
- `[+]` button on card → add to deck list (max 4 per card, 60 total)
- `[-]` button → remove from deck
- "Save Deck" → `POST /deck` with `{cards: [...], name: "my_deck"}` → success toast

---

### 6.3 `match_viewer.html` — Match Viewer

**Purpose:** Step through a saved game log turn by turn. Shows what the agent saw, what actions were available, and what win-probability score each action got.

**Layout:**

```
┌─────────────────────────────────────────────────────────┐
│  Game: game_0001 | Winner: agent_a | Turns: 24          │
│  [◀ Prev Turn]   Turn 8 / 24   [Next Turn ▶]           │
├─────────────────┬───────────────────────────────────────┤
│  Game State     │  Action Scores                        │
│                 │                                       │
│  MY SIDE        │  ATTACK_0          ████████ 0.82      │
│  Active: Chariz │  PLAY_ENERGY_2     ██       0.21      │
│  HP: 180/310    │  PLAY_SUPPORTER_5  ███      0.35      │
│  Energy: RRR    │  PASS              █        0.09      │
│                 │                                       │
│  Bench: 3 Pkmn  │  ▶ Chosen: ATTACK_0                  │
│                 │                                       │
│  OPPONENT SIDE  │                                       │
│  Active: Mirair │                                       │
│  HP: 90/280     │                                       │
│  Energy: LL     │                                       │
│                 │                                       │
│  Prizes: 3 / 4  │                                       │
└─────────────────┴───────────────────────────────────────┘
```

**Key interactions:**
- URL param `?log=game_0001` → `GET /logs/game_0001` → load full log
- Prev/Next buttons → render the current turn's game_state and action_scores
- Action bar chart: horizontal bars scaled by win-probability, chosen action highlighted

---

### 6.4 Shared Assets

**`assets/style.css`** — dark theme, minimal, monospace accents. Pokémon type colors as CSS variables:

```css
:root {
  --bg: #0f0f14;
  --surface: #1a1a24;
  --border: #2a2a3a;
  --text: #e8e8f0;
  --text-muted: #7a7a99;
  --accent: #6c63ff;
  --green: #4ade80;
  --red: #f87171;
  --yellow: #fbbf24;

  /* Pokémon type colors */
  --type-grass: #4ade80;
  --type-fire: #f97316;
  --type-water: #38bdf8;
  --type-lightning: #fbbf24;
  --type-psychic: #e879f9;
  --type-fighting: #fb923c;
  --type-darkness: #a78bfa;
  --type-metal: #94a3b8;
  --type-colorless: #cbd5e1;
}
```

**`assets/app.js`** — shared fetch utility:

```javascript
const API = "http://localhost:8000";

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

function showToast(msg, type = "success") {
  // type: "success" | "error"
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}
```

---

## 7. API Endpoint Reference

| Method | Path | Purpose | Request | Response |
|---|---|---|---|---|
| GET | `/` | Health check | — | `{status, model_loaded}` |
| POST | `/predict` | Score all legal actions | `PredictRequest` | `PredictResponse` |
| POST | `/simulate` | Run N self-play games | `SimulateRequest` | `SimulateResponse` |
| GET | `/cards` | List cards (filterable) | Query params | `CardResponse[]` |
| GET | `/cards/{id}` | Single card details | — | `CardResponse` |
| GET | `/deck` | Load current deck | — | `{cards, name}` |
| POST | `/deck` | Save deck | `DeckRequest` | `{saved, total_cards}` |
| GET | `/logs` | List game logs | — | `{logs, total}` |
| GET | `/logs/{id}` | Full game log | — | Game log JSON |
| GET | `/stats` | Aggregate stats + feature importance | — | `StatsResponse` |

---

## 8. Development Sequence

Build in this exact order — each step unblocks the next.

### Step 1 — Data Layer (Day 1)
- Parse `EN_Card_Data.csv` → build `card_lookup.json`
- Verify all 1,266 card IDs load correctly
- Write `load_card_lookup()` util

### Step 2 — Encoder (Day 1–2)
- Implement `encoder.py` with the 80-dim feature vector
- Unit test: `encode(mock_state, "ATTACK_0")` returns array of correct shape
- All values must be floats in [0, 1] or small integers — no raw strings

### Step 3 — Heuristic Agent (Day 2)
- Implement `heuristic.py`
- Test in simulator: does it complete a full match without crashing?
- This is the Phase 1 submission

### Step 4 — Self-Play Runner (Day 3)
- Implement `self_play.py`
- Run 50 games: heuristic vs random agent
- Confirm `game_logs/` populates with valid JSON

### Step 5 — Training Pipeline (Day 4)
- `feature_builder.py`: reads logs → builds X (N × 80), y (N,)
- `train.py`: train XGBoost, evaluate, save `model.pkl`
- Print feature importance — first sanity check on the model

### Step 6 — FastAPI Server (Day 5)
- Implement `server/main.py` with all routes
- Test with `curl` or Swagger UI at `/docs`
- Mount `/ui` static directory

### Step 7 — `index.html` (Day 6)
- Stats cards from `/stats`
- Feature importance chart (plain canvas bar chart, no library needed)
- Simulate form

### Step 8 — `deck_builder.html` (Day 7)
- Card grid from `/cards`
- Filter/search UI
- Add/remove cards, save deck

### Step 9 — `match_viewer.html` (Day 8)
- Turn navigation
- Action score bar chart
- Game state panel

### Step 10 — Integration + Hardening (Day 9–10)
- All pages link together
- Error states (API down, model not loaded, empty logs)
- Timeout guards in agent (every action call wrapped in try/except → PASS fallback)

---

## 9. Requirements

### `requirements.txt`

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
xgboost>=2.0.3
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
joblib>=1.3.2
python-multipart>=0.0.9
pydantic>=2.0.0
```

### Runtime
- Python 3.11+
- No GPU required (XGBoost CPU inference is fast enough for real-time turn decisions)
- Simulator provided by The Pokémon Company (separate install per competition instructions)

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Simulator observation schema undocumented | Build encoder against mock state dict first; adapt to real schema on integration |
| `model.pkl` too slow for 10-min match timeout | XGBoost inference is <1ms per action; not a concern |
| CORS issues between HTML files and FastAPI | CORS middleware set to `allow_origins=["*"]` in development |
| Card images not in dataset (only IDs) | UI shows card name + type badge instead of images; link to `ptcg.pokemon.co.jp` if available |
| Game log schema changes between self-play versions | Version field in log JSON; feature_builder checks version and skips incompatible logs |

---

## 11. Acceptance Criteria

| Layer | Criteria |
|---|---|
| ML Model | AUC-ROC ≥ 0.60 on held-out validation set; model.pkl loads in <1s |
| FastAPI | All 9 endpoints return correct status codes; Swagger UI at `/docs` works |
| index.html | Stats load on page open; simulate button runs and refreshes stats |
| deck_builder.html | All 1,266 cards display; filters work; deck saves and loads correctly |
| match_viewer.html | Any saved game log navigates turn-by-turn; action scores display as chart |
| End-to-end | Agent completes 10 simulated games without any unhandled exceptions |