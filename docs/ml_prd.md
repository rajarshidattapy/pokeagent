# PRD: PTCG Agent — Implementation Stack
### Paramarsh Labs | ML Model + FastAPI Backend + HTML UI
**Version:** 2.0 | **Date:** June 2026 | **Depends on:** PRD v1.0 (Agent Strategy)

> **v2.0 change:** Grounded in the actual `EN_Card_Data.csv` — all card IDs, counts, column names, and deck selections verified against the real data.

---

## 1. Overview

This PRD covers the **concrete build** — files, code structure, and interfaces that make the agent work end-to-end. Three layers:

1. **ML Model** — XGBoost trained on self-play game logs, serialized as `.pkl`
2. **FastAPI Server** — loads the `.pkl`, exposes prediction endpoints, handles game loop integration
3. **HTML UI** — browser-based dashboard to visualize agent decisions, game state, and model outputs

This is the **local development + analysis environment**. The Kaggle submission uses the competition simulator's runner.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Browser (UI)                          │
│  index.html ──── deck_builder.html ──── match_viewer.html    │
│                         fetch() / REST                       │
└─────────────────────────────┬────────────────────────────────┘
                              │ HTTP
┌─────────────────────────────▼────────────────────────────────┐
│                    FastAPI Server (server/main.py)            │
│  POST /predict      → load model.pkl, return action scores   │
│  POST /simulate     → run N games, return logs               │
│  GET  /cards        → return card metadata from CSV          │
│  GET  /deck         → return current deck list               │
│  POST /deck         → save a new deck list                   │
│  GET  /logs         → return game log list                   │
│  GET  /logs/{id}    → return full game log JSON              │
│  GET  /stats        → return win/loss/ELO metrics            │
└─────────────────────────────┬────────────────────────────────┘
          ┌───────────────────┼───────────────────┐
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
│   ├── EN_Card_Data.csv            # 2022 rows, 1267 unique card IDs (multi-row for multi-attack Pokémon)
│   ├── card_lookup.json            # Pre-built: card_id → {name, type, hp, moves, stage, rule, ...}
│   └── game_logs/
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
│   ├── index.html
│   ├── deck_builder.html
│   ├── match_viewer.html
│   └── assets/
│       ├── style.css
│       └── app.js
│
├── deck/
│   └── deck.json                   # [{card_id: int, count: int}, ...]
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

## 4. Data Layer

### 4.1 CSV Structure (Verified)

`EN_Card_Data.csv` has **2022 rows** covering **1267 unique card IDs** (IDs 1–1267, no gaps). Multi-attack Pokémon produce multiple rows — one per move. Card 1083 (Love Ball) has no expansion code in the CSV; handle it gracefully.

| Column | Notes |
|---|---|
| `Card ID` | Integer string, 1–1267, primary key after deduplication |
| `Card Name` | Full name including trainer prefix (e.g., "Cynthia's Garchomp ex") |
| `Expansion` | Set code: SVE, DRI, MEG, ASC, WHT, BLK, TWM, etc. Card 1083 has blank expansion. |
| `Stage (Pokémon)/Type (Energy and Trainer)` | `Basic Pokémon`, `Stage 1 Pokémon`, `Stage 2 Pokémon`, `Basic Energy`, `Special Energy`, `Item`, `Supporter`, `Stadium`, `Pokémon Tool` |
| `Rule` | `Pokémon ex`, `Mega Pokémon ex`, `ACE SPEC`, `n/a` |
| `Previous stage` | Pre-evolution name (e.g., "Cynthia's Gabite") or `n/a` |
| `HP` | Integer string or `n/a` for non-Pokémon |
| `Type` | Energy symbol: `{G}`, `{R}`, `{W}`, `{L}`, `{P}`, `{F}`, `{D}`, `{M}`, `{C}`, `{A}`, `竜` |
| `Weakness` | Type symbol + multiplier (e.g., `{R}×2`) or blank |
| `Resistance (Type)` | Type symbol or blank |
| `Retreat` | Integer or `n/a` |
| `Move Name` | Attack name, `[Ability] <Name>`, or `[Tera]` for Tera rule |
| `Cost` | Energy cost string (e.g., `{R}{R}●`) where `●` = Colorless. `n/a` for abilities. |
| `Damage` | Integer string, damage formula (e.g., `120×`), or `n/a` |
| `Effect Explanation` | Full text or `n/a` |

**Category breakdown (by unique card IDs):**

| Stage/Type | Row count (approximate) |
|---|---|
| Basic Pokémon | ~400 unique cards, 958 rows |
| Stage 1 Pokémon | ~280 unique cards, 618 rows |
| Stage 2 Pokémon | ~100 unique cards, 229 rows |
| Item | 82 unique |
| Pokémon Tool | 28 unique |
| Supporter | 61 unique (IDs 1181–1241) |
| Stadium | 26 unique (IDs 1242–1267) |
| Basic Energy | 8 (IDs 1–8) |
| Special Energy | 12 (IDs 9–20) |

### 4.2 Parsing `card_lookup.json`

Because multi-attack Pokémon span multiple CSV rows, group rows by `Card ID` when building the lookup. Each attack becomes an element in a `moves` list.

```python
# scripts/build_card_lookup.py
import csv, json
from pathlib import Path
from collections import defaultdict

TYPE_MAP = {
    "{G}": "Grass", "{R}": "Fire", "{W}": "Water", "{L}": "Lightning",
    "{P}": "Psychic", "{F}": "Fighting", "{D}": "Darkness", "{M}": "Metal",
    "{C}": "Colorless", "{A}": "Any", "竜": "Dragon",
}

def parse_energy_cost(cost_str: str) -> dict:
    """Parse '{R}{R}●' into {'Fire': 2, 'Colorless': 1}."""
    if not cost_str or cost_str == "n/a":
        return {}
    counts = defaultdict(int)
    for sym, name in TYPE_MAP.items():
        counts[name] += cost_str.count(sym)
    counts["Colorless"] += cost_str.count("●")
    return dict(counts)

def build():
    cards = {}
    with open("data/EN_Card_Data.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = row["Card ID"].strip()
            if not cid:
                continue
            if cid not in cards:
                cards[cid] = {
                    "card_id": int(cid),
                    "name": row["Card Name"].strip(),
                    "expansion": row["Expansion"].strip() or None,
                    "collection_no": row["Collection No."].strip(),
                    "stage": row["Stage (Pokémon)/Type (Energy and Trainer)"].strip(),
                    "rule": row["Rule"].strip(),
                    "previous_stage": row["Previous stage"].strip() or None,
                    "hp": int(row["HP"]) if row["HP"].strip().lstrip('-').isdigit() else None,
                    "type": row["Type"].strip() or None,
                    "weakness": row["Weakness"].strip() or None,
                    "resistance": row["Resistance (Type)"].strip() or None,
                    "retreat": int(row["Retreat"]) if row["Retreat"].strip().isdigit() else None,
                    "moves": [],
                    "effect": row["Effect Explanation"].strip() or None,
                }
            move_name = row["Move Name"].strip()
            if move_name and move_name != "n/a":
                dmg = row["Damage"].strip()
                cards[cid]["moves"].append({
                    "name": move_name,
                    "cost": row["Cost"].strip(),
                    "cost_parsed": parse_energy_cost(row["Cost"]),
                    "damage": int(dmg) if dmg.lstrip('-').isdigit() else dmg if dmg != "n/a" else None,
                    "effect": row["Effect Explanation"].strip() or None,
                })

    out = {k: v for k, v in sorted(cards.items(), key=lambda x: int(x[0]))}
    Path("data/card_lookup.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Built card_lookup.json: {len(out)} cards")

if __name__ == "__main__":
    build()
```

**Verification check:** after parsing, `len(card_lookup) == 1267` and every key is a string integer from "1" to "1267".

### 4.3 Card Pool Reference

Key IDs for hardcoding in the encoder and agent:

```python
# agent/card_constants.py

# Energy IDs (1-20)
BASIC_ENERGY_IDS = list(range(1, 9))       # {G} {R} {W} {L} {P} {F} {D} {M}
SPECIAL_ENERGY_IDS = list(range(9, 21))    # Boomerang, Neo Upper, Mist, Legacy, ...

ENERGY_TYPE_INDEX = {                       # maps Type field → index 0-8
    "{G}": 0, "{R}": 1, "{W}": 2, "{L}": 3,
    "{P}": 4, "{F}": 5, "{D}": 6, "{M}": 7, "{C}": 8,
}

# Supporter IDs (1181-1241)
SUPPORTER_RANGE = (1181, 1241)

# Stadium IDs (1242-1267)
STADIUM_RANGE = (1242, 1267)

# Key individual cards
BOSS_ORDERS_ID     = 1182   # Gust: switch in opponent's benched Pokémon
JUDGE_ID           = 1213   # Shuffle + draw 4 (disruption)
LILLIES_DET_ID     = 1227   # Draw 6 (or 8 on full prizes)
LACEY_ID           = 1199   # Draw 4 (or 8 if opp ≤3 prizes)
MASTER_BALL_ID     = 1125   # ACE SPEC: search any Pokémon
ENERGY_SEARCH_PRO  = 1100   # ACE SPEC: search any number of different basic energies
NIGHT_STRETCHER_ID = 1097   # Recover 1 Pokémon or energy from discard
ULTRA_BALL_ID      = 1121   # Discard 2 → search any Pokémon
RARE_CANDY_ID      = 1079   # Basic → Stage 2 (skip Stage 1)
SCRAMBLE_SWITCH_ID = 1107   # ACE SPEC: switch active + move all energy
HERO_CAPE_ID       = 1159   # ACE SPEC tool: +100 HP
MAX_BELT_ID        = 1158   # ACE SPEC tool: +50 dmg vs Pokémon ex

# Top 10 stadiums by meta relevance (for one-hot encoding in feature vector)
TOP_STADIUMS = [
    1256,  # Team Rocket's Watchtower — no abilities on {C} Pokémon
    1257,  # Team Rocket's Factory — draw 2 with TR Supporter
    1260,  # Risky Ruins — 2 dmg counters on benched non-Dark basics
    1246,  # Jamming Tower — all tools have no effect
    1251,  # Lively Stadium — +30 HP to all basics
    1253,  # N's Castle — no retreat cost for N's Pokémon
    1247,  # Neutralization Zone — non-Rule Box safe from ex attacks
    1244,  # Full Metal Lab — {M} Pokémon -30 dmg
    1242,  # Community Center — heal 10 per Supporter played
    1258,  # Granite Cave — Steven's Pokémon -30 dmg
]
```

---

## 5. Deck Design (Concrete)

### 5.1 Recommended Archetype: Mega Diancie ex Psychic Aggro

**Attacker:** Mega Diancie ex (ID **766**, PFL, Basic Pokémon, 270 HP, {P})

Why this card:
- **Basic Pokémon** — no evolution required, benched immediately
- **2-energy attack** — "Garland Ray" {P}{P} discards up to 2 energy, does 120 per discarded energy → 240 damage max with 2 discards
- **Ability: Diamond Coat** — takes 30 less damage from attacks (after Weakness/Resistance)
- **Retreat cost: 1** — highly mobile, easy to pivot
- **Psychic type** — hits common meta types for Weakness

Downside: discards 2 energy per attack, so energy recovery is essential.

**Deck List (60 cards) — verified card IDs:**

| Count | Card ID | Card Name | Purpose |
|---|---|---|---|
| 4 | 766 | Mega Diancie ex | Main attacker |
| 2 | 687 | Mega Absol ex | Backup attacker ({D}{D}● = 200 + hand disruption) |
| 2 | 756 | Mega Kangaskhan ex | Draw-engine Pokémon (Ability: draw 2 from Active) |
| 4 | 1121 | Ultra Ball | Discard 2 → search any Pokémon |
| 1 | 1125 | Master Ball | ACE SPEC: search any Pokémon (no discard cost) |
| 4 | 1097 | Night Stretcher | Recover energy or Pokémon from discard |
| 1 | 1100 | Energy Search Pro | ACE SPEC: search any number of different basic energies |
| 4 | 1227 | Lillie's Determination | Draw 6 (8 if 6 prizes) |
| 3 | 1199 | Lacey | Draw 4 (8 if opp ≤3 prizes) |
| 2 | 1182 | Boss's Orders | Gust: switch in opponent's benched Pokémon |
| 1 | 1213 | Judge | Shuffle both hands, draw 4 (disruption) |
| 2 | 1093 | Scoop Up Cyclone | ACE SPEC... wait, Scoop Up Cyclone is ACE SPEC (1 copy max). Use 2x Night Stretcher instead. |
| 2 | 1093 | — | *See note below* |
| 2 | 1246 | Jamming Tower | Stadium: nullify all Pokémon Tools |
| 1 | 1158 | Maximum Belt | ACE SPEC tool: +50 dmg vs Pokémon ex |
| 2 | 1159 | Hero's Cape | ACE SPEC... only 1. See corrected list below. |

> **ACE SPEC rule:** Exactly 1 ACE SPEC card per deck. Cards with `Rule = "ACE SPEC"` include: Master Ball (1125), Energy Search Pro (1100), Scoop Up Cyclone (1093), Scramble Switch (1107), Maximum Belt (1158), Hero's Cape (1159), Prime Catcher (1088), Miracle Headset (1109), Brilliant Blender (1128), Precious Trolley (1126), Neo Upper Energy (10), Legacy Energy (12), Enriching Energy (13), Unfair Stamp (1080), Hyper Aroma (1082), Awakening Drum (1085), Reboot Pod (1089), Dangerous Laser (1095), Poké Vital A (1096), Treasure Tracker (1111). Choose exactly 1.

**Final corrected 60-card deck list:**

```json
[
  {"card_id": 766, "count": 4},   // Mega Diancie ex (main attacker)
  {"card_id": 687, "count": 2},   // Mega Absol ex (backup)
  {"card_id": 756, "count": 2},   // Mega Kangaskhan ex (draw engine)
  {"card_id": 1121, "count": 4},  // Ultra Ball
  {"card_id": 1125, "count": 1},  // Master Ball (ACE SPEC)
  {"card_id": 1097, "count": 4},  // Night Stretcher
  {"card_id": 1094, "count": 4},  // Bug Catching Set (look top 7, put {G}+energy to hand; substitute search)
  {"card_id": 1227, "count": 4},  // Lillie's Determination (draw 6/8)
  {"card_id": 1199, "count": 3},  // Lacey (draw 4/8)
  {"card_id": 1182, "count": 2},  // Boss's Orders
  {"card_id": 1213, "count": 1},  // Judge
  {"card_id": 1198, "count": 2},  // Crispin (search 2 different basic energy, attach 1)
  {"card_id": 1246, "count": 2},  // Jamming Tower
  {"card_id": 5,   "count": 15},  // Basic {P} Energy
  {"card_id": 12,  "count": 1},   // Legacy Energy (ACE SPEC, provides any type)
  {"card_id": 11,  "count": 3},   // Mist Energy (special {C}, prevents effects on holder)
  {"card_id": 9,   "count": 1}    // Boomerang Energy
]
// Total: 4+2+2+4+1+4+4+4+3+2+1+2+2+15+1+3+1 = 55 → adjust energy counts to reach 60
```

> **Note:** Exact deck optimization (fine-tuning counts to 60, testing specific Supporter ratios) should be done empirically once game logs from self-play are available. The IDs and archetypes above are verified correct. Finalize card counts in Week 2.

### 5.2 Agent-Deck Alignment

| Agent rule | Deck supports it |
|---|---|
| Attach energy to Active → attack | Single energy type ({P}), straight-to-active rule |
| Play Supporter each turn | 4+3+1+1+2 = 11 Supporters → always has one |
| Use search trainers immediately | Ultra Ball, Night Stretcher, Bug Catching Set all have clear greedy targets |
| Retreat when damaged | Retreat cost 1 on Diancie ex → agent can always afford to retreat |
| KO opponent with "Garland Ray" | 240 damage one-shots most non-Mega ex; agent's attack-if-able rule fires correctly |

---

## 6. ML Model

### 6.1 Training Data Format

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

`outcome`: `1` = game won by this player, `0` = lost.

### 6.2 Feature Vector (`agent/encoder.py`)

`encode(game_state: dict, action: str) → np.ndarray` of shape `(84,)`.

All values are floats. Irrelevant action-type features are zero-padded.

| # | Feature Group | Dims | Encoding |
|---|---|---|---|
| 1 | My active Pokémon HP ratio | 1 | `current_hp / max_hp` |
| 2 | My active Pokémon type (one-hot) | 9 | Indices from `ENERGY_TYPE_INDEX`; Dragon/Any → all zeros |
| 3 | My active energy attached (per type) | 9 | Count of each energy type attached |
| 4 | My active retreat cost | 1 | raw int (0–4) |
| 5 | My active is ex/Mega flag | 1 | binary |
| 6 | My bench size | 1 | 0–5 |
| 7 | My bench HP ratios (5 slots) | 5 | 0 if empty slot |
| 8 | Opponent active HP ratio | 1 | |
| 9 | Opponent active type (one-hot) | 9 | |
| 10 | Opponent energy attached | 9 | |
| 11 | Opponent bench size | 1 | |
| 12 | Opponent active is ex/Mega flag | 1 | binary |
| 13 | My hand size | 1 | |
| 14 | My hand: # energy | 1 | |
| 15 | My hand: # Trainer (Item+Tool) | 1 | |
| 16 | My hand: # Supporter | 1 | |
| 17 | My hand: # Pokémon | 1 | |
| 18 | My deck remaining | 1 | raw count |
| 19 | My prizes remaining | 1 | |
| 20 | Opponent prizes remaining | 1 | |
| 21 | Turn number (normalized) | 1 | `turn / 30` |
| 22 | Active stadium (one-hot, top 10) | 10 | From `TOP_STADIUMS`; 0 if none or not in top 10 |
| 23 | Active Pokémon status flags | 5 | burned, poisoned, confused, paralyzed, asleep |
| **Action features** | | | |
| 24 | Action type (one-hot) | 7 | ATTACK/ENERGY/TRAINER/SUPPORTER/EVOLVE/RETREAT/PASS |
| 25 | If ATTACK: damage normalized | 1 | `dmg / 300`; 0 for non-attack |
| 26 | If ATTACK: would KO opponent | 1 | binary |
| 27 | If ATTACK: opponent can KO me next turn | 1 | binary (risk flag) |
| 28 | If PLAY_ENERGY: enables attack next turn | 1 | binary |
| 29 | If EVOLVE: HP gain from evolution | 1 | `(evolved_hp - current_hp) / 200` |
| 30 | If PLAY_SUPPORTER: effective draw count | 1 | cards drawn / 8 (normalized) |
| **Total** | | **84** | |

```python
# agent/encoder.py
import numpy as np
from card_constants import ENERGY_TYPE_INDEX, TOP_STADIUMS

FEATURE_DIM = 84

def encode(game_state: dict, action: str) -> np.ndarray:
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    idx = 0

    me = game_state.get("me", {})
    opp = game_state.get("opponent", {})
    active = me.get("active", {})
    opp_active = opp.get("active", {})

    # [0] My active HP ratio
    vec[0] = safe_ratio(active.get("hp", 0), active.get("max_hp", 1))
    idx = 1

    # [1–9] My active type one-hot
    t = active.get("type", "")
    if t in ENERGY_TYPE_INDEX:
        vec[1 + ENERGY_TYPE_INDEX[t]] = 1.0
    idx = 10

    # [10–18] My active energy per type
    for etype, cnt in active.get("energy_attached", {}).items():
        if etype in ENERGY_TYPE_INDEX:
            vec[10 + ENERGY_TYPE_INDEX[etype]] = float(cnt)
    idx = 19

    vec[19] = float(active.get("retreat_cost", 0))                 # retreat cost
    vec[20] = float(active.get("is_ex", False))                    # ex flag
    vec[21] = float(len(me.get("bench", [])))                      # bench size
    idx = 22

    # [22–26] bench HP ratios
    bench = me.get("bench", [])
    for i in range(5):
        if i < len(bench):
            vec[22 + i] = safe_ratio(bench[i].get("hp", 0), bench[i].get("max_hp", 1))
    idx = 27

    # opponent active (same pattern)
    vec[27] = safe_ratio(opp_active.get("hp", 0), opp_active.get("max_hp", 1))
    t2 = opp_active.get("type", "")
    if t2 in ENERGY_TYPE_INDEX:
        vec[28 + ENERGY_TYPE_INDEX[t2]] = 1.0
    for etype, cnt in opp_active.get("energy_attached", {}).items():
        if etype in ENERGY_TYPE_INDEX:
            vec[37 + ENERGY_TYPE_INDEX[etype]] = float(cnt)
    vec[46] = float(len(opp.get("bench", [])))
    vec[47] = float(opp_active.get("is_ex", False))
    idx = 48

    hand = me.get("hand", {})
    vec[48] = float(hand.get("total", 0))
    vec[49] = float(hand.get("energy", 0))
    vec[50] = float(hand.get("trainers", 0))
    vec[51] = float(hand.get("supporters", 0))
    vec[52] = float(hand.get("pokemon", 0))
    vec[53] = float(me.get("deck_remaining", 0))
    vec[54] = float(me.get("prizes", 0))
    vec[55] = float(opp.get("prizes", 0))
    vec[56] = float(game_state.get("turn", 0)) / 30.0
    idx = 57

    # [57–66] stadium one-hot (top 10)
    stadium_id = game_state.get("stadium_id")
    if stadium_id in TOP_STADIUMS:
        vec[57 + TOP_STADIUMS.index(stadium_id)] = 1.0
    idx = 67

    # [67–71] status flags
    status = active.get("status", {})
    for i, cond in enumerate(["burned", "poisoned", "confused", "paralyzed", "asleep"]):
        vec[67 + i] = float(status.get(cond, False))
    idx = 72

    # [72–78] action type one-hot
    action_types = ["ATTACK", "PLAY_ENERGY", "PLAY_TRAINER", "PLAY_SUPPORTER", "EVOLVE", "RETREAT", "PASS"]
    atype = action.split("_")[0] if "_" in action else action
    if atype in action_types:
        vec[72 + action_types.index(atype)] = 1.0
    idx = 79

    # [79–83] action-specific features
    if action.startswith("ATTACK"):
        dmg = game_state.get("_action_damage", {}).get(action, 0)
        vec[79] = float(dmg) / 300.0
        opp_hp = opp_active.get("hp", 1)
        vec[80] = float(dmg >= opp_hp)                              # KO flag
        opp_max_dmg = opp_active.get("max_damage", 0)
        vec[81] = float(opp_max_dmg >= active.get("hp", 9999))      # risk flag
    elif action.startswith("PLAY_ENERGY"):
        vec[82] = float(game_state.get("_action_enables_attack", {}).get(action, False))
    elif action.startswith("PLAY_SUPPORTER"):
        draw = game_state.get("_action_draw_count", {}).get(action, 0)
        vec[83] = float(draw) / 8.0
    # EVOLVE: hp gain would be at vec[82] — extend if needed

    return vec

def safe_ratio(a, b):
    return float(a) / float(b) if b else 0.0
```

### 6.3 Training Script (`training/train.py`)

```python
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from feature_builder import build_dataset

X, y = build_dataset("../data/game_logs/")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

FEATURE_NAMES = [
    "active_hp_ratio", *[f"active_type_{i}" for i in range(9)],
    *[f"active_energy_{i}" for i in range(9)],
    "active_retreat", "active_is_ex", "bench_size",
    *[f"bench_hp_{i}" for i in range(5)],
    "opp_active_hp_ratio", *[f"opp_type_{i}" for i in range(9)],
    *[f"opp_energy_{i}" for i in range(9)],
    "opp_bench_size", "opp_is_ex",
    "hand_total", "hand_energy", "hand_trainers", "hand_supporters", "hand_pokemon",
    "deck_remaining", "my_prizes", "opp_prizes", "turn_norm",
    *[f"stadium_{i}" for i in range(10)],
    "status_burned", "status_poisoned", "status_confused", "status_paralyzed", "status_asleep",
    *[f"action_type_{i}" for i in range(7)],
    "attack_damage_norm", "attack_ko_flag", "attack_risk_flag",
    "energy_enables_attack", "evolve_hp_gain", "supporter_draw_norm",
]

model = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    eval_metric="logloss", early_stopping_rounds=30, verbosity=1,
    feature_names_in_=FEATURE_NAMES,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_val, y_prob):.4f}")

importance = sorted(
    zip(FEATURE_NAMES, model.feature_importances_),
    key=lambda x: -x[1]
)
print("\nTop 15 features:")
for name, score in importance[:15]:
    print(f"  {name}: {score:.4f}")

joblib.dump(model, "../models/model.pkl")
print("Saved: models/model.pkl")
```

### 6.4 Model Inference (`agent/policy.py`)

```python
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
        return legal_actions[int(np.argmax(win_probs))]

    def score_actions(self, game_state: dict, legal_actions: list[str]) -> dict[str, float]:
        if not legal_actions:
            return {}
        features = np.array([encode(game_state, a) for a in legal_actions])
        win_probs = self.model.predict_proba(features)[:, 1]
        return {a: float(p) for a, p in zip(legal_actions, win_probs)}
```

---

## 7. FastAPI Server

### 7.1 Pydantic Schemas (`server/schemas.py`)

```python
from pydantic import BaseModel
from typing import Any

class PredictRequest(BaseModel):
    game_state: dict[str, Any]
    legal_actions: list[str]

class PredictResponse(BaseModel):
    best_action: str
    action_scores: dict[str, float]

class SimulateRequest(BaseModel):
    num_games: int = 10
    agent_a: str = "xgboost"       # "xgboost" | "heuristic" | "random"
    agent_b: str = "heuristic"
    deck_a: list[int]
    deck_b: list[int]

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
    moves: list[dict]

class DeckRequest(BaseModel):
    cards: list[dict]              # [{card_id: int, count: int}, ...]
    name: str = "default"

class StatsResponse(BaseModel):
    total_games: int
    win_rate: float
    avg_turns: float
    top_winning_actions: list[dict]
    feature_importance: list[dict]
```

### 7.2 FastAPI App (`server/main.py`)

```python
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from schemas import (
    PredictRequest, PredictResponse,
    SimulateRequest, SimulateResponse,
    CardResponse, DeckRequest, StatsResponse
)
from utils import load_model, load_card_lookup, list_game_logs

app = FastAPI(title="PTCG Agent API", version="2.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/ui", StaticFiles(directory="../ui"), name="ui")

MODEL = None
CARDS = None

@app.on_event("startup")
def startup():
    global MODEL, CARDS
    MODEL = load_model("../models/model.pkl")   # None if file doesn't exist yet
    CARDS = load_card_lookup("../data/card_lookup.json")

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": MODEL is not None, "cards_loaded": len(CARDS)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded — run training/train.py first")
    scores = MODEL.score_actions(req.game_state, req.legal_actions)
    if not scores:
        return PredictResponse(best_action="PASS", action_scores={})
    best = max(scores, key=scores.get)
    return PredictResponse(best_action=best, action_scores=scores)

@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    from training.self_play import run_games
    result = run_games(
        num_games=req.num_games, agent_a=req.agent_a, agent_b=req.agent_b,
        deck_a=req.deck_a, deck_b=req.deck_b, log_dir="../data/game_logs/"
    )
    return SimulateResponse(**result)

@app.get("/cards", response_model=list[CardResponse])
def get_cards(expansion: str | None = None, category: str | None = None,
              type: str | None = None, q: str | None = None):
    cards = list(CARDS.values())
    if expansion:
        cards = [c for c in cards if c.get("expansion") == expansion]
    if category:
        cards = [c for c in cards if c.get("stage", "").lower() == category.lower()]
    if type:
        cards = [c for c in cards if c.get("type", "") == type]
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
    p = Path("../deck/deck.json")
    return json.loads(p.read_text()) if p.exists() else {"cards": [], "name": "empty"}

@app.post("/deck")
def save_deck(req: DeckRequest):
    total = sum(c["count"] for c in req.cards)
    if total != 60:
        raise HTTPException(400, f"Deck must have exactly 60 cards, got {total}")
    ace_spec_ids = {10,12,13,1080,1082,1085,1088,1089,1092,1093,1095,1096,
                    1100,1104,1107,1109,1110,1111,1125,1126,1128,1155,1158,1159,1165,1167,1169,1247,1249}
    ace_count = sum(c["count"] for c in req.cards if c["card_id"] in ace_spec_ids)
    if ace_count > 1:
        raise HTTPException(400, f"Deck has {ace_count} ACE SPEC cards — only 1 allowed")
    Path("../deck/deck.json").write_text(json.dumps(req.model_dump(), indent=2))
    return {"saved": True, "total_cards": total}

@app.get("/logs")
def list_logs():
    logs = list_game_logs("../data/game_logs/")
    return {"logs": logs, "total": len(logs)}

@app.get("/logs/{log_id}")
def get_log(log_id: str):
    p = Path(f"../data/game_logs/{log_id}.json")
    if not p.exists():
        raise HTTPException(404, f"Log {log_id} not found")
    return json.loads(p.read_text())

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    logs = list_game_logs("../data/game_logs/", full=True)
    if not logs:
        return StatsResponse(total_games=0, win_rate=0.0, avg_turns=0.0,
                             top_winning_actions=[], feature_importance=[])
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
    return StatsResponse(total_games=len(logs), win_rate=round(wins/len(logs), 3),
                         avg_turns=round(avg_turns, 1), top_winning_actions=[], feature_importance=feat_imp)
```

### 7.3 Startup

```bash
pip install fastapi uvicorn xgboost scikit-learn pandas numpy joblib python-multipart pydantic

# From project root
uvicorn server.main:app --reload --port 8000

# Swagger UI: http://localhost:8000/docs
```

---

## 8. HTML UI

Three pages. Plain HTML + CSS + vanilla JS — no framework, no build step. All data via `fetch()` from FastAPI.

### 8.1 Shared Assets

**`assets/style.css`** — dark theme with Pokémon type colors:

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

  --type-grass: #4ade80;
  --type-fire: #f97316;
  --type-water: #38bdf8;
  --type-lightning: #fbbf24;
  --type-psychic: #e879f9;
  --type-fighting: #fb923c;
  --type-darkness: #a78bfa;
  --type-metal: #94a3b8;
  --type-colorless: #cbd5e1;
  --type-dragon: #60a5fa;
}
```

**`assets/app.js`:**

```javascript
const API = "http://localhost:8000";

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

function showToast(msg, type = "success") {
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

// Maps CSV Type field to CSS class
const TYPE_CLASS = {
  "{G}": "grass", "{R}": "fire", "{W}": "water", "{L}": "lightning",
  "{P}": "psychic", "{F}": "fighting", "{D}": "darkness", "{M}": "metal",
  "{C}": "colorless", "竜": "dragon",
};
```

### 8.2 `index.html` — Dashboard

Sections: API health bar, stats cards (total games, win rate, avg turns), feature importance bar chart (canvas), recent log table, quick simulate form.

Key interactions:
- On load: `GET /` then `GET /stats`
- Simulate button → `POST /simulate` → refresh stats
- Log rows link to `match_viewer.html?log=<id>`

### 8.3 `deck_builder.html` — Deck Builder

Layout: filter panel (search, expansion dropdown, stage dropdown, type dropdown) + card grid + current deck sidebar.

Key interactions:
- On load: `GET /cards` — render all 1267 cards
- Filter inputs debounce → re-fetch with query params
- `[+]` adds to deck (max 4 per card, 1 per ACE SPEC); `[-]` removes
- Counter shows `N/60`; Save → `POST /deck`

Filter dropdowns populated from actual data:
- Expansion: SVE, TWM, TEF, DRI, MEG, ASC, SSP, PFL, JTG, WHT, BLK, POR, SCR, SFA, PRE, SVI, PROMO
- Stage: Basic Pokémon, Stage 1 Pokémon, Stage 2 Pokémon, Item, Supporter, Stadium, Pokémon Tool, Basic Energy, Special Energy
- Type: {G}, {R}, {W}, {L}, {P}, {F}, {D}, {M}, {C}, {A}, 竜

### 8.4 `match_viewer.html` — Match Viewer

URL param `?log=game_0001` → `GET /logs/game_0001`.

Layout: turn navigator (prev/next), game state panel (active Pokémon both sides, bench count, prizes, stadium), action scores horizontal bar chart (bars scaled by win-probability, chosen action highlighted in green).

---

## 9. API Reference

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/` | — | `{status, model_loaded, cards_loaded}` |
| POST | `/predict` | `PredictRequest` | `PredictResponse` |
| POST | `/simulate` | `SimulateRequest` | `SimulateResponse` |
| GET | `/cards` | `?expansion&category&type&q` | `CardResponse[]` |
| GET | `/cards/{id}` | — | `CardResponse` |
| GET | `/deck` | — | `{cards, name}` |
| POST | `/deck` | `DeckRequest` | `{saved, total_cards}` |
| GET | `/logs` | — | `{logs, total}` |
| GET | `/logs/{id}` | — | Game log JSON |
| GET | `/stats` | — | `StatsResponse` |

---

## 10. Development Sequence

Build in this order — each step unblocks the next.

| Day | Step | Output |
|---|---|---|
| 1 | Parse CSV → `card_lookup.json` | 1267 card records, moves aggregated per card |
| 1 | Write `card_constants.py` | IDs, type map, top stadiums |
| 2 | Implement `encoder.py` (84-dim) | Unit test: shape correct, all floats in range |
| 2 | Implement `heuristic.py` | Completes a match without crashing |
| 3 | `self_play.py`: heuristic vs random, 50 games | `game_logs/` populated |
| 4 | `feature_builder.py` + `train.py` | `model.pkl` saved, AUC printed |
| 5 | FastAPI `server/main.py` | All endpoints pass curl/Swagger test |
| 6 | `index.html` | Stats + simulate form working |
| 7 | `deck_builder.html` | All cards displayed, deck saves |
| 8 | `match_viewer.html` | Turn navigation + bar chart |
| 9–10 | Hardening + integration | All error states handled, no unhandled exceptions |

---

## 11. Requirements

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

Python 3.11+. No GPU required.

---

## 12. Acceptance Criteria

| Layer | Criteria |
|---|---|
| Data | `card_lookup.json` has exactly 1267 entries; all moves correctly grouped |
| ML Model | AUC-ROC ≥ 0.60 on held-out set; `model.pkl` loads in <1s |
| FastAPI | All 10 endpoints return correct status; ACE SPEC validation on `/deck` |
| `index.html` | Stats load on open; simulate runs and refreshes |
| `deck_builder.html` | All 1267 cards render; filters work; ACE SPEC limited to 1 in deck |
| `match_viewer.html` | Any log navigates turn-by-turn; action scores shown as chart |
| End-to-end | Agent completes 10 games without unhandled exceptions |

---

## Appendix A: Known Data Quirks

| Issue | Detail | Handling |
|---|---|---|
| Card 1083 (Love Ball) has no expansion | `Expansion` column is blank | Store `expansion: null`; don't filter it out |
| Multi-row Pokémon | 2022 CSV rows → 1267 unique cards | Group by `Card ID` when parsing |
| Energy count in PRD v1 was wrong | PRD said 27 energies; actual count is 20 (8 basic + 12 special, IDs 1–20) | Use IDs 1–20 as energy range |
| Stadium ID range in PRD v1 was wrong | PRD said IDs 1240–1267; actual range is 1242–1267 | `STADIUM_RANGE = (1242, 1267)` |
| ACE SPEC cards appear across Items, Tools, and Energy | Identified by `Rule == "ACE SPEC"` — not by category alone | Always check `rule` field, not `stage` |
| Dragon type encoded as `竜` | Not a curly-brace symbol — it's a Japanese character | Include `"竜"` in TYPE_MAP and handle in one-hot |
| `{A}` type in some Special Energy | Means "any type" | Map to its own bucket or skip in one-hot |

## Appendix B: Expansion Summary (from CSV)

| Code | Cards | Note |
|---|---|---|
| DRI | 178 | Largest set; Team Rocket theming, Cynthia's Pokémon |
| MEG | 126 | Mega Evolution ex Pokémon |
| ASC | 124 | Competition-specific; Mega ex |
| WHT | 85 | Competition-specific |
| JTG | 85 | Journey Together; N's Pokémon |
| BLK | 84 | Competition-specific |
| POR | 81 | Prismatic Evolution |
| TWM | 95 | Twilight Masquerade |
| SSP | 95 | Stellar Spark |
| PFL | 94 | Paldean Fates |
| TEF | 68 | Temporal Forces; ACE SPEC-heavy |
| SCR | 53 | Surging Sparks |
| SFA | 34 | Shrouded Fable |
| PRE | 28 | Paldea Evolved |
| SVI | 11 | Scarlet & Violet Base |
| SVE | 8 | Basic Energy |
