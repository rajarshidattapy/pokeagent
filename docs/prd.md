# PRD: Pokémon TCG AI Battle Agent
### Paramarsh Labs — PTCG AI Battle Challenge (Kaggle / The Pokémon Company)
**Version:** 1.0 | **Date:** June 2026 | **Track:** Simulation + Strategy Category

---

## 1. Overview

### 1.1 Problem Statement

The Pokémon TCG AI Battle Challenge requires teams to submit an AI agent capable of playing the Pokémon Trading Card Game against other agents in automated matches. Unlike Chess or Go, the PTCG is a game of **imperfect information** — opponents' hands are hidden, card draws are random, and board states evolve through complex multi-card interactions across a pool of 1,266 cards across 26 expansions.

The goal is not to solve the game — it is to build an agent that wins more than it loses in a tournament ladder by making consistently good decisions under uncertainty.

### 1.2 Objective

Build a **simple, pragmatic ML-based agent** that:

1. Encodes the game state into a structured feature vector
2. Uses a trained policy model to select actions at each decision point
3. Is stable enough to complete matches without crashing (stability is a scored dimension in the Strategy Category)
4. Has a documented strategic logic that can be written up as the required Strategy Report

### 1.3 Scope

| In Scope | Out of Scope |
|---|---|
| Game state featurization | Full game engine reimplementation |
| Supervised or RL policy model | Neural architecture search |
| Deck selection (fixed 60-card list) | Dynamic deck-switching between matches |
| Single-agent submission to Kaggle | Multi-agent ensemble or federation |
| Strategy Category report | Japanese language submission |

### 1.4 Deliverables

| # | Deliverable | Deadline |
|---|---|---|
| 1 | Working agent submission to Simulation Category | Aug 17, 2026 |
| 2 | Strategy Report (PDF) | Sep 14, 2026 |
| 3 | Codebase (clean, documented) | Sep 14, 2026 |

---

## 2. Competition Context

### 2.1 Format Rules

- Each deck: **60 cards**, drawn from the ~1,266 card Standard pool (IDs 1–1267)
- Matches are **turn-based**, 10-minute limit per match; timeout = loss
- Agents receive the **game state as a structured observation** from the Pokémon-provided simulator
- Agents must **return a legal action** within the simulator's time budget
- Leaderboard is ELO/ladder-based from automated matches

### 2.2 Card Pool Summary

| Category | Count |
|---|---|
| Total cards | 1,266 |
| ex Pokémon | ~150 |
| Energy cards | 27 (8 basic, 19 special) |
| Trainer/Supporter/Stadium | ~350 (est.) |
| Basic Pokémon (non-ex) | ~400 (est.) |
| Stage 1 / Stage 2 Pokémon | ~350 (est.) |

**Key expansions:** Destined Rivals (DRI, 178 cards), Mega Evolution (MEG, 126), Ascent (ASC, 124), Twilight Masquerade (TWM, 95), Stellar Spark (SSP, 95). Note: DRI/MEG/ASC/WHT/BLK are competition-specific sets — no real-world meta tier lists exist for them.

### 2.3 Scoring Criteria (Strategy Category)

The top 8 qualify for Finals. Rankings are based on:
1. **Agent stability** — doesn't crash, timeouts rarely, handles edge cases
2. **Deck design concept** — coherent archetype with documented win conditions
3. **Simulation Category performance** — ELO rank from automated matches

All three must be addressed. A high-ELO unstable agent that crashes will score lower than a stable mid-ELO agent with a well-documented strategy.

---

## 3. Technical Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────┐
│              Pokémon TCG Simulator               │
│         (provided by The Pokémon Company)        │
└────────────────────┬────────────────────────────┘
                     │ Game State (JSON/struct)
                     ▼
┌─────────────────────────────────────────────────┐
│              State Encoder                       │
│  • Observation parser                            │
│  • Feature vector builder (~200-300 dims)        │
│  • Action space enumerator                       │
└────────────────────┬────────────────────────────┘
                     │ Feature vector + legal actions
                     ▼
┌─────────────────────────────────────────────────┐
│              Policy Model                        │
│  • Phase 1: Rule-based baseline (heuristic)      │
│  • Phase 2: Gradient Boosted Trees (XGBoost)     │
│  • Phase 3 (optional): Lightweight MLP           │
└────────────────────┬────────────────────────────┘
                     │ Scored actions
                     ▼
┌─────────────────────────────────────────────────┐
│              Action Selector                     │
│  • Filters illegal actions                       │
│  • Applies safety guard (no-op fallback)         │
│  • Returns chosen action to simulator            │
└─────────────────────────────────────────────────┘
```

### 3.2 State Representation

The simulator exposes the game state at each decision point. We parse it into a fixed-size feature vector.

**Feature Groups:**

| Group | Features | Approx Dims |
|---|---|---|
| My Active Pokémon | HP, max HP, type, retreat cost, # attacks, energy attached (per type) | 15 |
| My Bench (up to 5) | HP, type, energy attached, evolution stage | 5 × 8 = 40 |
| Opponent Active | HP, max HP, type, known energy attached | 8 |
| Opponent Bench | Count, types observed | 6 |
| My Hand | Card count, # energy, # Trainer, # Supporter, # Pokémon | 6 |
| My Deck | Cards remaining | 1 |
| Prize Cards | My prizes remaining, opponent prizes remaining | 2 |
| Board Flags | Active stadium ID (one-hot or 0), any tool attached | 5 |
| Turn Info | Turn number, whose turn, actions taken this turn | 3 |
| Damage/Status | Active Pokémon status (burned, poisoned, confused, etc.) | 5 |
| **Total** | | **~91** |

> **Design note:** Keep the feature vector under 150 dims for Phase 1. Expand if a more complex model is used in Phase 3.

**Action Space:**

Each turn, legal actions are enumerated from the game state:
- `PLAY_ENERGY(target_card, energy_card)` — attach energy to a Pokémon
- `PLAY_TRAINER(card_id)` — play a trainer card
- `PLAY_SUPPORTER(card_id)` — play the one supporter allowed per turn
- `ATTACK(attack_index)` — use active Pokémon's attack 1 or 2
- `RETREAT(bench_target)` — retreat active Pokémon to bench slot N
- `EVOLVE(bench_or_active, card_id)` — evolve a Pokémon
- `PASS` — end turn (or pass if forced)

The policy model scores each legal action. The highest-scored legal action is returned.

### 3.3 Phase 1 — Rule-Based Heuristic Agent

**Purpose:** Create a working, stable, submittable agent in ≤1 week. Establishes the baseline ELO and ensures the team has a valid submission before starting ML work.

**Logic (priority-ordered):**

```
1. If active Pokémon can KO opponent this turn → ATTACK (highest damage)
2. If active Pokémon can attack (has energy) → ATTACK
3. If hand contains a Supporter → PLAY_SUPPORTER (draw/search priority)
4. If hand contains energy and bench/active has space → PLAY_ENERGY (active first)
5. If hand contains evolution card and valid target exists → EVOLVE
6. If active Pokémon is damaged and bench has full-HP Pokémon with energy → RETREAT
7. If hand has Trainer → PLAY_TRAINER
8. PASS
```

This covers ~90% of decision points without any ML. It will lose to smart opponents but provides:
- A valid, crash-free submission
- Game logs for training data collection

### 3.4 Phase 2 — Gradient Boosted Tree Policy

**Model:** XGBoost classifier (or LightGBM)

**Training Signal Options:**

*Option A — Self-play outcome labeling (preferred):*
Run the heuristic agent against itself and random agents for N games. Label each (state, action) pair with the game outcome (+1 win, -1 loss, 0 draw). Train a classifier to predict win-probability given state + action features.

*Option B — Imitation from rule engine:*
Have the rule-based agent generate "preferred" actions. Train a classifier to imitate these decisions. Faster to bootstrap but won't generalize beyond the rules.

**Feature Engineering per Action:**

For each candidate legal action, concatenate:
- The base game state vector (91 dims)
- Action-specific features:
  - For ATTACK: expected damage, opponent remaining HP after hit, KO flag
  - For PLAY_ENERGY: will this enable an attack next turn? (binary)
  - For PLAY_SUPPORTER: card type (draw vs search vs heal)
  - For EVOLVE: HP gain, unlock stronger attacks (binary)
  - For RETREAT: active HP ratio, bench attacker readiness

**Training Data Target:** 10,000+ (state, action, outcome) triples. Achievable in ~500 self-play games if games average 20 decision points per player.

**Model Config:**
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)
```

**Inference at Match Time:**
```python
def select_action(game_state, legal_actions):
    features = [encode(game_state, action) for action in legal_actions]
    scores = model.predict_proba(features)[:, 1]  # win probability
    best_idx = scores.argmax()
    return legal_actions[best_idx]
```

### 3.5 Phase 3 (Optional) — Lightweight MLP

Only if time permits and Phase 2 plateaus.

```python
nn.Sequential(
    nn.Linear(state_dim + action_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),   # win probability score
    nn.Sigmoid()
)
```

Train with binary cross-entropy on game outcomes. This allows gradient flow and could capture non-linear interactions the tree misses.

---

## 4. Deck Design

### 4.1 Selection Criteria

The deck must be:
1. **Consistent** — draws its win condition reliably by turn 3-4
2. **Simple** — one primary attacker, so the agent rarely faces complex decision trees
3. **Resilient** — doesn't auto-lose to missing one card
4. **Documentable** — the Strategy Report requires explaining the logic

### 4.2 Proposed Archetype: Single-Attacker ex Aggro

**Core concept:** One main ex Pokémon attacker, supported by search/draw Trainers to set it up fast, with Basic Energy to keep the energy model simple (avoids special energy edge cases in the agent).

**Deck Template (60 cards):**

| Count | Card Type | Purpose |
|---|---|---|
| 4 | Main attacker ex (Basic) | Primary win condition |
| 2 | Main attacker ex (backup) | Prize denial insurance |
| 4 | Secondary attacker (Basic non-ex) | Early pressure, absorb KOs |
| 2 | Support Pokémon (draw engine) | Hand refresh |
| 4 | Nest Ball / similar search | Bench setup |
| 4 | Ultra Ball / similar search | Attacker retrieval |
| 4 | Professor's Research (or equivalent) | Draw 7 |
| 3 | Iono / hand disruption Supporter | Disrupt + draw |
| 2 | Boss's Orders / equivalent | Gust opponent bench |
| 2 | Switch / Escape Rope | Mobility |
| 2 | Stadium of choice | Neutral or helpful |
| 3 | Tool cards | Damage boost or protection |
| 14 | Basic Energy (main type) | Attack fuel |
| 4 | Special Energy | Optional acceleration |
| **60** | **Total** | |

> **Exact card IDs to be determined** once the CSV (`EN Card Data.csv`) is parsed to identify the strongest ex attackers in DRI/MEG/ASC. Prioritize attackers with: (a) a 2-energy attack, (b) flat damage with no complex effect text, (c) high HP (270+).

### 4.3 Agent-Deck Alignment

The deck is designed specifically so the agent can execute it well:
- Only one Supporter played per turn → simple action-space
- Energy attachment goes to the active attacker 90% of the time → deterministic rule
- Attack if able → covered by Phase 1 heuristic
- Search Trainers → play immediately when drawn (simple greedy rule)

This alignment between deck design and agent capability is a key differentiator in the Strategy Report.

---

## 5. Development Roadmap

### 5.1 Timeline

```
Week 1 (Jun 23 – Jun 29)
├── Parse EN Card Data.csv — build card metadata lookup
├── Integrate with simulator — understand observation/action API
├── Implement state encoder (feature vector)
└── Ship Phase 1: Rule-based agent → FIRST SUBMISSION

Week 2 (Jun 30 – Jul 6)
├── Run 200+ self-play games, collect (state, action, outcome) logs
├── Build training pipeline (feature extraction → XGBoost)
└── Evaluate: Phase 2 agent vs Phase 1 baseline

Week 3 (Jul 7 – Jul 13)
├── Hyperparameter tuning, feature ablation
├── Fix edge cases / crashes found in self-play
├── Finalize deck (card selection based on win-rate analysis)
└── Submit Phase 2 agent

Week 4–6 (Jul 14 – Aug 3)
├── Collect more game logs from ladder matches
├── Retrain on expanded dataset
├── Optional: Phase 3 MLP experiment
└── Stability hardening (timeout guards, fallback logic)

Week 7–8 (Aug 4 – Aug 17)
├── Final tuning and submission polish
├── Begin drafting Strategy Report
└── FINAL SUBMISSION DEADLINE: Aug 17

Week 9–10 (Aug 18 – Sep 14)
├── Finalize Strategy Report (methodology, results, deck design)
└── STRATEGY REPORT DEADLINE: Sep 14
```

### 5.2 Milestones

| Milestone | Target Date | Success Metric |
|---|---|---|
| First valid submission | Jun 29 | Agent completes matches without crashing |
| Phase 2 trained | Jul 13 | >55% win rate vs Phase 1 baseline |
| Stable ladder position | Aug 3 | ELO score improving or stable |
| Final submission | Aug 17 | Best ELO checkpoint submitted |
| Strategy Report | Sep 14 | Submitted, covers all 3 scoring dimensions |

---

## 6. File Structure

```
ptcg-agent/
├── data/
│   ├── EN_Card_Data.csv          # Competition dataset
│   ├── card_lookup.json          # Parsed card metadata (id → name, type, HP, moves)
│   └── game_logs/                # Self-play game logs (.json per game)
│
├── agent/
│   ├── __init__.py
│   ├── encoder.py                # Game state → feature vector
│   ├── action_space.py           # Legal action enumeration
│   ├── heuristic.py              # Phase 1 rule-based policy
│   ├── policy.py                 # Phase 2/3 ML policy (XGBoost / MLP)
│   └── agent.py                  # Main agent class (called by simulator)
│
├── training/
│   ├── self_play.py              # Run N self-play games, collect logs
│   ├── feature_builder.py        # Build training dataset from logs
│   └── train.py                  # Train and save XGBoost model
│
├── deck/
│   └── deck.txt                  # 60-card deck list (card IDs)
│
├── models/
│   └── policy_v1.pkl             # Trained model checkpoint
│
├── report/
│   └── strategy_report.md        # Draft of Strategy Category submission
│
├── tests/
│   └── test_encoder.py           # Unit tests for state encoder
│
└── main.py                       # Entry point for simulator integration
```

---

## 7. Key Engineering Decisions

### 7.1 Why XGBoost over Deep RL

| Factor | XGBoost | Deep RL (PPO/DQN) |
|---|---|---|
| Training data required | Low (thousands of games) | High (millions of steps) |
| Training time | Minutes | Hours to days |
| Interpretability | High (feature importance) | Low |
| Strategy Report value | High (explain features) | Low (black box) |
| Stability risk | Low | Medium (reward shaping failures) |
| Iteration speed | Fast | Slow |

XGBoost is the right choice given the 8-week timeline and the fact that the Strategy Report requires explaining the agent's reasoning. Feature importance scores from XGBoost become the core of the strategy writeup.

### 7.2 Imperfect Information Handling

The opponent's hand is hidden. We handle this by:
- **Not modeling the opponent's hand** — features only include known information (face-up cards, observed plays)
- **Tracking opponent prize count** — infer game urgency from prizes remaining
- **Tracking opponent energy count** — count energy attached to opponent's Pokémon (visible)
- **Deferring complex inference** — opponent hand prediction is out of scope for Phase 2

### 7.3 Action Scoring vs. Action Classification

We treat this as **action scoring** (predicting win probability for each legal action) rather than direct classification (predicting the best action ID). This is important because:
- Legal actions vary in count and type each turn
- A classifier with fixed output dimensions can't generalize across different action sets
- Scoring each (state, action) pair independently is more flexible and interpretable

### 7.4 Stability First

The agent must handle these edge cases without crashing:
- Empty hand (no legal action other than ATTACK or PASS)
- Active Pokémon with no energy (forced PASS)
- Bench full (can't play Basics from hand)
- All bench Pokémon KO'd (lone attacker)

Every code path must have a `try/except` with a fallback to `PASS` or the first legal action. A crashed agent means a forfeited game and hurts the stability score.

---

## 8. Strategy Report Outline

The Strategy Category requires a written report covering:

### Section 1: Agent Architecture
- State representation design choices
- Action space formulation
- Model selection rationale (XGBoost + why)

### Section 2: Deck Design
- Archetype description and win condition
- Card selection methodology (based on CSV analysis)
- How deck and agent policy were co-designed

### Section 3: Training Methodology
- Self-play setup
- Feature engineering details
- Training/validation split and evaluation metrics

### Section 4: Results
- ELO progression over submission history
- Win rate vs baseline agents
- Feature importance analysis (top 10 features that predict wins)

### Section 5: Limitations and Future Work
- What a Phase 3 MLP would improve
- Opponent modeling potential
- Deck adaptation between matches (future)

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Simulator API changes mid-competition | Low | High | Abstract encoder layer — only `encoder.py` needs changes |
| Not enough self-play data for training | Medium | Medium | Supplement with random-agent games; use Option B (imitation) as fallback |
| Agent crashes on edge cases | High | High | Stability layer with universal fallback; extensive unit tests |
| Deck is weak in the meta | Medium | Medium | Submit Phase 1 baseline to get early ELO signal; adjust deck in Week 3 |
| CSV data parsing issues | Low | Low | Already validated 1,266 cards from PDF; cross-check with CSV |
| Strategy Report is underscored | Medium | High | Start report in Week 4, not Week 8; use XGBoost feature importance as the analytical backbone |

---

## 10. Success Criteria

| Metric | Target |
|---|---|
| Simulation Category: no-crash rate | >95% of games complete |
| Simulation Category: ELO rank | Top 50% of ladder |
| Phase 2 vs Phase 1 win rate | >55% |
| Strategy Report sections | All 5 sections complete |
| Strategy Report: top-8 qualification | Stretch goal |

---

## Appendix A: Card Pool Notes

- Energy cards (IDs 1–20): 8 basic types cover standard type matchups; 12 special energies available for tech choices
- Stadiums (IDs 1240–1267): 28 stadium cards available including N's Castle, Team Rocket's Factory, Jamming Tower — disruption-heavy meta expected given DRI dominance
- DRI (178 cards, largest set) has Team Rocket theming → expect many opponent decks to use Team Rocket's Energy, Team Rocket's Watchtower, and Team Rocket's Factory as staples
- One card ID missing (1083) — unknown card; do not hardcode ID-range assumptions

## Appendix B: Expansion Code Reference

| Code | Set Name |
|---|---|
| SVE | SV Basic Energy |
| TWM | Twilight Masquerade |
| TEF | Temporal Forces |
| DRI | Destined Rivals |
| MEG | Mega Evolution (competition-specific) |
| ASC | Ascent (competition-specific) |
| SSP | Stellar Spark |
| PFL | Paldean Fates |
| JTG | Journey Together |
| WHT | Whiteout (competition-specific) |
| BLK | Blackout (competition-specific) |
| POR | Prismatic Evolution |
| SCR | Surging Sparks |
| SFA | Shrouded Fable |
| PRE | Paldea Evolved |
| SVI | Scarlet & Violet Base |