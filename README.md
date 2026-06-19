# Arceus — Autonomous Pokémon TCG Battle Agent
Kaggle PTCG AI Battle Challenge (Strategy Category)

An ML-powered agent that plays the Pokémon Trading Card Game through self-play reinforcement learning. The system simulates full TCG games between a heuristic agent and a random opponent, logs every decision as a labeled training sample, and trains an XGBoost policy model to predict the highest-value action at any game state.

The agent encodes each board state into an 84-dimensional feature vector — capturing HP ratios, Pokémon types, energy counts, hand composition, prize counts, bench size, and action type — then scores all legal actions to select the best move in real time.

The trained model achieves 99.4% validation accuracy and 99.9% AUC-ROC on self-play data, with type matching and prize race pressure emerging as the strongest learned signals. The backend is served via FastAPI with endpoints for prediction, simulation, deck management, and game log replay, paired with a browser dashboard for visualization.

## Stack
- **Model** — XGBoost trained on self-play logs (`model.pkl`)
- **Backend** — FastAPI (`POST /predict`, `POST /simulate`, `GET /cards`, `GET /deck`, `GET /logs`)
- **Frontend** — HTML dashboard (`index.html`, `deck_builder.html`, `match_viewer.html`)

## Data
| File | Description |
|---|---|
| `data/EN_Card_Data.csv` | English card metadata (1,266 cards, 26 expansions) |
| `data/JP_Card_Data.csv` | Japanese card metadata |
| `data/Card_ID List_EN.pdf` | EN card ID reference |
| `data/Card_ID List_JP.pdf` | JP card ID reference |

## Training Results (v1)

Trained on 20 self-play games (heuristic vs random), 2,397 samples, 84 features.

| Metric | Value |
|---|---|
| Train samples | 2,037 |
| Val samples | 360 |
| Val Accuracy | 99.44% |
| Val AUC-ROC | 99.94% |
| Model | XGBoost (500 trees, max_depth=6, lr=0.05) |
| Output | `models/model.pkl` |

### Top 15 Features by Importance

| Feature | Importance | Notes |
|---|---|---|
| `opp_type_8` | 0.1593 | Opponent active Pokémon type (Psychic) |
| `active_type_8` | 0.1220 | My active Pokémon type (Psychic) |
| `opp_prizes` | 0.0852 | Opponent prize cards remaining |
| `opp_active_hp_ratio` | 0.0683 | Opponent active HP % |
| `active_hp_ratio` | 0.0582 | My active HP % |
| `active_retreat` | 0.0534 | My active retreat cost |
| `action_type_5` | 0.0515 | RETREAT action |
| `bench_hp_1` | 0.0454 | Bench slot 1 HP ratio |
| `opp_bench_size` | 0.0423 | Opponent bench count |
| `bench_hp_0` | 0.0296 | Bench slot 0 HP ratio |
| `opp_type_6` | 0.0293 | Opponent type (Dragon) |
| `active_type_6` | 0.0284 | My type (Dragon) |
| `bench_size` | 0.0264 | My bench count |
| `opp_type_4` | 0.0220 | Opponent type (Fighting) |
| `active_type_4` | 0.0214 | My type (Fighting) |

**Zero-importance features:** all non-Psychic energy types, status conditions, stadiums, `my_prizes`, `active_is_ex`, attack flags. These will activate with broader deck/game diversity in future training runs.

## Deadlines
| Deliverable | Due |
|---|---|
| Kaggle simulation submission | Aug 17, 2026 |
| Strategy Report (PDF) + codebase | Sep 14, 2026 |

## Docs
- [docs/prd.md](docs/prd.md) — Agent strategy & scope
- [docs/ml_prd.md](docs/ml_prd.md) — Implementation stack & architecture
