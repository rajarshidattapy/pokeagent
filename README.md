# Pokémon TCG AI Battle Agent
**Paramarsh Labs** — Kaggle PTCG AI Battle Challenge (Strategy Category)

An ML-based agent that plays the Pokémon Trading Card Game using a trained XGBoost policy model, exposed via FastAPI with a browser UI for development and the Strategy Report.

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

## Deadlines
| Deliverable | Due |
|---|---|
| Kaggle simulation submission | Aug 17, 2026 |
| Strategy Report (PDF) + codebase | Sep 14, 2026 |

## Docs
- [docs/prd.md](docs/prd.md) — Agent strategy & scope
- [docs/ml_prd.md](docs/ml_prd.md) — Implementation stack & architecture
