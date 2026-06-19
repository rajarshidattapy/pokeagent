"""Integration tests for the FastAPI server (requires server to be running)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip all tests if httpx or fastapi not available
pytest.importorskip("httpx")
from httpx import Client

BASE_URL = "http://localhost:8000"


@pytest.fixture
def client():
    with Client(base_url=BASE_URL, timeout=10.0) as c:
        yield c


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body["status"] == "ok"
    assert "cards_loaded" in body


def test_cards_endpoint(client):
    r = client.get("/cards?q=Diancie")
    assert r.status_code == 200
    cards = r.json()
    assert isinstance(cards, list)
    # Mega Diancie ex (ID 766) should be in the results
    names = [c["name"] for c in cards]
    assert any("Diancie" in n for n in names)


def test_cards_expansion_filter(client):
    r = client.get("/cards?expansion=DRI")
    assert r.status_code == 200
    cards = r.json()
    assert all(c.get("expansion") == "DRI" for c in cards)


def test_card_by_id(client):
    r = client.get("/cards/766")
    assert r.status_code == 200
    card = r.json()
    assert card["card_id"] == 766
    assert "Diancie" in card["name"]


def test_card_not_found(client):
    r = client.get("/cards/99999")
    assert r.status_code == 404


def test_deck_get_empty(client):
    r = client.get("/deck")
    assert r.status_code == 200


def test_deck_save_wrong_count(client):
    r = client.post("/deck", json={
        "name": "bad",
        "cards": [{"card_id": 766, "count": 4}],
    })
    assert r.status_code == 400
    assert "60" in r.json()["detail"]


def test_deck_save_two_ace_spec(client):
    cards = (
        [{"card_id": 766, "count": 4}] +
        [{"card_id": 1125, "count": 1}] +   # Master Ball (ACE SPEC)
        [{"card_id": 1100, "count": 1}] +   # Energy Search Pro (ACE SPEC)
        [{"card_id": 5, "count": 54}]        # fill to 60
    )
    r = client.post("/deck", json={"name": "two_ace", "cards": cards})
    assert r.status_code == 400
    assert "ACE SPEC" in r.json()["detail"]


def test_logs_endpoint(client):
    r = client.get("/logs")
    assert r.status_code == 200
    body = r.json()
    assert "logs" in body
    assert "total" in body


def test_stats_endpoint(client):
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert "total_games" in body
    assert "win_rate" in body


def test_predict_no_model(client):
    r = client.post("/predict", json={
        "game_state": {},
        "legal_actions": ["PASS"],
    })
    # Either 200 (model loaded) or 503 (not trained yet)
    assert r.status_code in (200, 503)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
