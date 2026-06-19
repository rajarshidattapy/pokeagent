"""
Run N self-play games between two agent policies.
Writes one JSON game log per game to log_dir/.
Returns a summary dict matching SimulateResponse schema.

Usage:
    python training/self_play.py --num_games 50 --agent_a heuristic --agent_b random
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from pathlib import Path

# Allow imports from sibling packages
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.game_state import GameState, Player
from agent.action_space import get_legal_actions


def load_card_lookup() -> dict:
    path = Path("data/card_lookup.json")
    if not path.exists():
        raise FileNotFoundError(
            "data/card_lookup.json not found. Run: python scripts/build_card_lookup.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_deck(card_ids: list[int], card_lookup: dict) -> list[dict]:
    """Build a list of 60 card dicts from a list of card_id repetitions."""
    deck = []
    for cid in card_ids:
        card = card_lookup.get(str(cid))
        if card:
            deck.append(card)
    return deck


def expand_deck_list(deck_list: list[dict], card_lookup: dict) -> list[int]:
    """Expand [{card_id, count}, ...] → flat list of 60 card_ids."""
    ids = []
    for entry in deck_list:
        ids.extend([entry["card_id"]] * entry["count"])
    return ids


DEFAULT_DECK_LIST = [
    {"card_id": 766, "count": 4},    # Mega Diancie ex
    {"card_id": 687, "count": 2},    # Mega Absol ex
    {"card_id": 756, "count": 2},    # Mega Kangaskhan ex
    {"card_id": 1121, "count": 4},   # Ultra Ball
    {"card_id": 1125, "count": 1},   # Master Ball (ACE SPEC)
    {"card_id": 1097, "count": 4},   # Night Stretcher
    {"card_id": 1094, "count": 3},   # Bug Catching Set
    {"card_id": 1227, "count": 4},   # Lillie's Determination
    {"card_id": 1199, "count": 3},   # Lacey
    {"card_id": 1213, "count": 1},   # Judge
    {"card_id": 1182, "count": 2},   # Boss's Orders
    {"card_id": 1198, "count": 2},   # Crispin
    {"card_id": 1246, "count": 2},   # Jamming Tower
    {"card_id": 5,   "count": 19},   # Basic {P} Energy
    {"card_id": 11,  "count": 3},    # Mist Energy
    {"card_id": 9,   "count": 1},    # Boomerang Energy
    {"card_id": 1121, "count": 3},   # Ultra Ball (already counted above, adjust below)
]


def make_default_deck_ids() -> list[int]:
    """Return a valid 60-card deck using verified card IDs."""
    ids = (
        [766] * 4 +    # Mega Diancie ex
        [687] * 2 +    # Mega Absol ex
        [756] * 2 +    # Mega Kangaskhan ex
        [1121] * 4 +   # Ultra Ball
        [1125] * 1 +   # Master Ball
        [1097] * 4 +   # Night Stretcher
        [1094] * 3 +   # Bug Catching Set
        [1227] * 4 +   # Lillie's Determination
        [1199] * 3 +   # Lacey
        [1213] * 1 +   # Judge
        [1182] * 2 +   # Boss's Orders
        [1198] * 2 +   # Crispin
        [1246] * 2 +   # Jamming Tower
        [5] * 23 +     # Basic {P} Energy
        [11] * 2 +     # Mist Energy
        [9] * 1        # Boomerang Energy
    )
    assert len(ids) == 60, f"Default deck has {len(ids)} cards"
    return ids


def play_game(
    deck_a_ids: list[int],
    deck_b_ids: list[int],
    card_lookup: dict,
    policy_a,
    policy_b,
    game_id: str,
) -> dict:
    """
    Play one game between policy_a (player 0) and policy_b (player 1).
    Returns the game log dict.
    """
    deck_a = build_deck(deck_a_ids, card_lookup)
    deck_b = build_deck(deck_b_ids, card_lookup)

    # Filter out cards that didn't load
    deck_a = [c for c in deck_a if c]
    deck_b = [c for c in deck_b if c]

    player_a = Player(deck=deck_a, card_lookup=card_lookup)
    player_b = Player(deck=deck_b, card_lookup=card_lookup)

    gs = GameState(player_a, player_b)
    gs.setup()

    policies = [policy_a, policy_b]
    player_names = ["agent_a", "agent_b"]

    turns_log = []
    game_continues = True

    while game_continues:
        gs.start_turn()
        current_idx = gs.current_player_idx
        policy = policies[current_idx]

        # Each player takes actions until they PASS or attack
        while True:
            obs = gs.to_observation(current_idx)
            legal = get_legal_actions(gs)

            if not legal or legal == ["PASS"]:
                gs.apply_action("PASS", obs)
                turns_log.append({
                    "turn": gs.turn_number,
                    "player": player_names[current_idx],
                    "game_state": obs,
                    "legal_actions": ["PASS"],
                    "action_taken": "PASS",
                    "outcome": None,  # filled after game
                })
                break

            try:
                action = policy.select_action(obs, legal)
            except Exception:
                action = "PASS"

            if action not in legal:
                action = random.choice(legal)

            turns_log.append({
                "turn": gs.turn_number,
                "player": player_names[current_idx],
                "game_state": obs,
                "legal_actions": legal,
                "action_taken": action,
                "outcome": None,
            })

            game_continues = gs.apply_action(action, obs)
            if not game_continues:
                break

            # After attacking or passing, end the action loop
            if action.startswith("ATTACK") or action == "PASS":
                break

        if not game_continues:
            break

        gs.end_turn()

    winner_idx = gs.winner
    winner_name = player_names[winner_idx] if winner_idx is not None else "draw"

    # Fill outcomes
    for turn in turns_log:
        if turn["player"] == player_names[winner_idx]:
            turn["outcome"] = 1
        elif winner_idx is None:
            turn["outcome"] = 0
        else:
            turn["outcome"] = 0

    return {
        "game_id": game_id,
        "winner": winner_name,
        "total_turns": gs.turn_number,
        "turns": turns_log,
    }


def run_games(
    num_games: int = 10,
    agent_a: str = "heuristic",
    agent_b: str = "random",
    deck_a: list[int] | None = None,
    deck_b: list[int] | None = None,
    log_dir: str = "data/game_logs/",
) -> dict:
    """
    Run num_games games and write logs. Returns stats dict.
    deck_a/deck_b: flat list of 60 card_ids. Defaults to DEFAULT deck.
    """
    card_lookup = load_card_lookup()

    from agent.policy import load_policy
    import sys, os
    os.chdir(Path(__file__).parent.parent)

    policy_a = load_policy(agent_a)
    policy_b = load_policy(agent_b)

    if deck_a is None:
        deck_a = make_default_deck_ids()
    if deck_b is None:
        deck_b = make_default_deck_ids()

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Find next game number
    existing = list(Path(log_dir).glob("game_*.json"))
    start_n = len(existing) + 1

    wins_a = wins_b = draws = 0
    log_ids = []

    for i in range(num_games):
        game_num = start_n + i
        game_id = f"game_{game_num:04d}"

        log = play_game(deck_a, deck_b, card_lookup, policy_a, policy_b, game_id)
        winner = log["winner"]
        if winner == "agent_a":
            wins_a += 1
        elif winner == "agent_b":
            wins_b += 1
        else:
            draws += 1

        out_path = Path(log_dir) / f"{game_id}.json"
        out_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        log_ids.append(game_id)
        print(f"[{game_id}] winner={winner} turns={log['total_turns']}")

    total = wins_a + wins_b + draws
    return {
        "games_played": num_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": round(wins_a / total, 3) if total else 0.0,
        "log_ids": log_ids,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=50)
    parser.add_argument("--agent_a", default="heuristic")
    parser.add_argument("--agent_b", default="random")
    args = parser.parse_args()

    result = run_games(num_games=args.num_games, agent_a=args.agent_a, agent_b=args.agent_b)
    print(f"\nDone: {result['wins_a']}W / {result['wins_b']}L / {result['draws']}D")
    print(f"Win rate A: {result['win_rate_a']:.1%}")
