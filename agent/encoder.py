"""
GameState dict → 84-dimensional float32 numpy array.

The game_state dict structure (produced by self_play.py):
{
  "me":  { "active": {...}, "bench": [...], "hand": {...}, "deck_remaining": int, "prizes": int },
  "opponent": { "active": {...}, "bench": [...], "hand": {"total": int}, "deck_remaining": int, "prizes": int },
  "turn": int,
  "stadium_id": int | None,
  "_action_damage":         { "ATTACK_0": int, ... },
  "_action_enables_attack": { "PLAY_ENERGY_<card_id>": bool, ... },
  "_action_draw_count":     { "PLAY_SUPPORTER_<card_id>": int, ... },
}

active Pokémon dict:
{
  "card_id": int, "name": str, "hp": int, "max_hp": int,
  "type": str,          # e.g. "{P}"
  "retreat_cost": int,
  "is_ex": bool,
  "energy_attached": {"{P}": 2, ...},
  "status": {"burned": bool, "poisoned": bool, "confused": bool, "paralyzed": bool, "asleep": bool},
}
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from card_constants import ENERGY_TYPE_INDEX, TOP_STADIUMS

FEATURE_DIM = 84

# Named feature list (matches FEATURE_DIM)
FEATURE_NAMES: list[str] = [
    "active_hp_ratio",                          # 0
    *[f"active_type_{i}" for i in range(9)],    # 1-9
    *[f"active_energy_{i}" for i in range(9)],  # 10-18
    "active_retreat",                            # 19
    "active_is_ex",                              # 20
    "bench_size",                                # 21
    *[f"bench_hp_{i}" for i in range(5)],       # 22-26
    "opp_active_hp_ratio",                       # 27
    *[f"opp_type_{i}" for i in range(9)],       # 28-36
    *[f"opp_energy_{i}" for i in range(9)],     # 37-45
    "opp_bench_size",                            # 46
    "opp_is_ex",                                 # 47
    "hand_total",                                # 48
    "hand_energy",                               # 49
    "hand_trainers",                             # 50
    "hand_supporters",                           # 51
    "hand_pokemon",                              # 52
    "deck_remaining",                            # 53
    "my_prizes",                                 # 54
    "opp_prizes",                                # 55
    "turn_norm",                                 # 56
    *[f"stadium_{i}" for i in range(10)],       # 57-66
    "status_burned",                             # 67
    "status_poisoned",                           # 68
    "status_confused",                           # 69
    "status_paralyzed",                          # 70
    "status_asleep",                             # 71
    *[f"action_type_{i}" for i in range(7)],    # 72-78
    "attack_damage_norm",                        # 79
    "attack_ko_flag",                            # 80
    "attack_risk_flag",                          # 81
    "energy_enables_attack",                     # 82
    "evolve_hp_gain",                            # 83  (also supporter_draw_norm for supporters)
]

assert len(FEATURE_NAMES) == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {len(FEATURE_NAMES)}"

ACTION_TYPE_INDEX = {
    "ATTACK": 0, "PLAY_ENERGY": 1, "PLAY_TRAINER": 2,
    "PLAY_SUPPORTER": 3, "EVOLVE": 4, "RETREAT": 5, "PASS": 6,
}


def encode(game_state: dict, action: str) -> np.ndarray:
    """Return a 84-dim float32 vector for the (game_state, action) pair."""
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    me = game_state.get("me", {})
    opp = game_state.get("opponent", {})
    active = me.get("active") or {}
    opp_active = opp.get("active") or {}

    # ── My active Pokémon ───────────────────────────────────────────────────
    vec[0] = _safe_ratio(active.get("hp", 0), active.get("max_hp", 1))

    t = active.get("type", "")
    if t in ENERGY_TYPE_INDEX:
        vec[1 + ENERGY_TYPE_INDEX[t]] = 1.0

    for etype, cnt in (active.get("energy_attached") or {}).items():
        if etype in ENERGY_TYPE_INDEX:
            vec[10 + ENERGY_TYPE_INDEX[etype]] = float(cnt)

    vec[19] = float(active.get("retreat_cost", 0))
    vec[20] = float(bool(active.get("is_ex")))

    # ── My bench ────────────────────────────────────────────────────────────
    bench = me.get("bench") or []
    vec[21] = float(len(bench))
    for i in range(5):
        if i < len(bench):
            vec[22 + i] = _safe_ratio(bench[i].get("hp", 0), bench[i].get("max_hp", 1))

    # ── Opponent active ─────────────────────────────────────────────────────
    vec[27] = _safe_ratio(opp_active.get("hp", 0), opp_active.get("max_hp", 1))

    t2 = opp_active.get("type", "")
    if t2 in ENERGY_TYPE_INDEX:
        vec[28 + ENERGY_TYPE_INDEX[t2]] = 1.0

    for etype, cnt in (opp_active.get("energy_attached") or {}).items():
        if etype in ENERGY_TYPE_INDEX:
            vec[37 + ENERGY_TYPE_INDEX[etype]] = float(cnt)

    opp_bench = opp.get("bench") or []
    vec[46] = float(len(opp_bench))
    vec[47] = float(bool(opp_active.get("is_ex")))

    # ── Hand / deck / prizes ────────────────────────────────────────────────
    hand = me.get("hand") or {}
    vec[48] = float(hand.get("total", 0))
    vec[49] = float(hand.get("energy", 0))
    vec[50] = float(hand.get("trainers", 0))
    vec[51] = float(hand.get("supporters", 0))
    vec[52] = float(hand.get("pokemon", 0))
    vec[53] = float(me.get("deck_remaining", 0))
    vec[54] = float(me.get("prizes", 0))
    vec[55] = float(opp.get("prizes", 0))
    vec[56] = float(game_state.get("turn", 0)) / 30.0

    # ── Stadium ─────────────────────────────────────────────────────────────
    stadium_id = game_state.get("stadium_id")
    if stadium_id in TOP_STADIUMS:
        vec[57 + TOP_STADIUMS.index(stadium_id)] = 1.0

    # ── Status conditions ───────────────────────────────────────────────────
    status = active.get("status") or {}
    for i, cond in enumerate(["burned", "poisoned", "confused", "paralyzed", "asleep"]):
        vec[67 + i] = float(bool(status.get(cond)))

    # ── Action type ─────────────────────────────────────────────────────────
    # Match against known multi-word prefixes (e.g. "PLAY_ENERGY_5" → "PLAY_ENERGY")
    atype = _parse_action_type(action)
    if atype in ACTION_TYPE_INDEX:
        vec[72 + ACTION_TYPE_INDEX[atype]] = 1.0

    # ── Action-specific features ────────────────────────────────────────────
    dmg_map = game_state.get("_action_damage") or {}
    ena_map = game_state.get("_action_enables_attack") or {}
    draw_map = game_state.get("_action_draw_count") or {}

    if atype == "ATTACK":
        dmg = dmg_map.get(action, 0) or 0
        vec[79] = float(dmg) / 300.0
        opp_hp = opp_active.get("hp", 1) or 1
        vec[80] = float(dmg >= opp_hp)
        opp_max_dmg = opp_active.get("max_damage", 0) or 0
        my_hp = active.get("hp", 9999) or 9999
        vec[81] = float(opp_max_dmg >= my_hp)

    elif atype == "PLAY_ENERGY":
        vec[82] = float(bool(ena_map.get(action)))

    elif atype == "PLAY_SUPPORTER":
        draw = draw_map.get(action, 0) or 0
        vec[83] = float(draw) / 8.0

    elif atype == "EVOLVE":
        gain = game_state.get("_action_hp_gain", {}).get(action, 0) or 0
        vec[83] = float(gain) / 200.0

    return vec


def _parse_action_type(action: str) -> str:
    """Return the canonical action type for any action string."""
    for known in ACTION_TYPE_INDEX:
        if action == known or action.startswith(known + "_"):
            return known
    return action.split("_")[0]


def _safe_ratio(a, b) -> float:
    try:
        return float(a) / float(b) if b else 0.0
    except (TypeError, ZeroDivisionError):
        return 0.0
