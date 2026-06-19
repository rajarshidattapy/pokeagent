"""Phase 1: Rule-based heuristic agent.

Priority order (from PRD):
  1. If active can KO opponent this turn → ATTACK (highest damage)
  2. If active can attack at all → ATTACK
  3. If hand has Supporter → PLAY_SUPPORTER (draw priority)
  4. If hand has energy and targets exist → PLAY_ENERGY (active first)
  5. If hand has evolution card and valid target → EVOLVE
  6. If active is damaged and bench has healthy attacker → RETREAT
  7. If hand has Trainer → PLAY_TRAINER
  8. PASS

All code paths fall back to PASS to prevent crashes.
"""

from __future__ import annotations


def select_action(legal_actions: list[str], game_state: dict) -> str:
    """Choose the best action from the legal action list."""
    try:
        return _select(legal_actions, game_state)
    except Exception:
        return _first_or_pass(legal_actions)


def _select(legal_actions: list[str], gs: dict) -> str:
    if not legal_actions:
        return "PASS"

    action_set = set(legal_actions)
    me = gs.get("me", {})
    opp = gs.get("opponent", {})
    active = me.get("active") or {}
    opp_active = opp.get("active") or {}

    dmg_map = gs.get("_action_damage", {})
    opp_hp = opp_active.get("hp", 9999) or 9999

    # 1. Can KO this turn?
    ko_attacks = [
        a for a in legal_actions
        if a.startswith("ATTACK") and (dmg_map.get(a, 0) or 0) >= opp_hp
    ]
    if ko_attacks:
        return max(ko_attacks, key=lambda a: dmg_map.get(a, 0))

    # 2. Can attack at all?
    attacks = [a for a in legal_actions if a.startswith("ATTACK")]
    if attacks:
        return max(attacks, key=lambda a: dmg_map.get(a, 0))

    # 3. Play a supporter (draw/search first)
    supporters = [a for a in legal_actions if a.startswith("PLAY_SUPPORTER")]
    if supporters:
        draw_map = gs.get("_action_draw_count", {})
        return max(supporters, key=lambda a: draw_map.get(a, 0))

    # 4. Attach energy (prefer active, but any target is fine — game engine handles target)
    energies = [a for a in legal_actions if a.startswith("PLAY_ENERGY")]
    if energies:
        return energies[0]

    # 5. Evolve
    evolves = [a for a in legal_actions if a.startswith("EVOLVE")]
    if evolves:
        hp_gain_map = gs.get("_action_hp_gain", {})
        return max(evolves, key=lambda a: hp_gain_map.get(a, 0))

    # 6. Retreat if active is damaged and bench has healthy Pokémon
    retreats = [a for a in legal_actions if a.startswith("RETREAT")]
    if retreats:
        my_hp = active.get("hp", 1) or 1
        my_max_hp = active.get("max_hp", 1) or 1
        hp_ratio = my_hp / my_max_hp
        bench = me.get("bench") or []
        bench_hp_ratios = [
            (b.get("hp", 1) or 1) / (b.get("max_hp", 1) or 1)
            for b in bench
        ]
        best_bench_ratio = max(bench_hp_ratios) if bench_hp_ratios else 0
        if hp_ratio < 0.4 and best_bench_ratio > hp_ratio:
            return retreats[0]

    # 7. Play a trainer (item)
    trainers = [a for a in legal_actions if a.startswith("PLAY_TRAINER")]
    if trainers:
        return trainers[0]

    return "PASS"


def _first_or_pass(legal_actions: list[str]) -> str:
    return legal_actions[0] if legal_actions else "PASS"
