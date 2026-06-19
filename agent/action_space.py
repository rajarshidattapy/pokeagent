"""
Enumerate legal actions from a game state.

Legal actions are strings of the form:
    ATTACK_<index>              — use attack at index 0 or 1
    PLAY_ENERGY_<card_id>       — attach energy card to active or bench
    PLAY_TRAINER_<card_id>      — play an item/tool card
    PLAY_SUPPORTER_<card_id>    — play a supporter card (one per turn)
    EVOLVE_<target>_<card_id>   — evolve target ("active" or "bench_N") with card
    RETREAT_<bench_index>       — retreat active to bench slot N
    PASS                        — end turn
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.game_state import GameState


def get_legal_actions(gs) -> list[str]:
    """Return list of legal action strings for the current player."""
    actions: list[str] = []
    player = gs.current_player()
    active = player.active

    if active is None:
        return ["PASS"]

    # ── ATTACK ─────────────────────────────────────────────────────────────
    # Available only once per turn, if not paralyzed/asleep
    if not gs.has_attacked and not gs.is_status_blocking_attack(player):
        attacks = [m for m in (active.card.get("moves") or []) if not m.get("is_ability")]
        for i, attack in enumerate(attacks):
            if gs.can_pay_cost(player, active, attack.get("cost_parsed", {})):
                actions.append(f"ATTACK_{i}")

    # ── PLAY_ENERGY ─────────────────────────────────────────────────────────
    # One energy per turn
    if not gs.has_attached_energy:
        for card_id in player.hand_energy_ids():
            actions.append(f"PLAY_ENERGY_{card_id}")

    # ── PLAY_SUPPORTER ──────────────────────────────────────────────────────
    # One supporter per turn
    if not gs.has_played_supporter:
        for card_id in player.hand_supporter_ids():
            actions.append(f"PLAY_SUPPORTER_{card_id}")

    # ── PLAY_TRAINER (Items) ─────────────────────────────────────────────────
    for card_id in player.hand_item_ids():
        actions.append(f"PLAY_TRAINER_{card_id}")

    # ── EVOLVE ──────────────────────────────────────────────────────────────
    if not gs.is_first_turn:
        for card_id, target in gs.get_evolution_targets(player):
            actions.append(f"EVOLVE_{target}_{card_id}")

    # ── RETREAT ─────────────────────────────────────────────────────────────
    if not gs.has_retreated and player.bench:
        cost = active.retreat_cost
        total_energy = sum(active.energy_attached.values())
        if total_energy >= cost:
            for i in range(len(player.bench)):
                actions.append(f"RETREAT_{i}")

    # Always legal
    actions.append("PASS")
    return actions


_KNOWN_TYPES = ["ATTACK", "PLAY_ENERGY", "PLAY_TRAINER", "PLAY_SUPPORTER", "EVOLVE", "RETREAT", "PASS"]


def parse_action(action: str) -> tuple[str, str]:
    """Split 'PLAY_ENERGY_5' → ('PLAY_ENERGY', '5')."""
    for known in _KNOWN_TYPES:
        if action == known:
            return known, ""
        if action.startswith(known + "_"):
            return known, action[len(known) + 1:]
    return action, ""
