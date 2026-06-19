"""
Simplified PTCG game state for self-play.

Simplifications vs. real PTCG:
- No complex card text effects (search abilities, special energy interactions)
- No Weakness/Resistance
- Special conditions: paralysis skips turn, confusion 50% self-damage, others do damage on check
- Energy attachment always targets active Pokémon (agent can choose bench via target parameter)
- One energy per turn, one supporter per turn
- Trainers immediately draw/search (simplified to just draw 1 card)
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Card and Pokémon in play ────────────────────────────────────────────────

@dataclass
class ActivePokemon:
    card: dict           # full card dict from card_lookup.json
    hp: int
    energy_attached: dict = field(default_factory=dict)  # {"{P}": 2}
    status: dict = field(default_factory=lambda: {
        "burned": False, "poisoned": False, "confused": False,
        "paralyzed": False, "asleep": False,
    })

    @property
    def card_id(self) -> int:
        return self.card["card_id"]

    @property
    def name(self) -> str:
        return self.card["name"]

    @property
    def max_hp(self) -> int:
        return self.card.get("hp") or 60

    @property
    def retreat_cost(self) -> int:
        return self.card.get("retreat") or 0

    @property
    def ptype(self) -> str:
        return self.card.get("type") or "{C}"

    @property
    def is_ex(self) -> bool:
        return bool(self.card.get("is_ex"))

    @property
    def total_energy(self) -> int:
        return sum(self.energy_attached.values())

    def attacks(self) -> list[dict]:
        return [m for m in (self.card.get("moves") or []) if not m.get("is_ability")]

    def can_pay(self, cost_parsed: dict) -> bool:
        """Check if attached energy can satisfy attack cost (simplified: total energy check)."""
        return self.total_energy >= sum(cost_parsed.values())

    def max_damage(self) -> int:
        """Best damage this Pokémon can deal given current energy."""
        best = 0
        for atk in self.attacks():
            if self.can_pay(atk.get("cost_parsed", {})):
                dmg = atk.get("damage") or 0
                if isinstance(dmg, int) and dmg > best:
                    best = dmg
        return best

    def to_dict(self) -> dict:
        return {
            "card_id": self.card_id,
            "name": self.name,
            "hp": self.hp,
            "max_hp": self.max_hp,
            "type": self.ptype,
            "retreat_cost": self.retreat_cost,
            "is_ex": self.is_ex,
            "energy_attached": dict(self.energy_attached),
            "status": dict(self.status),
            "max_damage": self.max_damage(),
        }


# ── Player state ─────────────────────────────────────────────────────────────

class Player:
    def __init__(self, deck: list[dict], card_lookup: dict):
        self.deck = list(deck)
        self.hand: list[dict] = []
        self.discard: list[dict] = []
        self.active: Optional[ActivePokemon] = None
        self.bench: list[ActivePokemon] = []
        self.prizes: list[dict] = []
        self.card_lookup = card_lookup
        self.supporter_played_this_turn = False

    def draw(self, n: int = 1) -> list[dict]:
        drawn = []
        for _ in range(n):
            if not self.deck:
                return drawn  # deck-out handled by game
            drawn.append(self.deck.pop(0))
        self.hand.extend(drawn)
        return drawn

    def shuffle_deck(self):
        random.shuffle(self.deck)

    def hand_energy_ids(self) -> list[int]:
        return [c["card_id"] for c in self.hand if c.get("is_energy")]

    def hand_supporter_ids(self) -> list[int]:
        return [c["card_id"] for c in self.hand if c.get("is_supporter")]

    def hand_item_ids(self) -> list[int]:
        return [c["card_id"] for c in self.hand if c.get("is_item")]

    def hand_pokemon_ids(self) -> list[int]:
        return [c["card_id"] for c in self.hand if c.get("is_pokemon")]

    def remove_from_hand(self, card_id: int) -> Optional[dict]:
        for i, c in enumerate(self.hand):
            if c["card_id"] == card_id:
                return self.hand.pop(i)
        return None

    def has_card_in_hand(self, card_id: int) -> bool:
        return any(c["card_id"] == card_id for c in self.hand)

    def bench_has_basics(self) -> bool:
        return any(
            p.card.get("stage") == "Basic Pokémon"
            for p in self.bench
        )

    def setup_initial_hand(self):
        """Draw 7 cards, ensure at least 1 Basic Pokémon (mulligan up to 4 times)."""
        basics_in_hand = lambda: [c for c in self.hand if c.get("stage") == "Basic Pokémon"]
        for _ in range(5):
            self.hand = []
            self.deck = [c for c in self.deck]  # already shuffled
            self.draw(7)
            if basics_in_hand():
                break
            self.deck.extend(self.hand)
            self.shuffle_deck()

    def place_active(self, card_id: int):
        card = self.remove_from_hand(card_id)
        if card:
            self.active = ActivePokemon(card=card, hp=card.get("hp") or 60)

    def place_bench(self, card_id: int) -> bool:
        if len(self.bench) >= 5:
            return False
        card = self.remove_from_hand(card_id)
        if card:
            self.bench.append(ActivePokemon(card=card, hp=card.get("hp") or 60))
            return True
        return False

    def take_prize(self) -> Optional[dict]:
        if self.prizes:
            card = self.prizes.pop()
            self.hand.append(card)
            return card
        return None

    def hand_summary(self) -> dict:
        return {
            "total": len(self.hand),
            "energy": sum(1 for c in self.hand if c.get("is_energy")),
            "trainers": sum(1 for c in self.hand if c.get("is_item")),
            "supporters": sum(1 for c in self.hand if c.get("is_supporter")),
            "pokemon": sum(1 for c in self.hand if c.get("is_pokemon")),
        }

    def to_public_dict(self, full: bool = True) -> dict:
        """full=True for own side, full=False for opponent (hidden hand)."""
        bench_list = [p.to_dict() for p in self.bench]
        result = {
            "active": self.active.to_dict() if self.active else None,
            "bench": bench_list,
            "deck_remaining": len(self.deck),
            "prizes": len(self.prizes),
        }
        if full:
            result["hand"] = self.hand_summary()
        else:
            result["hand"] = {"total": len(self.hand)}
        return result


# ── Game state ────────────────────────────────────────────────────────────────

class GameState:
    MAX_TURNS = 60  # forfeit after this many total turns

    def __init__(self, player_a: Player, player_b: Player):
        self.players = [player_a, player_b]
        self.turn_number = 0          # total half-turns played
        self.current_player_idx = 0
        self.stadium_id: Optional[int] = None

        # Per-turn flags
        self.has_attacked = False
        self.has_attached_energy = False
        self.has_played_supporter = False
        self.has_retreated = False
        self.is_first_turn = True     # first turn of the whole game

        self.winner: Optional[int] = None  # 0=player_a, 1=player_b

    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    def opponent_player(self) -> Player:
        return self.players[1 - self.current_player_idx]

    def setup(self):
        for p in self.players:
            p.shuffle_deck()
            p.setup_initial_hand()

            # Place active: first Basic in hand
            basics = [c for c in p.hand if c.get("stage") == "Basic Pokémon"]
            if basics:
                p.place_active(basics[0]["card_id"])

            # Bench up to 3 more Basics from hand
            remaining_basics = [c for c in p.hand if c.get("stage") == "Basic Pokémon"]
            for card in remaining_basics[:min(3, len(remaining_basics))]:
                p.place_bench(card["card_id"])

            # Take 6 prize cards
            p.prizes = [p.deck.pop(0) for _ in range(min(6, len(p.deck)))]

        self.is_first_turn = True

    def start_turn(self):
        """Begin a new half-turn for the current player."""
        self.has_attacked = False
        self.has_attached_energy = False
        self.has_played_supporter = False
        self.has_retreated = False
        p = self.current_player()
        p.supporter_played_this_turn = False

        if self.turn_number > 0:  # first player doesn't draw on turn 1
            p.draw(1)

        self.is_first_turn = (self.turn_number == 0)

    def end_turn(self):
        """Apply end-of-turn effects and switch player."""
        self._apply_status_damage()
        self.current_player_idx = 1 - self.current_player_idx
        self.turn_number += 1

    def _apply_status_damage(self):
        """Burn/poison damage at end of turn."""
        p = self.current_player()
        if p.active is None:
            return
        s = p.active.status
        if s["burned"]:
            p.active.hp -= 20
            # 50% chance to heal burn
            if random.random() < 0.5:
                s["burned"] = False
        if s["poisoned"]:
            p.active.hp -= 10
        if p.active.hp <= 0:
            self._handle_ko(p, self.opponent_player(), is_own_side=True)

    def apply_action(self, action: str, game_state_dict: dict) -> bool:
        """
        Apply action. Returns True if the game continues, False if it's over.
        game_state_dict is used for action metadata (_action_damage, etc.).
        """
        p = self.current_player()
        opp = self.opponent_player()

        _KNOWN = ["PLAY_ENERGY", "PLAY_SUPPORTER", "PLAY_TRAINER", "EVOLVE", "RETREAT", "ATTACK", "PASS"]
        atype, args = action, ""
        for _k in _KNOWN:
            if action == _k:
                atype, args = _k, ""
                break
            if action.startswith(_k + "_"):
                atype, args = _k, action[len(_k) + 1:]
                break

        try:
            if atype == "ATTACK":
                return self._apply_attack(p, opp, int(args), game_state_dict)
            elif atype == "PLAY_ENERGY":
                self._apply_energy(p, int(args))
            elif atype == "PLAY_SUPPORTER":
                self._apply_supporter(p, int(args))
            elif atype == "PLAY_TRAINER":
                self._apply_trainer(p, int(args))
            elif atype == "EVOLVE":
                sub = args.split("_", 1)
                self._apply_evolve(p, sub[0], int(sub[1]) if len(sub) > 1 else 0)
            elif atype == "RETREAT":
                self._apply_retreat(p, int(args))
            elif atype == "PASS":
                pass
        except Exception:
            pass  # safety: bad action is ignored

        return self._check_win()

    def _apply_attack(self, p: Player, opp: Player, atk_idx: int, gs_dict: dict) -> bool:
        self.has_attacked = True

        # Confusion check
        if p.active and p.active.status.get("confused"):
            if random.random() < 0.5:
                p.active.hp -= 30
                if p.active.hp <= 0:
                    self._handle_ko(p, opp, is_own_side=True)
                return self._check_win()

        if p.active is None or opp.active is None:
            return self._check_win()

        attacks = p.active.attacks()
        if atk_idx >= len(attacks):
            return self._check_win()

        atk = attacks[atk_idx]
        dmg = gs_dict.get("_action_damage", {}).get(f"ATTACK_{atk_idx}", 0) or 0

        # Discharge energy for attacks that say "discard all energy"
        effect = (atk.get("effect") or "").lower()
        if "discard all energy" in effect or "discard the" in effect:
            p.active.energy_attached = {}
        else:
            # Pay energy cost (simplified: subtract 1 from total)
            cost = atk.get("total_energy", 0)
            self._spend_energy(p.active, cost)

        opp.active.hp -= dmg

        # Apply basic status effects
        if "burned" in effect:
            opp.active.status["burned"] = True
        if "poisoned" in effect:
            opp.active.status["poisoned"] = True
        if "paralyzed" in effect:
            opp.active.status["paralyzed"] = True
        if "asleep" in effect:
            opp.active.status["asleep"] = True
        if "confused" in effect:
            opp.active.status["confused"] = True

        if opp.active.hp <= 0:
            self._handle_ko(opp, p, is_own_side=False)

        return self._check_win()

    def _spend_energy(self, pokemon: ActivePokemon, n: int):
        """Remove n energy (any type) from the Pokémon."""
        remaining = n
        for etype in list(pokemon.energy_attached.keys()):
            if remaining <= 0:
                break
            count = pokemon.energy_attached[etype]
            spend = min(count, remaining)
            pokemon.energy_attached[etype] -= spend
            remaining -= spend
        pokemon.energy_attached = {k: v for k, v in pokemon.energy_attached.items() if v > 0}

    def _handle_ko(self, ko_player: Player, attacker_player: Player, is_own_side: bool):
        """Handle a KO'd Pokémon."""
        ko_pmon = ko_player.active
        if ko_pmon is None:
            return

        # Prize cards — attacker takes prizes
        # ex Pokémon give 2 prizes
        prizes_taken = 2 if ko_pmon.is_ex else 1
        for _ in range(prizes_taken):
            attacker_player.take_prize()

        # Move active to discard
        ko_player.discard.append(ko_pmon.card)
        ko_player.active = None

        # Promote bench if available
        if ko_player.bench:
            promoted = ko_player.bench.pop(0)
            ko_player.active = promoted

    def _apply_energy(self, p: Player, card_id: int):
        """Attach energy from hand to active Pokémon."""
        if self.has_attached_energy:
            return
        card = p.remove_from_hand(card_id)
        if card is None or p.active is None:
            return
        etype = card.get("type") or "{C}"
        p.active.energy_attached[etype] = p.active.energy_attached.get(etype, 0) + 1
        self.has_attached_energy = True

    def _apply_supporter(self, p: Player, card_id: int):
        """Play a supporter — simplified to draw 3 cards."""
        if self.has_played_supporter:
            return
        card = p.remove_from_hand(card_id)
        if card is None:
            return
        p.discard.append(card)
        self.has_played_supporter = True
        # Simplified effect: draw 3 cards
        p.draw(3)

    def _apply_trainer(self, p: Player, card_id: int):
        """Play an item/tool — simplified to draw 1 card."""
        card = p.remove_from_hand(card_id)
        if card is None:
            return
        p.discard.append(card)
        # Simplified: just draw 1 card (searching effects are hard to simulate)
        p.draw(1)

    def _apply_evolve(self, p: Player, target: str, card_id: int):
        """Evolve the target Pokémon."""
        evo_card = p.remove_from_hand(card_id)
        if evo_card is None:
            return
        if target == "active" and p.active:
            old_hp = p.active.hp
            p.active.card = evo_card
            p.active.hp = min(old_hp, evo_card.get("hp") or old_hp)
            # Clear status on evolution
            for k in p.active.status:
                p.active.status[k] = False
        elif target.startswith("bench_"):
            idx = int(target.split("_")[1])
            if idx < len(p.bench):
                old_hp = p.bench[idx].hp
                p.bench[idx].card = evo_card
                p.bench[idx].hp = min(old_hp, evo_card.get("hp") or old_hp)

    def _apply_retreat(self, p: Player, bench_idx: int):
        """Retreat active to bench, promote bench Pokémon."""
        if p.active is None or bench_idx >= len(p.bench):
            return
        # Pay retreat cost (simplified: just spend the energy)
        cost = p.active.retreat_cost
        self._spend_energy(p.active, cost)
        p.active.status = {k: False for k in p.active.status}

        new_active = p.bench.pop(bench_idx)
        p.bench.insert(0, p.active)
        p.active = new_active
        self.has_retreated = True

    def _check_win(self) -> bool:
        """Return True if game is still going, False if it ended."""
        for i, p in enumerate(self.players):
            if len(p.prizes) == 0:
                self.winner = i
                return False
            if p.active is None and not p.bench:
                self.winner = 1 - i
                return False
            if not p.deck and not p.hand:
                self.winner = 1 - i
                return False

        if self.turn_number >= self.MAX_TURNS:
            # Forfeit: whoever has fewer prizes wins
            prizes = [len(p.prizes) for p in self.players]
            if prizes[0] < prizes[1]:
                self.winner = 0
            elif prizes[1] < prizes[0]:
                self.winner = 1
            else:
                self.winner = 0  # arbitrary
            return False

        return True

    def can_pay_cost(self, player: Player, pokemon: ActivePokemon, cost: dict) -> bool:
        return pokemon.can_pay(cost)

    def is_status_blocking_attack(self, player: Player) -> bool:
        if player.active is None:
            return True
        s = player.active.status
        if s.get("paralyzed") or s.get("asleep"):
            return True
        return False

    def get_evolution_targets(self, player: Player) -> list[tuple[int, str]]:
        """Return list of (card_id, target) pairs for valid evolutions."""
        result = []
        hand_pokemon = {c["card_id"]: c for c in player.hand if c.get("is_pokemon")}

        for card_id, card in hand_pokemon.items():
            prev = card.get("previous_stage")
            if not prev:
                continue
            if player.active and player.active.name == prev:
                result.append((card_id, "active"))
            for i, bench_pmon in enumerate(player.bench):
                if bench_pmon.name == prev:
                    result.append((card_id, f"bench_{i}"))
        return result

    def to_observation(self, player_idx: int) -> dict:
        """Build the game_state dict for the encoder/agent."""
        me = self.players[player_idx]
        opp = self.players[1 - player_idx]
        active = me.active

        # Pre-compute action metadata
        action_damage: dict[str, int] = {}
        action_enables: dict[str, bool] = {}

        if active:
            for i, atk in enumerate(active.attacks()):
                dmg = atk.get("damage") or 0
                if not isinstance(dmg, int):
                    dmg = 0
                action_damage[f"ATTACK_{i}"] = dmg

            for card in me.hand:
                if card.get("is_energy"):
                    cid = card["card_id"]
                    key = f"PLAY_ENERGY_{cid}"
                    # Does attaching this energy enable an attack?
                    etype = card.get("type") or "{C}"
                    test_energy = dict(active.energy_attached)
                    test_energy[etype] = test_energy.get(etype, 0) + 1
                    test_total = sum(test_energy.values())
                    can_attack_now = any(
                        test_total >= a.get("total_energy", 99)
                        for a in active.attacks()
                    )
                    action_enables[key] = can_attack_now

        import sys as _sys, pathlib as _pl
        _sys.path.insert(0, str(_pl.Path(__file__).parent.parent / "agent"))
        from card_constants import SUPPORTER_DRAW_COUNT
        action_draw: dict[str, int] = {}
        for card in me.hand:
            if card.get("is_supporter"):
                cid = card["card_id"]
                action_draw[f"PLAY_SUPPORTER_{cid}"] = SUPPORTER_DRAW_COUNT.get(cid, 2)

        action_hp_gain: dict[str, int] = {}
        for card_id, target in self.get_evolution_targets(me):
            action_hp_gain[f"EVOLVE_{target}_{card_id}"] = 50  # simplified

        return {
            "me": me.to_public_dict(full=True),
            "opponent": opp.to_public_dict(full=False),
            "turn": self.turn_number,
            "stadium_id": self.stadium_id,
            "_action_damage": action_damage,
            "_action_enables_attack": action_enables,
            "_action_draw_count": action_draw,
            "_action_hp_gain": action_hp_gain,
        }
