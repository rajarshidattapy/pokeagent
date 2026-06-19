"""Unit tests for agent/heuristic.py and agent/policy.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.heuristic import select_action


def gs(opp_hp=100, my_hp=270, active_hp=150):
    return {
        "me": {
            "active": {"hp": my_hp, "max_hp": 270, "type": "{P}", "is_ex": True, "retreat_cost": 1, "status": {}},
            "bench": [{"hp": 270, "max_hp": 270}],
            "hand": {"total": 5, "energy": 1, "trainers": 1, "supporters": 1, "pokemon": 1},
            "deck_remaining": 30,
            "prizes": 4,
        },
        "opponent": {
            "active": {"hp": opp_hp, "max_hp": 280, "type": "{R}", "is_ex": True},
            "bench": [],
            "hand": {"total": 4},
            "prizes": 3,
        },
        "turn": 5,
        "_action_damage": {"ATTACK_0": 240, "ATTACK_1": 100},
        "_action_draw_count": {"PLAY_SUPPORTER_1227": 6, "PLAY_SUPPORTER_1199": 4},
        "_action_hp_gain": {},
    }


class TestHeuristic:
    def test_ko_preferred_over_regular_attack(self):
        """Rule 1: if ATTACK_0 KOs (240 >= 100), pick ATTACK_0."""
        legal = ["ATTACK_0", "ATTACK_1", "PLAY_ENERGY_5", "PASS"]
        action = select_action(legal, gs(opp_hp=100))
        assert action == "ATTACK_0"

    def test_attack_when_no_ko(self):
        """Rule 2: attack even if no KO."""
        legal = ["ATTACK_0", "ATTACK_1", "PLAY_ENERGY_5", "PASS"]
        action = select_action(legal, gs(opp_hp=300))
        assert action.startswith("ATTACK")

    def test_supporter_before_energy(self):
        """Rule 3: supporter > energy when no attack available."""
        legal = ["PLAY_SUPPORTER_1227", "PLAY_ENERGY_5", "PASS"]
        action = select_action(legal, gs())
        assert action.startswith("PLAY_SUPPORTER")

    def test_best_draw_supporter(self):
        """Choose supporter with highest draw count."""
        legal = ["PLAY_SUPPORTER_1227", "PLAY_SUPPORTER_1199", "PASS"]
        action = select_action(legal, gs())
        assert action == "PLAY_SUPPORTER_1227"  # 6 draws > 4

    def test_energy_before_trainer(self):
        """Rule 4: energy attachment before trainer."""
        legal = ["PLAY_ENERGY_5", "PLAY_TRAINER_1097", "PASS"]
        action = select_action(legal, gs())
        assert action == "PLAY_ENERGY_5"

    def test_pass_when_nothing(self):
        """Rule 8: PASS as fallback."""
        action = select_action(["PASS"], gs())
        assert action == "PASS"

    def test_empty_legal_actions(self):
        """Never crash on empty list."""
        action = select_action([], gs())
        assert action == "PASS"

    def test_exception_safety(self):
        """Must not raise even with totally malformed game state."""
        action = select_action(["ATTACK_0", "PASS"], {})
        assert action in {"ATTACK_0", "PASS"}

    def test_retreat_when_low_hp(self):
        """Rule 6: retreat if active HP is very low."""
        legal = ["RETREAT_0", "PLAY_TRAINER_1097", "PASS"]
        game = gs(my_hp=20)  # 20/270 ≈ 7% HP
        action = select_action(legal, game)
        assert action == "RETREAT_0"


class TestRandomPolicy:
    def test_returns_legal_action(self):
        from agent.policy import RandomPolicy
        p = RandomPolicy()
        legal = ["ATTACK_0", "PLAY_ENERGY_5", "PASS"]
        for _ in range(20):
            assert p.select_action({}, legal) in legal

    def test_empty_returns_pass(self):
        from agent.policy import RandomPolicy
        assert RandomPolicy().select_action({}, []) == "PASS"

    def test_score_actions_sums_to_1(self):
        from agent.policy import RandomPolicy
        scores = RandomPolicy().score_actions({}, ["A", "B", "C"])
        assert abs(sum(scores.values()) - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
