"""Unit tests for agent/encoder.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.encoder import encode, FEATURE_DIM, FEATURE_NAMES


def mock_game_state(active_hp=200, opp_hp=150, turn=5):
    return {
        "me": {
            "active": {
                "card_id": 766, "name": "Mega Diancie ex",
                "hp": active_hp, "max_hp": 270,
                "type": "{P}", "retreat_cost": 1,
                "is_ex": True,
                "energy_attached": {"{P}": 2},
                "status": {"burned": False, "poisoned": False, "confused": False, "paralyzed": False, "asleep": False},
                "max_damage": 240,
            },
            "bench": [
                {"card_id": 687, "name": "Mega Absol ex", "hp": 280, "max_hp": 280, "is_ex": True},
            ],
            "hand": {"total": 5, "energy": 2, "trainers": 1, "supporters": 1, "pokemon": 1},
            "deck_remaining": 30,
            "prizes": 4,
        },
        "opponent": {
            "active": {
                "card_id": 300, "name": "Some Pokémon",
                "hp": opp_hp, "max_hp": 280,
                "type": "{R}", "retreat_cost": 2,
                "is_ex": True,
                "energy_attached": {"{R}": 1},
                "status": {},
                "max_damage": 100,
            },
            "bench": [],
            "hand": {"total": 4},
            "deck_remaining": 35,
            "prizes": 3,
        },
        "turn": turn,
        "stadium_id": None,
        "_action_damage": {"ATTACK_0": 240, "ATTACK_1": 120},
        "_action_enables_attack": {"PLAY_ENERGY_5": True},
        "_action_draw_count": {"PLAY_SUPPORTER_1227": 6},
        "_action_hp_gain": {},
    }


class TestEncoder:
    def test_output_shape(self):
        gs = mock_game_state()
        vec = encode(gs, "ATTACK_0")
        assert vec.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {vec.shape}"

    def test_output_dtype(self):
        vec = encode(mock_game_state(), "PASS")
        assert vec.dtype == np.float32

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_hp_ratio_range(self):
        vec = encode(mock_game_state(active_hp=135), "PASS")
        ratio = vec[0]
        assert 0.0 <= ratio <= 1.0, f"HP ratio {ratio} out of [0,1]"
        assert abs(ratio - 135/270) < 0.01

    def test_active_type_one_hot(self):
        """Psychic type ({P}) should set index 1+4=5."""
        vec = encode(mock_game_state(), "PASS")
        type_vec = vec[1:10]
        assert type_vec.sum() == 1.0, "Type one-hot should sum to 1"
        assert type_vec[4] == 1.0, "{P} should be at index 4"

    def test_attack_damage_normalized(self):
        vec = encode(mock_game_state(), "ATTACK_0")
        assert abs(vec[79] - 240/300) < 0.01, "Attack damage not normalized correctly"

    def test_attack_ko_flag(self):
        gs = mock_game_state(opp_hp=200)
        vec_ko = encode(gs, "ATTACK_0")    # 240 >= 200 → KO
        vec_no = encode(gs, "ATTACK_1")    # 120 < 200 → no KO
        assert vec_ko[80] == 1.0, "KO flag should be set"
        assert vec_no[80] == 0.0, "KO flag should not be set"

    def test_supporter_draw_norm(self):
        vec = encode(mock_game_state(), "PLAY_SUPPORTER_1227")
        assert abs(vec[83] - 6/8) < 0.01

    def test_energy_enables_attack(self):
        vec = encode(mock_game_state(), "PLAY_ENERGY_5")
        assert vec[82] == 1.0

    def test_all_finite(self):
        gs = mock_game_state()
        for action in ["ATTACK_0", "PLAY_ENERGY_5", "PLAY_SUPPORTER_1227", "PASS", "RETREAT_0"]:
            vec = encode(gs, action)
            assert np.all(np.isfinite(vec)), f"Non-finite values for action {action}"

    def test_empty_game_state(self):
        """Encoder must not crash on sparse/empty game state."""
        vec = encode({}, "PASS")
        assert vec.shape == (FEATURE_DIM,)
        assert np.all(np.isfinite(vec))

    def test_turn_normalization(self):
        vec = encode(mock_game_state(turn=15), "PASS")
        assert abs(vec[56] - 15/30) < 0.01

    def test_prizes(self):
        gs = mock_game_state()
        vec = encode(gs, "PASS")
        assert vec[54] == 4.0  # my prizes
        assert vec[55] == 3.0  # opp prizes

    def test_bench_size(self):
        vec = encode(mock_game_state(), "PASS")
        assert vec[21] == 1.0  # one bench Pokémon

    def test_pass_action_type(self):
        vec = encode(mock_game_state(), "PASS")
        # PASS is index 6 in ACTION_TYPE_INDEX, starts at vec[72]
        action_type_vec = vec[72:79]
        assert action_type_vec[6] == 1.0
        assert action_type_vec.sum() == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
