"""Parse EN_Card_Data.csv → data/card_lookup.json.

Run from the project root:
    python scripts/build_card_lookup.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

TYPE_MAP = {
    "{G}": "Grass", "{R}": "Fire", "{W}": "Water", "{L}": "Lightning",
    "{P}": "Psychic", "{F}": "Fighting", "{D}": "Darkness", "{M}": "Metal",
    "{C}": "Colorless", "{A}": "Any", "竜": "Dragon",
}


def parse_energy_cost(cost_str: str) -> dict:
    """Parse '{R}{R}●' → {'Fire': 2, 'Colorless': 1}. '●' means Colorless."""
    if not cost_str or cost_str.strip() == "n/a":
        return {}
    counts: dict[str, int] = defaultdict(int)
    for sym, name in TYPE_MAP.items():
        counts[name] += cost_str.count(sym)
    counts["Colorless"] += cost_str.count("●")
    return {k: v for k, v in counts.items() if v > 0}


def total_energy_cost(cost_parsed: dict) -> int:
    return sum(cost_parsed.values())


def safe_int(s: str) -> int | None:
    s = s.strip()
    return int(s) if s.lstrip("-").isdigit() else None


def build() -> None:
    csv_path = Path("data/EN_Card_Data.csv")
    out_path = Path("data/card_lookup.json")

    cards: dict[str, dict] = {}

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = row["Card ID"].strip()
            if not cid:
                continue

            if cid not in cards:
                hp_raw = row["HP"].strip()
                retreat_raw = row["Retreat"].strip()
                cards[cid] = {
                    "card_id": int(cid),
                    "name": row["Card Name"].strip(),
                    "expansion": row["Expansion"].strip() or None,
                    "collection_no": row["Collection No."].strip(),
                    "stage": row["Stage (Pokémon)/Type (Energy and Trainer)"].strip(),
                    "rule": row["Rule"].strip(),
                    "previous_stage": row["Previous stage"].strip() if row["Previous stage"].strip() not in ("n/a", "") else None,
                    "hp": safe_int(hp_raw),
                    "type": row["Type"].strip() or None,
                    "weakness": row["Weakness"].strip() or None,
                    "resistance": row["Resistance (Type)"].strip() or None,
                    "retreat": safe_int(retreat_raw),
                    "moves": [],
                    "is_ex": row["Rule"].strip() in ("Pokémon ex", "Mega Pokémon ex"),
                    "is_ace_spec": row["Rule"].strip() == "ACE SPEC",
                    "is_supporter": row["Stage (Pokémon)/Type (Energy and Trainer)"].strip() == "Supporter",
                    "is_item": row["Stage (Pokémon)/Type (Energy and Trainer)"].strip() in ("Item", "Pokémon Tool"),
                    "is_stadium": row["Stage (Pokémon)/Type (Energy and Trainer)"].strip() == "Stadium",
                    "is_energy": row["Stage (Pokémon)/Type (Energy and Trainer)"].strip() in ("Basic Energy", "Special Energy"),
                    "is_pokemon": row["Stage (Pokémon)/Type (Energy and Trainer)"].strip() in (
                        "Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"
                    ),
                    "effect": row["Effect Explanation"].strip() if row["Effect Explanation"].strip() != "n/a" else None,
                }

            move_name = row["Move Name"].strip()
            if move_name and move_name != "n/a":
                dmg_raw = row["Damage"].strip()
                cost_str = row["Cost"].strip()
                cost_parsed = parse_energy_cost(cost_str)
                is_ability = move_name.startswith("[Ability]") or move_name.startswith("[Tera]")
                dmg_value = safe_int(dmg_raw) if dmg_raw != "n/a" else None

                cards[cid]["moves"].append({
                    "name": move_name,
                    "is_ability": is_ability,
                    "cost": cost_str if cost_str != "n/a" else "",
                    "cost_parsed": cost_parsed,
                    "total_energy": total_energy_cost(cost_parsed),
                    "damage": dmg_value,
                    "damage_text": dmg_raw if dmg_raw != "n/a" else None,
                    "effect": row["Effect Explanation"].strip() if row["Effect Explanation"].strip() != "n/a" else None,
                })

    # Sort by card_id
    ordered = dict(sorted(cards.items(), key=lambda x: int(x[0])))
    out_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built {out_path}: {len(ordered)} cards")

    # Verification
    assert len(ordered) == 1267, f"Expected 1267 cards, got {len(ordered)}"
    ids = [int(k) for k in ordered]
    assert ids == list(range(1, 1268)), "Card IDs are not continuous 1–1267"
    print("Verification passed: 1267 continuous IDs.")


if __name__ == "__main__":
    build()
