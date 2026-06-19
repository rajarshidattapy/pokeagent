"""Verified card IDs and constants from EN_Card_Data.csv."""

# ── Energy ──────────────────────────────────────────────────────────────────

BASIC_ENERGY_IDS = list(range(1, 9))     # {G} {R} {W} {L} {P} {F} {D} {M}
SPECIAL_ENERGY_IDS = list(range(9, 21))  # IDs 9-20
ALL_ENERGY_IDS = list(range(1, 21))

# Maps CSV Type field → feature vector index (0-8)
ENERGY_TYPE_INDEX: dict[str, int] = {
    "{G}": 0, "{R}": 1, "{W}": 2, "{L}": 3,
    "{P}": 4, "{F}": 5, "{D}": 6, "{M}": 7, "{C}": 8,
}

# Basic energy card_id → type symbol
BASIC_ENERGY_TYPE: dict[int, str] = {
    1: "{G}", 2: "{R}", 3: "{W}", 4: "{L}",
    5: "{P}", 6: "{F}", 7: "{D}", 8: "{M}",
}

# ── Card ranges ──────────────────────────────────────────────────────────────

SUPPORTER_ID_MIN = 1181
SUPPORTER_ID_MAX = 1241
STADIUM_ID_MIN   = 1242
STADIUM_ID_MAX   = 1267

# ── Key individual card IDs ──────────────────────────────────────────────────

# Supporters
BOSS_ORDERS_ID  = 1182   # Gust: switch in opponent's benched Pokémon
JUDGE_ID        = 1213   # Shuffle both hands, each draws 4
LILLIES_DET_ID  = 1227   # Draw 6 (8 if 6 prizes remaining)
LACEY_ID        = 1199   # Draw 4 (8 if opponent ≤ 3 prizes)
CYRANO_ID       = 1205   # Search up to 3 Pokémon ex
BROCKS_ID       = 1210   # Search up to 2 Basic or 1 Evolution

# Items
ULTRA_BALL_ID        = 1121   # Discard 2 → search any Pokémon
MASTER_BALL_ID       = 1125   # ACE SPEC: search any Pokémon (no discard)
NIGHT_STRETCHER_ID   = 1097   # Recover 1 Pokémon or energy from discard
RARE_CANDY_ID        = 1079   # Basic → Stage 2
ENERGY_SEARCH_ID     = 1119   # Search 1 basic energy
ENERGY_SEARCH_PRO_ID = 1100   # ACE SPEC: search any # of different basic energies
SCRAMBLE_SWITCH_ID   = 1107   # ACE SPEC: switch + move all energy
PRIME_CATCHER_ID     = 1088   # ACE SPEC: gust + your switch
MIRACLE_HEADSET_ID   = 1109   # ACE SPEC: recover 2 supporters from discard
SCOOP_UP_CYCLONE_ID  = 1093   # ACE SPEC: pick up any Pokémon + attached

# Tools
HERO_CAPE_ID   = 1159   # ACE SPEC: +100 HP
MAX_BELT_ID    = 1158   # ACE SPEC: +50 dmg vs Pokémon ex

# Stadiums
JAMMING_TOWER_ID     = 1246   # All tools have no effect
TR_WATCHTOWER_ID     = 1256   # {C} Pokémon lose abilities
TR_FACTORY_ID        = 1257   # Draw 2 when TR Supporter played
RISKY_RUINS_ID       = 1260   # 2 dmg counters on benched non-Dark basics
LIVELY_STADIUM_ID    = 1251   # +30 HP to all Basic Pokémon
NS_CASTLE_ID         = 1253   # No retreat cost for N's Pokémon

# Recommended deck attacker
MEGA_DIANCIE_EX_ID   = 766    # Basic, 270HP, {P}{P}→240 dmg, Diamond Coat -30 dmg
MEGA_ABSOL_EX_ID     = 687    # Basic, 280HP, {D}{D}●→200 dmg + hand disruption
MEGA_KANGASKHAN_ID   = 756    # Basic, 300HP, Ability: draw 2 from Active

# ── ACE SPEC card IDs (exactly 1 allowed per deck) ────────────────────────

ACE_SPEC_IDS: frozenset[int] = frozenset({
    10, 12, 13,                                         # Special Energies
    1080, 1082, 1085, 1088, 1089, 1092, 1093,           # Items
    1095, 1096, 1100, 1104, 1107, 1109, 1110, 1111,
    1125, 1126, 1128,
    1155, 1158, 1159, 1165, 1167, 1169,                 # Tools
    1247, 1249,                                          # Stadiums
})

# ── Top 10 stadiums for one-hot encoding in feature vector ────────────────

TOP_STADIUMS: list[int] = [
    1256,   # Team Rocket's Watchtower
    1257,   # Team Rocket's Factory
    1260,   # Risky Ruins
    1246,   # Jamming Tower
    1251,   # Lively Stadium
    1253,   # N's Castle
    1247,   # Neutralization Zone
    1244,   # Full Metal Lab
    1242,   # Community Center
    1258,   # Granite Cave
]

# ── Supporter draw counts (used by encoder action features) ──────────────

SUPPORTER_DRAW_COUNT: dict[int, int] = {
    LILLIES_DET_ID: 6,    # 8 if 6 prizes; use 6 as conservative estimate
    LACEY_ID:       4,    # 8 if opp ≤3 prizes
    JUDGE_ID:       4,
    1216:           5,    # Team Rocket's Ariana: draw to 5
    1224:           3,    # Cheren
    1236:           3,    # Urbain
    1226:           4,    # Lt. Surge's Bargain (conditional)
    1200:           4,    # Kofu (draw 4)
    1181:           2,    # Billy & O'Nare
    1214:           2,    # Emcee's Hype
    1203:           5,    # Surfer (draw to 5)
    1199:           4,    # Lacey
}
