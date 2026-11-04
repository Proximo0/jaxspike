"""Monster kinds of the RO world (ro/world). Plain tuples, one row per kind; ids are fixed.

Ids 0 (Blobbin) and 1 (Grukk Brute) keep the legacy rat/orc slots (MONSTER_KIND_RAT/ORC), so
worlds built with the old config still work. The rest rise in level. Lines for config.py at merge:
    MONSTER_HP = game.data.monsters.MONSTER_HP_TUPLE
    MONSTER_DAMAGE = game.data.monsters.MONSTER_DAMAGE_TUPLE
"""
from collections import namedtuple

import jax.numpy as jnp

ELEMENTS = ("Neutral", "Water", "Earth", "Fire", "Wind", "Poison", "Holy", "Shadow", "Ghost", "Undead")
_E = {n: i for i, n in enumerate(ELEMENTS)}

Monster = namedtuple("Monster", "name hp atk_min atk_max def_ element level base_exp job_exp aggro move_ticks size mvp agi vit dex")

# name, hp, atk_min, atk_max, def, element, level, base_exp, job_exp, aggro, move_ticks, size, mvp, agi, vit, dex
# (agi, vit, dex: rAthena pre-re monster stats; AGI dodges player hits, VIT is flat soft DEF, DEX raises monster hit)
# Classic pre-renewal numbers from rAthena db/pre-re/mob_db.yml (ro-fidelity, 2026-11-02): HP, Attack/Attack2,
# Defense (hard DEF, percent), Level, BaseExp/JobExp, Element (level 1 table; Moonbun/Pilferbug are Neutral 3,
# Hode Earth 2, Mummy Undead 2, GTB Fire 2), Size (0 small, 1 medium, 2 large), aggro = rAthena Ai 04/07/21.
_ROWS = (
    ("Blobbin", 50, 7, 10, 0, "Water", 1, 2, 1, False, 2, 1, False, 1, 1, 6),
    ("Grukk Brute", 1400, 104, 126, 10, "Earth", 24, 408, 160, True, 1, 1, False, 24, 48, 34),
    ("Wrigglet", 63, 8, 11, 0, "Earth", 2, 3, 2, False, 2, 0, False, 2, 4, 7),
    ("Moonbun", 60, 9, 12, 0, "Neutral", 3, 6, 2, False, 1, 0, False, 3, 3, 8),
    ("Emberblob", 55, 10, 13, 0, "Fire", 3, 4, 3, False, 2, 1, False, 3, 3, 12),
    ("Vexblob", 344, 59, 72, 0, "Poison", 14, 81, 44, False, 2, 1, False, 14, 14, 19),
    ("Bramblet", 95, 9, 12, 5, "Earth", 4, 5, 4, False, 2, 1, False, 4, 8, 9),
    ("Fiddlehop", 198, 24, 29, 5, "Earth", 9, 20, 16, False, 2, 1, False, 9, 18, 14),
    ("Capling", 510, 24, 48, 0, "Water", 16, 66, 108, False, 2, 1, False, 12, 12, 19),
    ("Pilferbug", 126, 18, 24, 5, "Neutral", 6, 17, 5, True, 1, 0, False, 6, 6, 11),
    ("Rotling", 534, 67, 79, 0, "Undead", 15, 50, 33, True, 3, 1, False, 8, 7, 15),
    ("Snarlfang", 919, 37, 46, 0, "Earth", 25, 329, 199, False, 1, 1, False, 20, 28, 32),
    ("Clackbone", 234, 39, 47, 10, "Undead", 10, 18, 14, False, 2, 1, False, 5, 10, 12),
    ("Gallopeck", 531, 50, 64, 0, "Fire", 19, 159, 72, False, 1, 2, False, 13, 13, 27),
    ("Sandgulper", 2282, 146, 177, 0, "Earth", 26, 550, 300, False, 2, 1, False, 26, 42, 49),
    ("Clackbone Guard", 2334, 221, 245, 10, "Undead", 29, 372, 226, True, 2, 1, False, 15, 22, 40),
    ("Grumblebear", 1619, 198, 220, 10, "Earth", 25, 310, 188, False, 2, 2, False, 25, 55, 20),
    ("Tombwrap", 5176, 305, 360, 0, "Undead", 37, 800, 602, True, 3, 1, False, 19, 32, 63),
    ("Gildscarab", 126000, 870, 1145, 60, "Fire", 64, 14300, 7150, True, 1, 2, True, 75, 35, 85),
    ("Scorchstripe", 152000, 1215, 1565, 15, "Fire", 65, 25025, 12870, True, 1, 2, True, 70, 85, 90),
)

MONSTERS = tuple(Monster(r[0], r[1], r[2], r[3], r[4], _E[r[5]], *r[6:]) for r in _ROWS)
N_KINDS = len(MONSTERS)
assert N_KINDS == 20
KIND_BY_NAME = {m.name: i for i, m in enumerate(MONSTERS)}
assert len(KIND_BY_NAME) == N_KINDS and KIND_BY_NAME["Blobbin"] == 0 and KIND_BY_NAME["Grukk Brute"] == 1

NAMES = tuple(m.name for m in MONSTERS)
MONSTER_HP_TUPLE = tuple(m.hp for m in MONSTERS)
MONSTER_DAMAGE_TUPLE = tuple(m.atk_max for m in MONSTERS)


def _arr(field, dtype=jnp.int32):
    return jnp.asarray([getattr(m, field) for m in MONSTERS], dtype)


HP = _arr("hp")
ATK_MIN = _arr("atk_min")
ATK_MAX = _arr("atk_max")
DEF = _arr("def_")
ELEMENT = _arr("element")
LEVEL = _arr("level")
BASE_EXP = _arr("base_exp")
JOB_EXP = _arr("job_exp")
AGGRO = _arr("aggro", jnp.bool_)
MOVE_TICKS = _arr("move_ticks")
SIZE = _arr("size")
MVP = _arr("mvp", jnp.bool_)
AGI = _arr("agi")
VIT = _arr("vit")
DEX = _arr("dex")

# RO monster modes and movement per kind (team ro-monsters, docs/round2_contract.md section 0 "Monster modes").
# Kept apart from _ROWS so number fixes there and mode fixes here do not collide. Values follow pre-renewal
# rAthena mob_db where known (walk speed ms/cell, mode bits), approximated otherwise; ro-fidelity may correct.
# AGGRO above stays the "aggressive" mode bit. Sight is capped by the spatial grid (game/spatial.py sees one
# CELL = 8 tiles around a query), so RO's 10-12 cell sight is 8 here until the grid grows.
#   assist: joins a kind-mate within sight that is fighting   looter: picks up drops within sight (Blobbin-style)
#   immovable: never moves (plants)   cast_sensor: placeholder, aggro on a nearby cast once casts expose a target
#   sight: cells (manhattan) to notice a player or drop   chase: cells (chebyshev) from the spawn before giving up
#   speed_ms: ms per cell (replaces move_ticks; ro_move reads SPEED_MS)   atk_range: cells   respawn_ms: field timer
ModeRow = namedtuple("ModeRow", "assist looter immovable cast_sensor sight chase speed_ms atk_range respawn_ms")
_MODE_ROWS = {
    "Blobbin": (False, True, False, False, 6, 10, 400, 1, 6000),
    "Grukk Brute": (False, False, False, False, 8, 14, 200, 1, 9000),
    "Wrigglet": (False, False, False, False, 6, 10, 400, 1, 6000),
    "Moonbun": (False, False, False, False, 6, 10, 200, 1, 6000),
    "Emberblob": (False, True, False, False, 6, 10, 400, 1, 6000),
    "Vexblob": (False, True, False, False, 7, 12, 300, 1, 7000),
    "Bramblet": (False, False, False, False, 7, 12, 200, 1, 7000),
    "Fiddlehop": (False, False, False, False, 7, 12, 200, 1, 7000),
    "Capling": (False, False, False, False, 7, 12, 200, 1, 7000),
    "Pilferbug": (True, True, False, False, 8, 14, 150, 1, 8000),
    "Rotling": (False, False, False, True, 7, 12, 400, 1, 9000),
    "Snarlfang": (True, False, False, False, 8, 14, 200, 1, 9000),
    "Clackbone": (False, False, False, True, 8, 14, 200, 1, 9000),
    "Gallopeck": (False, False, False, False, 8, 14, 200, 1, 9000),
    "Sandgulper": (False, False, False, False, 8, 14, 200, 1, 10000),
    "Clackbone Guard": (True, False, False, True, 8, 16, 200, 1, 10000),
    "Grumblebear": (False, False, False, False, 8, 16, 300, 1, 10000),
    "Tombwrap": (False, False, False, True, 8, 16, 300, 1, 12000),
    "Gildscarab": (True, False, False, True, 8, 20, 100, 1, 0),  # MVP: timer lives in game/ro_mvp.py
    "Scorchstripe": (False, False, False, True, 8, 20, 300, 1, 0),
}
assert set(_MODE_ROWS) == set(NAMES)
MODES = tuple(ModeRow(*_MODE_ROWS[n]) for n in NAMES)


def _mode_arr(field, dtype=jnp.int32):
    return jnp.asarray([getattr(r, field) for r in MODES], dtype)


ASSIST = _mode_arr("assist", jnp.bool_)
LOOTER = _mode_arr("looter", jnp.bool_)
IMMOVABLE = _mode_arr("immovable", jnp.bool_)
CAST_SENSOR = _mode_arr("cast_sensor", jnp.bool_)
SIGHT = _mode_arr("sight")
CHASE = _mode_arr("chase")
SPEED_MS = _mode_arr("speed_ms")
ATTACK_RANGE = _mode_arr("atk_range")
RESPAWN_MS = _mode_arr("respawn_ms")
# Idle random walk (RO: a short wander every few seconds), bounded to a box around the spawn tile.
WANDER_MIN_MS = 3000
WANDER_MAX_MS = 5000
WANDER_MIN_CELLS = 1
WANDER_MAX_CELLS = 5
WANDER_BOX = 6  # chebyshev cells from the spawn a wander goal may reach

# Attack delay per kind (ro4-balance): rAthena pre-re mob_db AttackDelay in ms (the time between two swings), from
# the classic counterparts of each role (jelly 1872, larva 1672, bunny 1456, ember blob 1372, ...; may be corrected).
# combat.py still swings every config.MONSTER_ATTACK_COOLDOWN ticks (8 = 800 ms, ~2.3x faster than the classic jelly);
# the line for combat (ro4-balance hand-back) swings every max(1, (ADELAY_MS + TICK_MS // 2) // TICK_MS) ticks.
# Kept in ms: game/config.py imports this module, so TICK_MS cannot be read here.
_ADELAY_MS = {
    "Blobbin": 1872, "Grukk Brute": 1864, "Wrigglet": 1672, "Moonbun": 1456, "Emberblob": 1372, "Vexblob": 1672,
    "Bramblet": 1672, "Fiddlehop": 1864, "Capling": 1872, "Pilferbug": 1288, "Rotling": 2612, "Snarlfang": 1054,
    "Clackbone": 2228, "Gallopeck": 1564, "Sandgulper": 1480, "Clackbone Guard": 2276, "Grumblebear": 1260,
    "Tombwrap": 1772, "Gildscarab": 768, "Scorchstripe": 872,
}
assert set(_ADELAY_MS) == set(NAMES)
ADELAY_MS = jnp.asarray([_ADELAY_MS[n] for n in NAMES], jnp.int32)
