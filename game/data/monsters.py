"""Monster kinds of the RO world (ro/world). Plain tuples, one row per kind; ids are fixed.

Ids 0 (Poring) and 1 (Orc Warrior) keep the legacy rat/orc slots (MONSTER_KIND_RAT/ORC), so
worlds built with the old config still work. The rest rise in level. Lines for config.py at merge:
    MONSTER_HP = game.data.monsters.MONSTER_HP_TUPLE
    MONSTER_DAMAGE = game.data.monsters.MONSTER_DAMAGE_TUPLE
"""
from collections import namedtuple

import jax.numpy as jnp

ELEMENTS = ("Neutral", "Water", "Earth", "Fire", "Wind", "Poison", "Holy", "Shadow", "Ghost", "Undead")
_E = {n: i for i, n in enumerate(ELEMENTS)}

Monster = namedtuple("Monster", "name hp atk_min atk_max def_ element level base_exp job_exp aggro move_ticks size mvp")

# name, hp, atk_min, atk_max, def, element, level, base_exp, job_exp, aggro, move_ticks, size, mvp
_ROWS = (
    ("Poring", 40, 1, 3, 0, "Water", 1, 2, 1, False, 2, 1, False),
    ("Orc Warrior", 120, 6, 9, 5, "Earth", 14, 60, 40, True, 1, 2, False),
    ("Fabre", 55, 2, 4, 0, "Earth", 2, 4, 2, False, 2, 0, False),
    ("Lunatic", 60, 2, 5, 1, "Neutral", 3, 6, 3, False, 1, 0, False),
    ("Drops", 70, 3, 5, 2, "Fire", 3, 7, 4, False, 2, 0, False),
    ("Poporing", 90, 4, 7, 2, "Poison", 4, 10, 6, False, 2, 1, False),
    ("Willow", 100, 4, 7, 3, "Earth", 4, 11, 7, False, 2, 1, False),
    ("Rocker", 130, 5, 9, 4, "Earth", 6, 16, 10, False, 2, 1, False),
    ("Spore", 150, 6, 10, 3, "Water", 7, 20, 12, False, 2, 1, False),
    ("Thief Bug", 140, 6, 11, 5, "Shadow", 8, 24, 15, True, 1, 0, False),
    ("Zombie", 220, 8, 14, 6, "Undead", 10, 32, 20, True, 3, 1, False),
    ("Wolf", 240, 9, 16, 8, "Earth", 12, 40, 26, True, 1, 1, False),
    ("Skeleton", 280, 10, 18, 10, "Undead", 14, 48, 30, True, 2, 1, False),
    ("Peco Peco", 320, 11, 20, 8, "Fire", 16, 56, 35, False, 1, 2, False),
    ("Hode", 360, 12, 22, 10, "Earth", 18, 66, 42, True, 2, 1, False),
    ("Soldier Skeleton", 420, 14, 26, 14, "Undead", 20, 80, 50, True, 2, 1, False),
    ("Bigfoot", 520, 16, 30, 12, "Earth", 24, 100, 62, True, 2, 2, False),
    ("Mummy", 600, 18, 34, 16, "Undead", 28, 120, 75, True, 3, 1, False),
    ("Golden Thief Bug", 3000, 60, 100, 40, "Fire", 35, 1500, 900, True, 1, 2, True),
    ("Eddga", 5000, 80, 120, 30, "Fire", 45, 3000, 1800, True, 1, 2, True),
)

MONSTERS = tuple(Monster(r[0], r[1], r[2], r[3], r[4], _E[r[5]], *r[6:]) for r in _ROWS)
N_KINDS = len(MONSTERS)
assert N_KINDS == 20
KIND_BY_NAME = {m.name: i for i, m in enumerate(MONSTERS)}
assert len(KIND_BY_NAME) == N_KINDS and KIND_BY_NAME["Poring"] == 0 and KIND_BY_NAME["Orc Warrior"] == 1

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
