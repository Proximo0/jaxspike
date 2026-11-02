"""Skill, job, effect, target and buff tables for the skills phase (see docs/m2_skills_brief.md).

Indexing: per-skill arrays are int32 ``[N_SKILLS]`` indexed by skill id (row 0 = SKILL_NONE, all zeros).
Per-level arrays are int32 ``[N_SKILLS, MAX_SKILL_LEVEL + 1]`` indexed ``[skill_id, level]``; column 0 (level 0,
"not learned") is always zero and levels beyond a skill's SKILL_MAX_LEVEL repeat the max-level values.
Time is in ticks (1 tick = 120 ms, 1 s = 8 ticks). Power is x1000 per hit. Built from plain Python tuples and
converted with ``jnp.asarray`` at import; no other JAX ops.
"""
import jax.numpy as jnp

from game.data.elements import ELEM_FIRE, ELEM_GHOST, ELEM_NEUTRAL, ELEM_POISON, ELEM_WATER, ELEM_WIND

JOB_NOVICE, JOB_SWORDSMAN, JOB_MAGE, JOB_ARCHER, JOB_ACOLYTE, JOB_THIEF, JOB_MERCHANT = range(7)
N_JOBS = 7

SKILL_NONE = 0
(SK_BASIC_SKILL, SK_FIRST_AID, SK_PLAY_DEAD,
 SK_BASH, SK_MAGNUM_BREAK, SK_PROVOKE, SK_ENDURE, SK_HP_RECOVERY, SK_SWORD_MASTERY,
 SK_FIRE_BOLT, SK_COLD_BOLT, SK_LIGHTNING_BOLT, SK_NAPALM_BEAT, SK_SOUL_STRIKE, SK_FIRE_WALL,
 SK_DOUBLE_STRAFE, SK_ARROW_SHOWER, SK_OWLS_EYE, SK_VULTURES_EYE,
 SK_HEAL, SK_BLESSING, SK_INCREASE_AGI, SK_ANGELUS, SK_CURE,
 SK_DOUBLE_ATTACK, SK_STEAL, SK_HIDING, SK_ENVENOM,
 SK_MAMMONITE, SK_DISCOUNT, SK_VENDING, SK_PUSHCART) = range(1, 33)
N_SKILLS = 33
MAX_SKILL_LEVEL = 10

EFFECT_PASSIVE, EFFECT_ATK, EFFECT_MATK, EFFECT_HEAL, EFFECT_BUFF, EFFECT_NONE = range(6)
TARGET_SELF, TARGET_ALLY, TARGET_ENEMY, TARGET_GROUND = range(4)
(BUFF_NONE, BUFF_BLESSING, BUFF_INC_AGI, BUFF_ANGELUS, BUFF_ENDURE, BUFF_PROVOKE, BUFF_HIDING,
 BUFF_PLAY_DEAD) = range(8)
N_BUFFS = 8

SP_REGEN_TICKS = 8
SP_REGEN_FLAT = 1  # per regen: SP_REGEN_FLAT + max_sp // 100
SP_START = 40

# First castable skill per job, indexed by job id (used by the bot policy).
JOB_FIRST_SKILL = (SK_FIRST_AID, SK_BASH, SK_FIRE_BOLT, SK_DOUBLE_STRAFE, SK_HEAL, SK_ENVENOM, SK_MAMMONITE)


def _row(f, max_level=10):
    """Per-level row of 11 ints: column 0 = 0, levels 1..10 = f(min(level, max_level)) (beyond max repeats max)."""
    return (0,) + tuple(f(min(n, max_level)) for n in range(1, 11))


def _flat(v):
    return lambda n: v


def _lin(a, b):
    return lambda n: a + b * n


_Z = (0,) * 11

# sid -> (name, job, max_level, effect, element, target, aoe, buff, hits, flat_heal,
#         sp, cast, cooldown, range, power, buff_ticks)   (the last six are per-level rows)
_SK = {}


def _def(sid, name, job, maxl, effect, elem, target, aoe=0, buff=0, hits=1, flat=0,
         sp=_Z, cast=_Z, cd=_Z, rng=_Z, power=_Z, bt=_Z):
    _SK[sid] = (name, job, maxl, effect, elem, target, aoe, buff, hits, flat, sp, cast, cd, rng, power, bt)


def _passive(sid, name, job, maxl=10):
    _def(sid, name, job, maxl, EFFECT_PASSIVE, ELEM_NEUTRAL, TARGET_SELF)


# Novice
_passive(SK_BASIC_SKILL, "Basic Skill", JOB_NOVICE, 9)
_def(SK_FIRST_AID, "First Aid", JOB_NOVICE, 1, EFFECT_HEAL, ELEM_NEUTRAL, TARGET_SELF, flat=5,
     sp=_row(_flat(3), 1), cd=_row(_flat(8), 1))
_def(SK_PLAY_DEAD, "Play Dead", JOB_NOVICE, 1, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_SELF, buff=BUFF_PLAY_DEAD,
     sp=_row(_flat(5), 1), cd=_row(_flat(8), 1), bt=_row(_flat(100000), 1))
# Swordsman
_def(SK_BASH, "Bash", JOB_SWORDSMAN, 10, EFFECT_ATK, ELEM_NEUTRAL, TARGET_ENEMY,
     sp=_row(lambda n: 8 if n <= 5 else 15), cd=_row(_flat(4)), rng=_row(_flat(1)), power=_row(_lin(1000, 300)))
_def(SK_MAGNUM_BREAK, "Magnum Break", JOB_SWORDSMAN, 10, EFFECT_ATK, ELEM_FIRE, TARGET_SELF, aoe=1,
     sp=_row(_flat(30)), cd=_row(_flat(16)), power=_row(_lin(1000, 200)))
_def(SK_PROVOKE, "Provoke", JOB_SWORDSMAN, 10, EFFECT_NONE, ELEM_NEUTRAL, TARGET_ENEMY,
     sp=_row(_lin(3, 1)), cd=_row(_flat(8)), rng=_row(_flat(9)))
_def(SK_ENDURE, "Endure", JOB_SWORDSMAN, 10, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_SELF, buff=BUFF_ENDURE,
     sp=_row(_flat(10)), cd=_row(_flat(8)), bt=_row(_flat(80)))
_passive(SK_HP_RECOVERY, "HP Recovery", JOB_SWORDSMAN)
_passive(SK_SWORD_MASTERY, "Sword Mastery", JOB_SWORDSMAN)
# Mage
for _sid, _name, _el in ((SK_FIRE_BOLT, "Fire Bolt", ELEM_FIRE), (SK_COLD_BOLT, "Cold Bolt", ELEM_WATER),
                         (SK_LIGHTNING_BOLT, "Lightning Bolt", ELEM_WIND)):
    _def(_sid, _name, JOB_MAGE, 10, EFFECT_MATK, _el, TARGET_ENEMY,
         sp=_row(_lin(10, 2)), cast=_row(_lin(0, 6)), cd=_row(_flat(8)), rng=_row(_flat(9)),
         power=_row(_lin(0, 1000)))
_def(SK_NAPALM_BEAT, "Napalm Beat", JOB_MAGE, 10, EFFECT_MATK, ELEM_GHOST, TARGET_ENEMY, aoe=1,
     sp=_row(lambda n: 9 + (n - 1) // 3), cast=_row(_flat(8)), cd=_row(_flat(8)), rng=_row(_flat(9)),
     power=_row(_lin(600, 100)))
_def(SK_SOUL_STRIKE, "Soul Strike", JOB_MAGE, 10, EFFECT_MATK, ELEM_GHOST, TARGET_ENEMY,
     sp=_row(lambda n: (18, 14, 24, 20, 30, 26, 36, 32, 42, 38)[n - 1]), cast=_row(lambda n: 4 * ((n + 1) // 2)),
     cd=_row(_flat(8)), rng=_row(_flat(9)), power=_row(lambda n: ((n + 1) // 2) * 1000))
_def(SK_FIRE_WALL, "Fire Wall", JOB_MAGE, 10, EFFECT_MATK, ELEM_FIRE, TARGET_GROUND, aoe=1,
     sp=_row(_flat(40)), cast=_row(_flat(18)), cd=_row(_flat(16)), rng=_row(_flat(9)), power=_row(_lin(0, 500)))
# Archer
_def(SK_DOUBLE_STRAFE, "Double Strafe", JOB_ARCHER, 10, EFFECT_ATK, ELEM_NEUTRAL, TARGET_ENEMY, hits=2,
     sp=_row(_flat(12)), cd=_row(_flat(4)), rng=_row(_flat(9)), power=_row(_lin(900, 100)))
_def(SK_ARROW_SHOWER, "Arrow Shower", JOB_ARCHER, 10, EFFECT_ATK, ELEM_NEUTRAL, TARGET_GROUND, aoe=1,
     sp=_row(_flat(15)), cd=_row(_flat(8)), rng=_row(_flat(9)), power=_row(_lin(700, 50)))
_passive(SK_OWLS_EYE, "Owl's Eye", JOB_ARCHER)
_passive(SK_VULTURES_EYE, "Vulture's Eye", JOB_ARCHER)
# Acolyte
_def(SK_HEAL, "Heal", JOB_ACOLYTE, 10, EFFECT_HEAL, ELEM_NEUTRAL, TARGET_ALLY,
     sp=_row(_lin(10, 3)), cast=_row(_flat(8)), cd=_row(_flat(8)), rng=_row(_flat(9)), power=_row(_lin(500, 1000)))
_def(SK_BLESSING, "Blessing", JOB_ACOLYTE, 10, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_ALLY, buff=BUFF_BLESSING,
     sp=_row(_lin(24, 4)), cast=_row(_flat(8)), cd=_row(_flat(8)), rng=_row(_flat(9)),
     bt=_row(lambda n: (40 + 20 * n) * 8))
_def(SK_INCREASE_AGI, "Increase AGI", JOB_ACOLYTE, 10, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_ALLY, buff=BUFF_INC_AGI,
     sp=_row(_lin(15, 3)), cast=_row(_flat(8)), cd=_row(_flat(8)), rng=_row(_flat(9)),
     bt=_row(lambda n: (40 + 20 * n) * 8))
_def(SK_ANGELUS, "Angelus", JOB_ACOLYTE, 10, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_ALLY, buff=BUFF_ANGELUS,
     sp=_row(_lin(20, 3)), cast=_row(_flat(4)), cd=_row(_flat(8)), rng=_row(_flat(9)),
     bt=_row(lambda n: 30 * n * 8))
_def(SK_CURE, "Cure", JOB_ACOLYTE, 1, EFFECT_NONE, ELEM_NEUTRAL, TARGET_ALLY,
     sp=_row(_flat(15), 1), cast=_row(_flat(4), 1), cd=_row(_flat(8), 1), rng=_row(_flat(9), 1))
# Thief
_passive(SK_DOUBLE_ATTACK, "Double Attack", JOB_THIEF)
_def(SK_STEAL, "Steal", JOB_THIEF, 10, EFFECT_NONE, ELEM_NEUTRAL, TARGET_ENEMY,
     sp=_row(_flat(10)), cd=_row(_flat(8)), rng=_row(_flat(1)))
_def(SK_HIDING, "Hiding", JOB_THIEF, 10, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_SELF, buff=BUFF_HIDING,
     sp=_row(_flat(10)), cd=_row(_flat(8)), bt=_row(lambda n: (30 + 30 * n) * 8))
_def(SK_ENVENOM, "Envenom", JOB_THIEF, 10, EFFECT_ATK, ELEM_POISON, TARGET_ENEMY,
     sp=_row(_flat(12)), cd=_row(_flat(4)), rng=_row(_flat(1)), power=_row(_lin(1000, 150)))
# Merchant
_def(SK_MAMMONITE, "Mammonite", JOB_MERCHANT, 10, EFFECT_ATK, ELEM_NEUTRAL, TARGET_ENEMY,
     sp=_row(_flat(5)), cd=_row(_flat(4)), rng=_row(_flat(1)), power=_row(_lin(1000, 500)))
_passive(SK_DISCOUNT, "Discount", JOB_MERCHANT)
_def(SK_VENDING, "Vending", JOB_MERCHANT, 10, EFFECT_NONE, ELEM_NEUTRAL, TARGET_SELF,
     sp=_row(_flat(30)), cd=_row(_flat(16)))
_passive(SK_PUSHCART, "Pushcart", JOB_MERCHANT)

assert sorted(_SK) == list(range(1, N_SKILLS))


def _vec(i, row0=0):
    return jnp.asarray((row0,) + tuple(_SK[s][i] for s in range(1, N_SKILLS)), dtype=jnp.int32)


def _mat(i):
    return jnp.asarray((_Z,) + tuple(_SK[s][i] for s in range(1, N_SKILLS)), dtype=jnp.int32)


SKILL_NAME = ("None",) + tuple(_SK[s][0] for s in range(1, N_SKILLS))
SKILL_JOB = _vec(1)
SKILL_MAX_LEVEL = _vec(2)
SKILL_EFFECT = _vec(3)
SKILL_ELEMENT = _vec(4)
SKILL_TARGET = _vec(5)
SKILL_AOE = _vec(6)
SKILL_BUFF = _vec(7)
SKILL_HITS = _vec(8, row0=1)  # row 0 = 1 so a multiply never zeroes damage
SKILL_FLAT_HEAL = _vec(9)
SKILL_SP = _mat(10)
SKILL_CAST = _mat(11)
SKILL_COOLDOWN = _mat(12)
SKILL_RANGE = _mat(13)
SKILL_POWER = _mat(14)
SKILL_BUFF_TICKS = _mat(15)
