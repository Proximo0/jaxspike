"""Skill, job, effect, target and buff tables for the skills phase (see docs/m2_skills_brief.md).

Indexing: per-skill arrays are int32 ``[N_SKILLS]`` indexed by skill id (row 0 = SKILL_NONE, all zeros).
Per-level arrays are int32 ``[N_SKILLS, MAX_SKILL_LEVEL + 1]`` indexed ``[skill_id, level]``; column 0 (level 0,
"not learned") is always zero and levels beyond a skill's SKILL_MAX_LEVEL repeat the max-level values.
Time is in ticks (game.config.TICK_MS each; written here in RO milliseconds and rounded with ``_ms``), except
SKILL_AFTER_DELAY_MS which stays in ms. Power is x1000 per hit. Built from plain Python tuples and converted with
``jnp.asarray`` at import; no other JAX ops.

Numbers are classic pre-renewal RO (rAthena db/pre-re/skill_db.yml, iRO wiki classic pages; docs/ro_fidelity.md).
"""
import jax.numpy as jnp

from game.config import TICK_MS
from game.data.elements import (ELEM_FIRE, ELEM_GHOST, ELEM_HOLY, ELEM_NEUTRAL, ELEM_POISON, ELEM_WATER,
                                ELEM_WIND)

JOB_NOVICE, JOB_SWORDSMAN, JOB_MAGE, JOB_ARCHER, JOB_ACOLYTE, JOB_THIEF, JOB_MERCHANT = range(7)
N_JOBS = 7

SKILL_NONE = 0
(SK_BASIC_SKILL, SK_FIRST_AID, SK_PLAY_DEAD,
 SK_BASH, SK_MAGNUM_BREAK, SK_PROVOKE, SK_ENDURE, SK_HP_RECOVERY, SK_SWORD_MASTERY,
 SK_FIRE_BOLT, SK_COLD_BOLT, SK_LIGHTNING_BOLT, SK_NAPALM_BEAT, SK_SOUL_STRIKE, SK_FIRE_WALL,
 SK_DOUBLE_STRAFE, SK_ARROW_SHOWER, SK_OWLS_EYE, SK_VULTURES_EYE,
 SK_HEAL, SK_BLESSING, SK_INCREASE_AGI, SK_ANGELUS, SK_CURE,
 SK_DOUBLE_ATTACK, SK_STEAL, SK_HIDING, SK_ENVENOM,
 SK_MAMMONITE, SK_DISCOUNT, SK_VENDING, SK_PUSHCART,
 # appended by ro-fidelity (ids 33..41; never renumber)
 SK_FROST_DIVER, SK_STONE_CURSE, SK_IMPROVE_CONCENTRATION, SK_TELEPORT, SK_WARP_PORTAL,
 SK_DETOXIFY, SK_IMPROVE_DODGE, SK_OVERCHARGE, SK_ITEM_APPRAISAL) = range(1, 42)
N_SKILLS = 42
MAX_SKILL_LEVEL = 10

EFFECT_PASSIVE, EFFECT_ATK, EFFECT_MATK, EFFECT_HEAL, EFFECT_BUFF, EFFECT_NONE = range(6)
TARGET_SELF, TARGET_ALLY, TARGET_ENEMY, TARGET_GROUND = range(4)
(BUFF_NONE, BUFF_BLESSING, BUFF_INC_AGI, BUFF_ANGELUS, BUFF_ENDURE, BUFF_PROVOKE, BUFF_HIDING,
 BUFF_PLAY_DEAD) = range(8)
N_BUFFS = 8


def _ms(ms):
    """RO milliseconds -> ticks, rounded to the nearest tick (a non-zero time is at least one tick)."""
    return 0 if ms <= 0 else max(1, (ms + TICK_MS // 2) // TICK_MS)


# SP regen of the skills.sp fallback pool: RO pre-re every 8 s standing, 1 + MaxSP/100 (+ INT/6 in ro_stats).
SP_REGEN_TICKS = _ms(8000)
SP_REGEN_FLAT = 1  # per regen: SP_REGEN_FLAT + max_sp // 100
SP_START = 40

# First castable skill per job, indexed by job id (used by the bot policy).
JOB_FIRST_SKILL = (SK_FIRST_AID, SK_BASH, SK_FIRE_BOLT, SK_DOUBLE_STRAFE, SK_HEAL, SK_ENVENOM, SK_MAMMONITE)

# After-cast delay (ACD) in ms when a skill's delay is "ASPD based" in RO (Bash, Mammonite, Double Strafe ...): the
# swing time (amotion) of the caster. Stored here as amotion at ASPD 150 until ro_skills reads SKILL_DELAY_ASPD.
_ASPD_DELAY_MS = 500
# Every skill use blocks the next for at least this long (rAthena conf/battle/skill.conf min_skill_delay_limit).
_MIN_DELAY_MS = 100


def _row(f, max_level=10):
    """Per-level row of 11 ints: column 0 = 0, levels 1..10 = f(min(level, max_level)) (beyond max repeats max)."""
    return (0,) + tuple(f(min(n, max_level)) for n in range(1, 11))


def _flat(v):
    return lambda n: v


def _lin(a, b):
    return lambda n: a + b * n


def _list(values):
    return lambda n: values[n - 1]


def _ticks(f):
    """Per-level function in ms -> per-level function in ticks."""
    return lambda n: _ms(f(n))


_Z = (0,) * 11

# sid -> (name, job, max_level, effect, element, target, aoe, buff, hits, flat_heal, aspd_delay,
#         sp, cast, cooldown, range, power, buff_ticks, after_delay_ms)   (the last seven are per-level rows)
_SK = {}


def _def(sid, name, job, maxl, effect, elem, target, aoe=0, buff=0, hits=1, flat=0, aspd_delay=False,
         sp=_Z, cast=_Z, cd=_Z, rng=_Z, power=_Z, bt=_Z, acd=None):
    """aoe = Chebyshev radius (1 = 3x3, 2 = 5x5); cast / cd / bt are rows in ticks, acd a row in ms."""
    if acd is None:
        acd = _row(_flat(_ASPD_DELAY_MS if aspd_delay else 0), maxl)
    acd = (0,) + tuple(max(v, _MIN_DELAY_MS) for v in acd[1:])
    _SK[sid] = (name, job, maxl, effect, elem, target, aoe, buff, hits, flat, int(aspd_delay),
                sp, cast, cd, rng, power, bt, acd)


def _passive(sid, name, job, maxl=10):
    _def(sid, name, job, maxl, EFFECT_PASSIVE, ELEM_NEUTRAL, TARGET_SELF)


# Novice
_passive(SK_BASIC_SKILL, "Basic Skill", JOB_NOVICE, 9)
_def(SK_FIRST_AID, "First Aid", JOB_NOVICE, 1, EFFECT_HEAL, ELEM_NEUTRAL, TARGET_SELF, flat=5,
     sp=_row(_flat(3), 1))
_def(SK_PLAY_DEAD, "Play Dead", JOB_NOVICE, 1, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_SELF, buff=BUFF_PLAY_DEAD,
     sp=_row(_flat(5), 1), bt=_row(_ticks(_flat(600_000)), 1))
# Swordsman
_def(SK_BASH, "Bash", JOB_SWORDSMAN, 10, EFFECT_ATK, ELEM_NEUTRAL, TARGET_ENEMY, aspd_delay=True,
     sp=_row(lambda n: 8 if n <= 5 else 15), rng=_row(_flat(1)),
     power=_row(_lin(1000, 300)))
_def(SK_MAGNUM_BREAK, "Magnum Break", JOB_SWORDSMAN, 10, EFFECT_ATK, ELEM_FIRE, TARGET_SELF, aoe=2,
     sp=_row(_flat(30)), power=_row(_lin(1000, 200)), acd=_row(_flat(2000)))
_def(SK_PROVOKE, "Provoke", JOB_SWORDSMAN, 10, EFFECT_NONE, ELEM_NEUTRAL, TARGET_ENEMY, aspd_delay=True,
     sp=_row(_lin(3, 1)), cd=_row(_ticks(_flat(1000))), rng=_row(_flat(9)))
_def(SK_ENDURE, "Endure", JOB_SWORDSMAN, 10, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_SELF, buff=BUFF_ENDURE,
     sp=_row(_flat(10)), cd=_row(_ticks(_flat(10_000))), bt=_row(_ticks(lambda n: 7000 + 3000 * n)))
_passive(SK_HP_RECOVERY, "HP Recovery", JOB_SWORDSMAN)
_passive(SK_SWORD_MASTERY, "Sword Mastery", JOB_SWORDSMAN)
# Mage: bolts hit `level` times at 100% MATK each (stored as one hit of level x 1000)
for _sid, _name, _el in ((SK_FIRE_BOLT, "Fire Bolt", ELEM_FIRE), (SK_COLD_BOLT, "Cold Bolt", ELEM_WATER),
                         (SK_LIGHTNING_BOLT, "Lightning Bolt", ELEM_WIND)):
    _def(_sid, _name, JOB_MAGE, 10, EFFECT_MATK, _el, TARGET_ENEMY,
         sp=_row(_lin(10, 2)), cast=_row(_ticks(lambda n: 700 * n)), rng=_row(_flat(9)),
         power=_row(_lin(0, 1000)), acd=_row(_lin(800, 200)))
_def(SK_NAPALM_BEAT, "Napalm Beat", JOB_MAGE, 10, EFFECT_MATK, ELEM_GHOST, TARGET_ENEMY, aoe=1,
     sp=_row(lambda n: 9 + 3 * ((n - 1) // 3)), cast=_row(_ticks(_flat(1000))), rng=_row(_flat(9)),
     power=_row(_lin(700, 100)), acd=_row(_list((1000, 1000, 1000, 900, 900, 800, 800, 700, 600, 500))))
_def(SK_SOUL_STRIKE, "Soul Strike", JOB_MAGE, 10, EFFECT_MATK, ELEM_GHOST, TARGET_ENEMY,
     sp=_row(_list((18, 14, 24, 20, 30, 26, 36, 32, 42, 38))), cast=_row(_ticks(_flat(500))),
     rng=_row(_flat(9)), power=_row(lambda n: ((n + 1) // 2) * 1000),
     acd=_row(_list((1200, 1000, 1400, 1200, 1600, 1400, 1800, 1600, 2000, 1800))))
# Fire Wall: RO puts a 1x3 wall that hits 4 + level times at 50% MATK each; v1 resolves one strike of 50% x level.
_def(SK_FIRE_WALL, "Fire Wall", JOB_MAGE, 10, EFFECT_MATK, ELEM_FIRE, TARGET_GROUND, aoe=1,
     sp=_row(_flat(40)), cast=_row(_ticks(lambda n: 2150 - 150 * n)), rng=_row(_flat(9)), power=_row(_lin(0, 500)))
# Archer
_def(SK_DOUBLE_STRAFE, "Double Strafe", JOB_ARCHER, 10, EFFECT_ATK, ELEM_NEUTRAL, TARGET_ENEMY, hits=2,
     aspd_delay=True, sp=_row(_flat(12)), rng=_row(_flat(9)), power=_row(_lin(900, 100)))
_def(SK_ARROW_SHOWER, "Arrow Shower", JOB_ARCHER, 10, EFFECT_ATK, ELEM_NEUTRAL, TARGET_GROUND, aoe=2,
     sp=_row(_flat(15)), rng=_row(_flat(9)), power=_row(_lin(750, 50)), acd=_row(_flat(1000)))
_passive(SK_OWLS_EYE, "Owl's Eye", JOB_ARCHER)
_passive(SK_VULTURES_EYE, "Vulture's Eye", JOB_ARCHER)
# Acolyte. Heal: RO heals floor((BaseLv + INT) / 8) * (4 + 8 * level); power stays the engine's MATK-scaled stand-in.
_def(SK_HEAL, "Heal", JOB_ACOLYTE, 10, EFFECT_HEAL, ELEM_HOLY, TARGET_ALLY,
     sp=_row(_lin(10, 3)), rng=_row(_flat(9)), power=_row(_lin(500, 1000)), acd=_row(_flat(1000)))
_def(SK_BLESSING, "Blessing", JOB_ACOLYTE, 10, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_ALLY, buff=BUFF_BLESSING,
     sp=_row(_lin(24, 4)), rng=_row(_flat(9)), bt=_row(_ticks(lambda n: (40 + 20 * n) * 1000)))
_def(SK_INCREASE_AGI, "Increase AGI", JOB_ACOLYTE, 10, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_ALLY, buff=BUFF_INC_AGI,
     sp=_row(_lin(15, 3)), cast=_row(_ticks(_flat(1000))), rng=_row(_flat(9)),
     bt=_row(_ticks(lambda n: (40 + 20 * n) * 1000)), acd=_row(_flat(1000)))
_def(SK_ANGELUS, "Angelus", JOB_ACOLYTE, 10, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_ALLY, buff=BUFF_ANGELUS,
     sp=_row(_lin(20, 3)), cast=_row(_ticks(_flat(500))), rng=_row(_flat(9)),
     bt=_row(_ticks(lambda n: 30_000 * n)), acd=_row(_flat(3500)))
_def(SK_CURE, "Cure", JOB_ACOLYTE, 1, EFFECT_NONE, ELEM_NEUTRAL, TARGET_ALLY,
     sp=_row(_flat(15), 1), rng=_row(_flat(9), 1), acd=_row(_flat(1000), 1))
# Thief
_passive(SK_DOUBLE_ATTACK, "Double Attack", JOB_THIEF)
_def(SK_STEAL, "Steal", JOB_THIEF, 10, EFFECT_NONE, ELEM_NEUTRAL, TARGET_ENEMY,
     sp=_row(_flat(10)), rng=_row(_flat(1)), acd=_row(_flat(1000)))
_def(SK_HIDING, "Hiding", JOB_THIEF, 10, EFFECT_BUFF, ELEM_NEUTRAL, TARGET_SELF, buff=BUFF_HIDING,
     sp=_row(_flat(10)), bt=_row(_ticks(lambda n: 30_000 * n)))
# Envenom: RO adds a flat +15 x level ATK (plus poison); power 1000 + 150 x level equals that at 100 ATK.
_def(SK_ENVENOM, "Envenom", JOB_THIEF, 10, EFFECT_ATK, ELEM_POISON, TARGET_ENEMY, aspd_delay=True,
     sp=_row(_flat(12)), rng=_row(_flat(1)), power=_row(_lin(1000, 150)))
# Merchant
_def(SK_MAMMONITE, "Mammonite", JOB_MERCHANT, 10, EFFECT_ATK, ELEM_NEUTRAL, TARGET_ENEMY, aspd_delay=True,
     sp=_row(_flat(5)), rng=_row(_flat(1)), power=_row(_lin(1000, 500)))
_passive(SK_DISCOUNT, "Discount", JOB_MERCHANT)
_def(SK_VENDING, "Vending", JOB_MERCHANT, 10, EFFECT_NONE, ELEM_NEUTRAL, TARGET_SELF, sp=_row(_flat(30)))
_passive(SK_PUSHCART, "Pushcart", JOB_MERCHANT)
# Appended (ids 33..41). Status effects (freeze, stone, poison cure) and teleports have no engine hook yet: they
# cost SP and play the cast, and the numbers are ready for the owners (docs/ro_fidelity.md).
_def(SK_FROST_DIVER, "Frost Diver", JOB_MAGE, 10, EFFECT_MATK, ELEM_WATER, TARGET_ENEMY,
     sp=_row(lambda n: 26 - n), cast=_row(_ticks(_flat(800))), rng=_row(_flat(9)),
     power=_row(_lin(1000, 100)), acd=_row(_flat(1500)))
_def(SK_STONE_CURSE, "Stone Curse", JOB_MAGE, 10, EFFECT_NONE, ELEM_NEUTRAL, TARGET_ENEMY,
     sp=_row(lambda n: 26 - n), cast=_row(_ticks(_flat(1000))), rng=_row(_flat(2)))
_def(SK_IMPROVE_CONCENTRATION, "Improve Concentration", JOB_ARCHER, 10, EFFECT_NONE, ELEM_NEUTRAL, TARGET_SELF,
     sp=_row(_lin(20, 5)), bt=_row(_ticks(lambda n: (40 + 20 * n) * 1000)))
_def(SK_TELEPORT, "Teleport", JOB_ACOLYTE, 2, EFFECT_NONE, ELEM_NEUTRAL, TARGET_SELF, sp=_row(lambda n: 11 - n, 2))
_def(SK_WARP_PORTAL, "Warp Portal", JOB_ACOLYTE, 4, EFFECT_NONE, ELEM_NEUTRAL, TARGET_GROUND,
     sp=_row(lambda n: 38 - 3 * n, 4), cast=_row(_ticks(_flat(1000)), 4), rng=_row(_flat(9), 4),
     acd=_row(_flat(1000), 4))
_def(SK_DETOXIFY, "Detoxify", JOB_THIEF, 1, EFFECT_NONE, ELEM_NEUTRAL, TARGET_ALLY,
     sp=_row(_flat(10), 1), rng=_row(_flat(9), 1))
_passive(SK_IMPROVE_DODGE, "Improve Dodge", JOB_THIEF)
_passive(SK_OVERCHARGE, "Overcharge", JOB_MERCHANT)
_def(SK_ITEM_APPRAISAL, "Item Appraisal", JOB_MERCHANT, 1, EFFECT_NONE, ELEM_NEUTRAL, TARGET_SELF,
     sp=_row(_flat(10), 1))

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
SKILL_DELAY_ASPD = _vec(10)  # 1 = RO after-cast delay is the caster's amotion (ASPD based), not a fixed time
SKILL_SP = _mat(11)
SKILL_CAST = _mat(12)
SKILL_COOLDOWN = _mat(13)  # RO per-skill reuse delay (only Provoke 1 s and Endure 10 s among these)
SKILL_RANGE = _mat(14)
SKILL_POWER = _mat(15)
SKILL_BUFF_TICKS = _mat(16)
SKILL_AFTER_DELAY_MS = _mat(17)  # RO after-cast delay in ms (global: no skill until it ends); read by ro_skills
