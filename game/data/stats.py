"""M1 stats, jobs and exp tables. Plain int tuples plus matching jnp.int32 arrays (suffix _J) built at import."""
import jax.numpy as jnp

from game.config import MONSTER_HP, TICK_MS

# ---- ids ----
JOB_NOVICE, JOB_SWORDSMAN, JOB_MAGE, JOB_ARCHER, JOB_ACOLYTE, JOB_THIEF, JOB_MERCHANT = range(7)
NUM_JOBS = 7
STAT_STR, STAT_AGI, STAT_VIT, STAT_INT, STAT_DEX, STAT_LUK = range(6)
NUM_STATS = 6
NUM_MONSTER_KINDS = len(MONSTER_HP)

# ---- scalar constants ----
MAX_BASE_LEVEL = 99
MAX_JOB_LEVEL_NOVICE = 10
MAX_JOB_LEVEL_FIRST = 50
MAX_STAT = 99
STAT_POINTS_START = 48            # unspent status points of a fresh character
SKILL_POINTS_PER_JOB_LEVEL = 1
# SP regen cadence in ticks: RO pre-re regenerates SP every 8 s standing (4 s sitting; iRO wiki "SP Recovery").
SP_REGEN_TICKS = (8000 + TICK_MS // 2) // TICK_MS
DEATH_PENALTY_PERMILLE = 10       # 1% of the level's exp requirement
DEATH_PENALTY_MIN_LEVEL = 10
TICK_HZ_X1000 = 1000 * 1000 // TICK_MS   # ticks per second x1000 (8333 at 120 ms)

STAT_NAMES = ("str", "agi", "vit", "int", "dex", "luk")
JOB_NAMES = ("Novice", "Swordsman", "Mage", "Archer", "Acolyte", "Thief", "Merchant")


# ---- exp tables: rAthena db/pre-re/job_exp.yml (Novice + first jobs), verbatim ----
# BASE_EXP_TABLE: indexed by base level L, exp to reach L+1 (plain exp); 0 at index 0 and 99 (cap).
_BASE_RAW = (
    9, 16, 25, 36, 77, 112, 153, 200, 253, 320, 385, 490, 585, 700, 830, 970, 1120, 1260, 1420, 1620,
    1860, 1990, 2240, 2504, 2950, 3426, 3934, 4474, 6889, 7995, 9174, 10425, 11748, 13967, 15775, 17678, 19677,
    21773, 30543, 34212, 38065, 42102, 46323, 53026, 58419, 64041, 69892, 75973, 102468, 115254, 128692, 142784,
    157528, 178184, 196300, 215198, 234879, 255341, 330188, 365914, 403224, 442116, 482590, 536948, 585191,
    635278, 687211, 740988, 925400, 1473746, 1594058, 1718928, 1848355, 1982340, 2230113, 2386162, 2547417,
    2713878, 3206160, 3681024, 4022472, 4377024, 4744680, 5125440, 5767272, 6204000, 6655464, 7121664, 7602600,
    9738720, 11649960, 13643520, 18339300, 23836800, 35658000, 48687000, 58135000, 99999998,
)
BASE_EXP_TABLE = (0,) + _BASE_RAW + (0,)
BASE_EXP_TABLE_J = jnp.asarray(BASE_EXP_TABLE, jnp.int32)

# JOB_EXP_NOVICE_TABLE: indexed by job level 0..10, exp to reach the next job level; 0 at 0 and 10 (cap).
JOB_EXP_NOVICE_TABLE = (0, 10, 18, 28, 40, 91, 151, 205, 268, 340, 0)
# JOB_EXP_FIRST_TABLE: indexed by job level 0..50, exp to reach the next job level; 0 at 0 and 50 (cap).
JOB_EXP_FIRST_TABLE = (0,) + (
    30, 43, 58, 76, 116, 180, 220, 272, 336, 520, 604, 699, 802, 948, 1125, 1668, 1937, 2226, 3040, 3988, 5564,
    6272, 7021, 9114, 11473, 15290, 16891, 18570, 23229, 28359, 36478, 39716, 43088, 52417, 62495, 78160, 84175,
    90404, 107611, 125915, 153941, 191781, 204351, 248352, 286212, 386371, 409795, 482092, 509596,
) + (0,)
# JOB_EXP_TABLE: [NUM_JOBS][51] indexed by [job][job level]; row 0 novice padded with 0s, rows 1..6 first-job.
JOB_EXP_TABLE = (JOB_EXP_NOVICE_TABLE + (0,) * 40,) + (JOB_EXP_FIRST_TABLE,) * (NUM_JOBS - 1)
JOB_EXP_TABLE_J = jnp.asarray(JOB_EXP_TABLE, jnp.int32)

# MAX_JOB_LEVEL: indexed by job id, job level cap.
MAX_JOB_LEVEL = (MAX_JOB_LEVEL_NOVICE,) + (MAX_JOB_LEVEL_FIRST,) * (NUM_JOBS - 1)
MAX_JOB_LEVEL_J = jnp.asarray(MAX_JOB_LEVEL, jnp.int32)

# ---- HP / SP: exact RO pre-re per-job tables (rAthena db/pre-re/job_basepoints.yml, checked level by level) ----
# BaseHP(job, L) = 35 + 5 L + sum_{i=2..L} round(HP_JOB_A * i); BaseSP(job, L) = 10 + SP_JOB * L.
# MaxHP = BaseHP * (100 + VIT) / 100, MaxSP = BaseSP * (100 + INT) / 100 (iRO wiki "HP" / "SP", pre-renewal).
HP_JOB_A_X10 = (0, 7, 3, 5, 4, 5, 4)    # indexed by job id (rAthena HpFactor / 10)
SP_JOB = (1, 2, 6, 2, 5, 2, 3)          # indexed by job id (rAthena SpIncrease / 100)


def _job_hp(job, level):
    return 35 + 5 * level + sum((HP_JOB_A_X10[job] * i + 5) // 10 for i in range(2, level + 1))


# JOB_BASE_HP_TABLE / JOB_BASE_SP_TABLE: [NUM_JOBS][100] indexed by [job][base level], 0 at level 0.
JOB_BASE_HP_TABLE = tuple((0,) + tuple(_job_hp(j, L) for L in range(1, 100)) for j in range(NUM_JOBS))
JOB_BASE_SP_TABLE = tuple((0,) + tuple(10 + SP_JOB[j] * L for L in range(1, 100)) for j in range(NUM_JOBS))
JOB_BASE_HP_TABLE_J = jnp.asarray(JOB_BASE_HP_TABLE, jnp.int32)
JOB_BASE_SP_TABLE_J = jnp.asarray(JOB_BASE_SP_TABLE, jnp.int32)

# Interim shape the engine reads today (base table x job factor). BASE_HP_TABLE is the exact Archer/Thief curve and
# BASE_SP_TABLE the exact Swordsman/Archer/Thief line; the factors are the mean ratio of each job's exact table over
# levels 10-99 (first jobs; within 10% for Acolyte/Merchant, 19% Mage, 15% Swordsman); Novice is set for levels
# 1-3 (exact 40 HP at L1 so config.PLAYER_MAX_HP = 40 matches; +34% by L10), SP 11 -> 10 at L1.
# derived() should read JOB_BASE_*_TABLE_J[job, lvl] instead (docs/ro_fidelity.md, engine lines).
BASE_HP_TABLE = JOB_BASE_HP_TABLE[3]
BASE_SP_TABLE = JOB_BASE_SP_TABLE[3]
BASE_HP_TABLE_J = jnp.asarray(BASE_HP_TABLE, jnp.int32)
BASE_SP_TABLE_J = jnp.asarray(BASE_SP_TABLE, jnp.int32)

# JOB_HP_FACTOR: indexed by job id, x1000 multiplier on base HP.
JOB_HP_FACTOR = (1000, 1250, 730, 1000, 857, 1000, 857)
# JOB_SP_FACTOR: indexed by job id, x1000 multiplier on base SP.
JOB_SP_FACTOR = (850, 1000, 2781, 1000, 2335, 1000, 1445)

# ---- ASPD: rAthena db/pre-re/job_aspd.yml (base amotion per weapon type); ASPD = 200 - amotion / 10 ----
# JOB_WEAPON_ASPD: [NUM_JOBS][7] indexed by [job][weapon class WC_* of game/data/items.py: none (fist), dagger,
# 1h sword, bow, staff, mace, 1h axe]; 0 = the job cannot use that class. Stats then add
# (200 - base) * (4 AGI + DEX) / 1000 (amotion shrinks by (4 AGI + DEX) / 1000), cap 190.
#                  fist dag  swd  bow  stf  mace axe
JOB_WEAPON_ASPD = (
    (150, 135, 130,   0, 135, 130, 120),   # Novice
    (160, 150, 145,   0,   0, 135, 130),   # Swordsman
    (150, 140,   0,   0, 130,   0,   0),   # Mage
    (160, 140,   0, 130,   0,   0,   0),   # Archer
    (160,   0,   0,   0, 140, 140,   0),   # Acolyte
    (160, 150, 135, 120,   0,   0, 120),   # Thief
    (160, 140, 130,   0,   0, 130, 130),   # Merchant
)
JOB_WEAPON_ASPD_J = jnp.asarray(JOB_WEAPON_ASPD, jnp.int32)
# JOB_ASPD_BASE: indexed by job id, ASPD before stats with the job's usual starter weapon (Novice knife, Swordsman
# sword, Mage rod, Archer bow, Acolyte rod, Thief knife, Merchant axe). Interim until combat picks by weapon class.
JOB_ASPD_BASE = (135, 145, 130, 130, 140, 150, 130)
JOB_HP_FACTOR_J = jnp.asarray(JOB_HP_FACTOR, jnp.int32)
JOB_SP_FACTOR_J = jnp.asarray(JOB_SP_FACTOR, jnp.int32)
JOB_ASPD_BASE_J = jnp.asarray(JOB_ASPD_BASE, jnp.int32)

# MONSTER_BASE_EXP / MONSTER_JOB_EXP: indexed by monster kind, exp paid per kill, from the RO monster table
# (game/data/monsters.py; Poring 2/1 ... Eddga 3000/1800).
from game.data.monsters import MONSTERS as _MONSTERS  # noqa: E402
MONSTER_BASE_EXP = tuple(m.base_exp for m in _MONSTERS)
MONSTER_JOB_EXP = tuple(m.job_exp for m in _MONSTERS)
MONSTER_BASE_EXP_J = jnp.asarray(MONSTER_BASE_EXP, jnp.int32)
MONSTER_JOB_EXP_J = jnp.asarray(MONSTER_JOB_EXP, jnp.int32)

# STAT_POINTS_PER_LEVEL: indexed by the base level just reached, status points gained (plain points). RO pre-re
# (rAthena pc_gets_status_point: (old_level + 15) / 5); 48 start + levels 2..99 = 1273 points at 99.
STAT_POINTS_PER_LEVEL = (0,) + tuple(3 + (L - 1) // 5 for L in range(1, 100))
# STAT_COST: indexed by the stat's current value, points to raise it by one; 99 is a sentinel (never raised).
# RO pre-re: 2 + (value - 1) / 10 (rAthena PC_STATUS_POINT_COST).
STAT_COST = (0,) + tuple(2 + (v - 1) // 10 for v in range(1, 99)) + (1 << 20,)
STAT_POINTS_PER_LEVEL_J = jnp.asarray(STAT_POINTS_PER_LEVEL, jnp.int32)
STAT_COST_J = jnp.asarray(STAT_COST, jnp.int32)

# ---- length checks ----
assert len(BASE_EXP_TABLE) == 100 and BASE_EXP_TABLE[0] == 0 and BASE_EXP_TABLE[99] == 0
assert all(BASE_EXP_TABLE[i] < BASE_EXP_TABLE[i + 1] for i in range(1, 98))
assert len(JOB_EXP_NOVICE_TABLE) == 11 and JOB_EXP_NOVICE_TABLE[10] == 0
assert len(JOB_EXP_FIRST_TABLE) == 51 and JOB_EXP_FIRST_TABLE[50] == 0
assert len(JOB_EXP_TABLE) == NUM_JOBS and all(len(r) == 51 for r in JOB_EXP_TABLE)
assert JOB_EXP_TABLE_J.shape == (NUM_JOBS, 51)
assert len(MAX_JOB_LEVEL) == NUM_JOBS
assert len(BASE_HP_TABLE) == 100 and BASE_HP_TABLE[1] == 40 and BASE_HP_TABLE[99] == 3029
assert len(BASE_SP_TABLE) == 100 and BASE_SP_TABLE[1] == 12 and BASE_SP_TABLE[99] == 208
assert JOB_BASE_HP_TABLE[0][99] == 530 and JOB_BASE_HP_TABLE[1][99] == 3999 and JOB_BASE_SP_TABLE[2][99] == 604
assert len(_BASE_RAW) == 98 and len(JOB_EXP_FIRST_TABLE) == 51
assert len(JOB_HP_FACTOR) == len(JOB_SP_FACTOR) == len(JOB_ASPD_BASE) == NUM_JOBS
assert all(100 <= a <= 160 for a in JOB_ASPD_BASE)
assert len(JOB_WEAPON_ASPD) == NUM_JOBS and all(len(r) == 7 for r in JOB_WEAPON_ASPD)
assert len(MONSTER_BASE_EXP) == len(MONSTER_JOB_EXP) == NUM_MONSTER_KINDS
assert len(STAT_POINTS_PER_LEVEL) == 100 and len(STAT_COST) == 100
assert len(STAT_NAMES) == NUM_STATS and len(JOB_NAMES) == NUM_JOBS
