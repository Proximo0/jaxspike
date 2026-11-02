"""M1 stats, jobs and exp tables. Plain int tuples plus matching jnp.int32 arrays (suffix _J) built at import."""
import math

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
SP_REGEN_TICKS = 16               # SP regen cadence in ticks
DEATH_PENALTY_PERMILLE = 10       # 1% of the level's exp requirement
DEATH_PENALTY_MIN_LEVEL = 10
TICK_HZ_X1000 = 1000 * 1000 // TICK_MS   # ticks per second x1000 (8333 at 120 ms)

STAT_NAMES = ("str", "agi", "vit", "int", "dex", "luk")
JOB_NAMES = ("Novice", "Swordsman", "Mage", "Archer", "Acolyte", "Thief", "Merchant")


def _loglog(x, anchors):
    """Piecewise power-law interpolation through (x, y) anchors (monotone for rising anchors)."""
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1:
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
    return float(anchors[-1][1])


def _monotone(values):
    """Round to ints and force strictly rising values."""
    out, prev = [], 0
    for v in values:
        prev = max(int(round(v)), prev + 1)
        out.append(prev)
    return out


def _base_exp(level):
    """Base exp to go from `level` to level+1: ~9 at 1, ~300 at 10, ~100k at 50, ~3M at 98."""
    if level < 1 or level >= MAX_BASE_LEVEL:
        return 0
    return _BASE_RAW[level - 1]


_BASE_RAW = _monotone([_loglog(L, ((1, 9), (10, 300), (50, 100_000), (98, 3_000_000))) for L in range(1, 99)])
# BASE_EXP_TABLE: indexed by base level L, exp to reach L+1 (plain exp); 0 at index 0 and 99 (cap).
BASE_EXP_TABLE = tuple(_base_exp(L) for L in range(100))
BASE_EXP_TABLE_J = jnp.asarray(BASE_EXP_TABLE, jnp.int32)

# JOB_EXP_NOVICE_TABLE: indexed by job level 0..10, exp to reach the next job level; 0 at 0 and 10 (cap).
_NOV_RAW = _monotone([_loglog(L, ((1, 10), (9, 450))) for L in range(1, 10)])
JOB_EXP_NOVICE_TABLE = (0,) + tuple(_NOV_RAW) + (0,)
# JOB_EXP_FIRST_TABLE: indexed by job level 0..50, exp to reach the next job level; 0 at 0 and 50 (cap).
_FIRST_RAW = _monotone([_loglog(L, ((1, 60), (10, 1_200), (49, 60_000))) for L in range(1, 50)])
JOB_EXP_FIRST_TABLE = (0,) + tuple(_FIRST_RAW) + (0,)
# JOB_EXP_TABLE: [NUM_JOBS][51] indexed by [job][job level]; row 0 novice padded with 0s, rows 1..6 first-job.
JOB_EXP_TABLE = (JOB_EXP_NOVICE_TABLE + (0,) * 40,) + (JOB_EXP_FIRST_TABLE,) * (NUM_JOBS - 1)
JOB_EXP_TABLE_J = jnp.asarray(JOB_EXP_TABLE, jnp.int32)

# MAX_JOB_LEVEL: indexed by job id, job level cap.
MAX_JOB_LEVEL = (MAX_JOB_LEVEL_NOVICE,) + (MAX_JOB_LEVEL_FIRST,) * (NUM_JOBS - 1)
MAX_JOB_LEVEL_J = jnp.asarray(MAX_JOB_LEVEL, jnp.int32)

# BASE_HP_TABLE: indexed by base level, job-neutral max HP (plain HP); 150 at L1, 3500 at L99.
BASE_HP_TABLE = (0,) + tuple(int(round(150 + 3350 * ((L - 1) / 98) ** 1.15)) for L in range(1, 100))
# BASE_SP_TABLE: indexed by base level, job-neutral max SP (plain SP); 10 at L1, 650 at L99.
BASE_SP_TABLE = (0,) + tuple(int(round(10 + 640 * ((L - 1) / 98) ** 1.1)) for L in range(1, 100))
BASE_HP_TABLE_J = jnp.asarray(BASE_HP_TABLE, jnp.int32)
BASE_SP_TABLE_J = jnp.asarray(BASE_SP_TABLE, jnp.int32)

# JOB_HP_FACTOR: indexed by job id, x1000 multiplier on base HP.
JOB_HP_FACTOR = (1000, 1500, 700, 1000, 1000, 1100, 1300)
# JOB_SP_FACTOR: indexed by job id, x1000 multiplier on base SP.
JOB_SP_FACTOR = (1000, 700, 2000, 1000, 1500, 800, 900)
# JOB_ASPD_BASE: indexed by job id, ASPD before stats (plain points, 140..160).
JOB_ASPD_BASE = (150, 150, 140, 158, 148, 160, 146)
JOB_HP_FACTOR_J = jnp.asarray(JOB_HP_FACTOR, jnp.int32)
JOB_SP_FACTOR_J = jnp.asarray(JOB_SP_FACTOR, jnp.int32)
JOB_ASPD_BASE_J = jnp.asarray(JOB_ASPD_BASE, jnp.int32)

# MONSTER_BASE_EXP: indexed by monster kind, base exp paid per kill (plain exp). rat, orc.
MONSTER_BASE_EXP = (10, 40)
# MONSTER_JOB_EXP: indexed by monster kind, job exp paid per kill (plain exp). rat, orc.
MONSTER_JOB_EXP = (5, 25)
MONSTER_BASE_EXP_J = jnp.asarray(MONSTER_BASE_EXP, jnp.int32)
MONSTER_JOB_EXP_J = jnp.asarray(MONSTER_JOB_EXP, jnp.int32)

# STAT_POINTS_PER_LEVEL: indexed by the base level just reached, status points gained (plain points).
STAT_POINTS_PER_LEVEL = (0,) + tuple(3 + (L - 1) // 5 for L in range(1, 100))
# STAT_COST: indexed by the stat's current value, points to raise it by one; 99 is a sentinel (never raised).
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
assert len(BASE_HP_TABLE) == 100 and BASE_HP_TABLE[1] == 150 and 3000 <= BASE_HP_TABLE[99] <= 4000
assert len(BASE_SP_TABLE) == 100 and BASE_SP_TABLE[1] == 10 and 500 <= BASE_SP_TABLE[99] <= 800
assert len(JOB_HP_FACTOR) == len(JOB_SP_FACTOR) == len(JOB_ASPD_BASE) == NUM_JOBS
assert all(140 <= a <= 160 for a in JOB_ASPD_BASE)
assert len(MONSTER_BASE_EXP) == len(MONSTER_JOB_EXP) == NUM_MONSTER_KINDS
assert len(STAT_POINTS_PER_LEVEL) == 100 and len(STAT_COST) == 100
assert len(STAT_NAMES) == NUM_STATS and len(JOB_NAMES) == NUM_JOBS
