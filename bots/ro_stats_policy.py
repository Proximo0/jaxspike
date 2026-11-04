"""Bot stat-allot and job-change policy for the M1 stats block. Pure JAX, read-only on the world.

The lead merges these into bots.policies.bot_inputs as inputs["allot"] and inputs["job"].
"""
import jax.numpy as jnp

from game.config import MAX_PLAYERS
from game.data.stats import (JOB_NOVICE, MAX_JOB_LEVEL_NOVICE, MAX_STAT, NUM_JOBS,
                             STAT_COST_J)

# Per job: (primary stat, secondary stat), stat ids STR=0 AGI=1 VIT=2 INT=3 DEX=4 LUK=5.
JOB_BUILD = ((0, 2), (0, 2), (3, 4), (4, 1), (3, 2), (1, 0), (0, 2))
assert len(JOB_BUILD) == NUM_JOBS
_PRIMARY = jnp.asarray([b[0] for b in JOB_BUILD], jnp.int32)  # [NUM_JOBS]
_SECONDARY = jnp.asarray([b[1] for b in JOB_BUILD], jnp.int32)  # [NUM_JOBS]
# DEX in the plan (ro5-balance): HIT = base level + DEX, so a bot whose build lacks DEX hits band-1 prey (Fiddlehop
# level 9 + AGI 9) at under 70% for ever. Such a job keeps DEX at DEX_PER_LEVEL_PCT percent of its base level
# (100: DEX = base level, so HIT = 2 x level), raised before the primary/secondary rule. A fresh Novice keeps its
# STR 18 / VIT 4 (DEX 1 = level 1): a share of the primary instead (50% of STR) cost 22% of level-1 kills on the CPU
# soak (3.80 -> 2.98, seed 0) because Blobbin is already hit at 80%. Jobs with DEX in the build (Mage, Archer) weigh 0.
STAT_DEX = 4
DEX_PER_LEVEL_PCT = 100
_DEX_PCT = jnp.asarray([0 if STAT_DEX in b else DEX_PER_LEVEL_PCT for b in JOB_BUILD], jnp.int32)  # [NUM_JOBS]
_SLOT = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)


def bot_allot(world, key):
    """Return int32 [MAX_PLAYERS]: -1 or the stat index to raise this tick.

    DEX first while it is below DEX_PER_LEVEL_PCT % of the base level (jobs without DEX in their build), else the
    job's primary stat unless it is already 2x the secondary + 10, then the secondary; -1 when the chosen stat is
    capped or its cost exceeds the unspent points. `key` is unused.
    """
    del key
    s = world["stats"]
    stacked = jnp.stack([s["str"], s["agi"], s["vit"], s["int"], s["dex"], s["luk"]])  # [6, N]
    job = s["job"]
    pri, sec = _PRIMARY[job], _SECONDARY[job]
    v_pri = jnp.take_along_axis(stacked, pri[None], axis=0)[0]
    v_sec = jnp.take_along_axis(stacked, sec[None], axis=0)[0]
    pts = s["stat_points"]

    def can(v):
        return (v < MAX_STAT) & (pts >= STAT_COST_J[jnp.clip(v, 0, MAX_STAT)])

    want_pri = (v_pri < 2 * v_sec + 10) & can(v_pri)
    out = jnp.where(want_pri, pri, jnp.where(can(v_sec), sec, -1))
    v_dex = s["dex"]
    want_dex = (v_dex * 100 < _DEX_PCT[job] * s["base_level"]) & can(v_dex)
    out = jnp.where(want_dex, STAT_DEX, out)
    return jnp.where(world["players"]["active"], out, -1).astype(jnp.int32)


def bot_job(world, key):
    """Return int32 [MAX_PLAYERS]: -1 or a first job (1 + slot % 6) once a Novice reaches job level 10. `key` is unused."""
    del key
    s = world["stats"]
    ready = world["players"]["active"] & (s["job"] == JOB_NOVICE) & (s["job_level"] >= MAX_JOB_LEVEL_NOVICE)
    return jnp.where(ready, 1 + _SLOT % 6, -1).astype(jnp.int32)
