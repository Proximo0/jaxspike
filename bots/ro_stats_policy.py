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
_SLOT = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)


def bot_allot(world, key):
    """Return int32 [MAX_PLAYERS]: -1 or the stat index to raise this tick.

    Raise the job's primary stat unless it is already 2x the secondary + 10, then the secondary;
    -1 when the chosen stat is capped or its cost exceeds the unspent points. `key` is unused.
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
    return jnp.where(world["players"]["active"], out, -1).astype(jnp.int32)


def bot_job(world, key):
    """Return int32 [MAX_PLAYERS]: -1 or a first job (1 + slot % 6) once a Novice reaches job level 10. `key` is unused."""
    del key
    s = world["stats"]
    ready = world["players"]["active"] & (s["job"] == JOB_NOVICE) & (s["job_level"] >= MAX_JOB_LEVEL_NOVICE)
    return jnp.where(ready, 1 + _SLOT % 6, -1).astype(jnp.int32)
