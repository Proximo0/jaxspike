"""Bot policy with skills: the base hunting policy plus learn / skill / target inputs. Pure JAX, read-only."""
import jax.numpy as jnp

from bots import policies
from game.config import MAX_PLAYERS, PLAYER_MAX_HP
from game.data.skills import (JOB_ACOLYTE, N_JOBS, SK_BASH, SK_DOUBLE_STRAFE, SK_ENVENOM, SK_FIRE_BOLT,
                              SK_FIRST_AID, SK_HEAL, SK_MAMMONITE, SKILL_SP)
from game.spatial import build_grid, nearest_in_block

LEARN_TARGET_LEVEL = 5  # bots learn their job's first castable skill until this level

# Indexed by job id: Novice, Swordsman, Mage, Archer, Acolyte, Thief, Merchant.
JOB_FIRST_SKILL = jnp.array(
    [SK_FIRST_AID, SK_BASH, SK_FIRE_BOLT, SK_DOUBLE_STRAFE, SK_HEAL, SK_ENVENOM, SK_MAMMONITE], jnp.int32)

_SLOT = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)


def _jobs(world):
    # Static pytree-structure check, jit-safe.
    if "stats" in world and "job" in world["stats"]:
        return world["stats"]["job"].astype(jnp.int32)
    return _SLOT % N_JOBS


def bot_inputs(world, key):
    """Return the base bot inputs plus int32 learn / skill / target, each [MAX_PLAYERS]."""
    out = policies.bot_inputs(world, key)
    pl, mon, sk = world["players"], world["monsters"], world["skills"]
    live = pl["active"] & ~pl["dead"]
    job = _jobs(world)
    first = JOB_FIRST_SKILL[job]
    lvl = sk["learned"][_SLOT, first]

    learn = jnp.where(pl["active"] & (lvl < LEARN_TARGET_LEVEL), first, 0).astype(jnp.int32)

    table, _ = build_grid(mon["pos"], mon["active"])
    idx, _ = nearest_in_block(table, pl["pos"], mon["pos"])
    adjacent = out["attack"]  # chase & dist == 1, exactly the base policy's melee condition
    # Level used for the cost lookups: at least 1 so an unlearned skill indexes a valid column.
    l1 = jnp.maximum(lvl, 1)
    ready = (live & (lvl > 0) & (sk["casting"] == 0) & (sk["cooldown"][_SLOT, first] == 0)
             & (sk["sp"] >= SKILL_SP[first, l1]))
    fight = adjacent & ready

    heal_ready = (live & (job == JOB_ACOLYTE) & (lvl > 0) & (sk["casting"] == 0)
                  & (sk["cooldown"][_SLOT, SK_HEAL] == 0) & (sk["sp"] >= SKILL_SP[SK_HEAL, l1])
                  & (pl["hp"] * 2 < PLAYER_MAX_HP))
    skill = jnp.where(heal_ready, SK_HEAL, jnp.where(fight, first, 0)).astype(jnp.int32)
    target = jnp.where(heal_ready, _SLOT, jnp.where(fight, MAX_PLAYERS + idx, -1)).astype(jnp.int32)
    return {**out, "learn": learn, "skill": skill, "target": target}
