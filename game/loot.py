"""Loot phase: RO drop tables for kills, zeny to the killer, player pickup, ground item decay."""
import jax.numpy as jnp

from game import ro_anim, ro_drops, ro_mvp
from game.config import ITEM_NONE, MAX_MONSTERS, MAX_PLAYERS
from game.data import items as item_data
from game.state import next_key

PICKUP_MS = 300  # pickup clip length (docs/anim_wire.md)


def killer_of(world):
    """int32 [MAX_MONSTERS]: the player slot credited with each monster (-1 = nobody).

    The credit is ro_mvp's cumulative-damage `top_damager` (combat reports auto-attacks and ro_skills_effects
    reports skill hits through ro_mvp.record_damage, so melee, ranged and skill kills all pay; the mvp phase
    clears it after loot and stats), kept only when that player is still logged in. Worlds without the mvp block
    credit nobody. ro_stats pays exp from the same credit.
    """
    if "mvp" not in world:
        return jnp.full((MAX_MONSTERS,), -1, jnp.int32)
    return ro_mvp.credit_of(world)


def _spawn_drops(world):
    """Roll the kind's drop table for every monster that died this tick and place the hits on its tile.

    Zeny goes straight to the credited killer's `items.zeny` (capped at MAX_ZENY); a kill nobody is credited with
    leaves its zeny on the ground as GOLD_COIN (ro_drops.place_drops), which pickup turns into zeny.
    """
    sub, world = next_key(world)
    items, counts, zeny = ro_drops.roll_drops(world, sub)
    died = world["monsters"]["died"]
    killer = killer_of(world)
    credited = died & (killer >= 0)
    if "items" in world:
        it = world["items"]
        gain = jnp.zeros((MAX_PLAYERS,), jnp.int32).at[jnp.where(credited, killer, MAX_PLAYERS)].add(
            jnp.where(credited, zeny, 0), mode="drop")
        new_zeny = jnp.minimum(it["zeny"] + jnp.minimum(gain, item_data.MAX_ZENY), item_data.MAX_ZENY)
        world = {**world, "items": {**it, "zeny": new_zeny}}
        zeny = jnp.where(credited, 0, zeny)
    return ro_drops.place_drops(world, items, counts, zeny)


def _pickup(world):
    """Living active players on a drop tile take it (count, refine, cards) into their first empty slot; the taker
    plays the pickup clip. Rules live in ro_drops.pickup (lowest player wins; a full inventory takes nothing)."""
    world, got = ro_drops.pickup(world)
    return ro_anim.set_action(
        world, jnp.concatenate([got, jnp.zeros((MAX_MONSTERS,), jnp.bool_)]), ro_anim.PICKUP, PICKUP_MS
    )


def _decay(world):
    """Count down drop lifetimes and deactivate expired drops."""
    drops = world["drops"]
    ttl = jnp.where(drops["active"], drops["ttl"] - 1, drops["ttl"])
    alive = drops["active"] & (ttl > 0)
    new = {
        **drops,
        "ttl": ttl,
        "active": alive,
        "item": jnp.where(alive, drops["item"], ITEM_NONE),
    }
    return {**world, "drops": new}


def loot(world):
    """Run the loot phase: spawn drops from kills, pick up, decay."""
    return _decay(_pickup(_spawn_drops(world)))
