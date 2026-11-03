"""Loot phase: RO drop tables for kills, zeny to the killer, player pickup, ground item decay."""
import jax.numpy as jnp

from game import ro_drops
from game.config import ITEM_NONE, MAP, MAX_DROPS, MAX_MONSTERS, MAX_PLAYERS
from game.data import items as item_data
from game.state import next_key


def killer_of(world):
    """int32 [MAX_MONSTERS]: the player slot credited with each monster (-1 = nobody).

    The credit is ro_mvp's `top_damager` (combat reports every landed hit through ro_mvp.record_damage; the mvp
    phase clears it after loot), kept only when that player is still logged in. Worlds without the mvp block
    credit nobody.
    """
    if "mvp" not in world:
        return jnp.full((MAX_MONSTERS,), -1, jnp.int32)
    who = world["mvp"]["top_damager"]
    safe = jnp.clip(who, 0, MAX_PLAYERS - 1)
    return jnp.where((who >= 0) & world["players"]["active"][safe], who, -1).astype(jnp.int32)


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
    """Living active players on a drop tile take it into their first empty inventory slot."""
    players, drops = world["players"], world["drops"]
    inv = players["inv"]
    didx = jnp.arange(MAX_DROPS, dtype=jnp.int32)
    pidx = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)

    # Lowest active drop index on each tile.
    dtile = drops["pos"][:, 0] * MAP + drops["pos"][:, 1]
    tile_min = jnp.full((MAP * MAP,), MAX_DROPS, jnp.int32).at[
        jnp.where(drops["active"], dtile, MAP * MAP)
    ].min(didx, mode="drop")
    ptile = jnp.clip(players["pos"][:, 0], 0, MAP - 1) * MAP + jnp.clip(
        players["pos"][:, 1], 0, MAP - 1
    )
    cand = tile_min[ptile]

    empty = inv == ITEM_NONE
    has_empty = empty.any(axis=1)
    first_empty = jnp.argmax(empty, axis=1)
    claim = players["active"] & ~players["dead"] & has_empty & (cand < MAX_DROPS)

    # Lowest player index wins each drop.
    winner = jnp.full((MAX_DROPS,), MAX_PLAYERS, jnp.int32).at[
        jnp.where(claim, cand, MAX_DROPS)
    ].min(pidx, mode="drop")
    got = claim & (winner[jnp.where(claim, cand, 0)] == pidx)

    new_inv = inv.at[jnp.where(got, pidx, MAX_PLAYERS), first_empty].set(
        drops["item"][jnp.where(got, cand, 0)], mode="drop"
    )
    taken = winner < MAX_PLAYERS
    new_drops = {
        **drops,
        "active": drops["active"] & ~taken,
        "item": jnp.where(taken, ITEM_NONE, drops["item"]),
    }
    return {**world, "players": {**players, "inv": new_inv}, "drops": new_drops}


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
