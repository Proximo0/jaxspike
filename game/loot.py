"""Loot phase: drops from kills, player pickup, ground item decay."""
import jax
import jax.numpy as jnp

from game.config import (
    DROP_TTL,
    ITEM_GOLD_COIN,
    ITEM_MEAT,
    ITEM_NONE,
    ITEM_SWORD,
    LOOT_COIN_MAX,
    LOOT_MEAT_MAX,
    LOOT_SWORD_MAX,
    MAP,
    MAX_DROPS,
    MAX_MONSTERS,
    MAX_PLAYERS,
)
from game.state import next_key


def roll_item(u):
    """Map uniform rolls in [0, 1) to item ids using the LOOT_*_MAX thresholds in config."""
    return jnp.where(
        u < LOOT_COIN_MAX, ITEM_GOLD_COIN,
        jnp.where(u < LOOT_MEAT_MAX, ITEM_MEAT, jnp.where(u < LOOT_SWORD_MAX, ITEM_SWORD, ITEM_NONE)),
    ).astype(jnp.int32)


def _spawn_drops(world):
    """Create drops for monsters that died this tick, in free drop slots."""
    sub, world = next_key(world)
    mons, drops = world["monsters"], world["drops"]
    item = roll_item(jax.random.uniform(sub, (MAX_MONSTERS,)))
    spawn = mons["died"] & (item != ITEM_NONE)

    # Rank of each spawning monster among spawners, and of each free slot among free slots.
    m_rank = jnp.cumsum(spawn) - 1
    free = ~drops["active"]
    f_rank = jnp.cumsum(free) - 1
    slot_of_rank = jnp.full((MAX_DROPS,), MAX_DROPS, jnp.int32).at[
        jnp.where(free, f_rank, MAX_DROPS)
    ].set(jnp.arange(MAX_DROPS, dtype=jnp.int32), mode="drop")
    # Rank beyond the free-slot count maps to MAX_DROPS, i.e. the drop is discarded.
    slot = slot_of_rank.at[jnp.where(spawn, m_rank, MAX_DROPS)].get(
        mode="fill", fill_value=MAX_DROPS
    )

    new = {
        "pos": drops["pos"].at[slot].set(mons["pos"], mode="drop"),
        "item": drops["item"].at[slot].set(item, mode="drop"),
        "active": drops["active"].at[slot].set(True, mode="drop"),
        "ttl": drops["ttl"].at[slot].set(DROP_TTL, mode="drop"),
    }
    return {**world, "drops": new}


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
