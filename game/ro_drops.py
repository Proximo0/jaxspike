"""Monster drop rolls and ground-drop placement (M3), fixed-shape JAX."""
import jax
import jax.numpy as jnp

from game.config import DROP_TTL, MAX_DROPS, MAX_MONSTERS
from game.data import items as item_data
from game.data.drops import (
    DROP_CHANCE,
    DROP_COUNT,
    DROP_ITEM,
    DROPS_PER_KIND,
    N_DROP_KINDS,
    ZENY_MAX,
    ZENY_MIN,
)


def roll_drops(world, key):
    """Roll every drop entry for every monster slot (the caller masks by died).

    Returns (items, counts) int32 [MAX_MONSTERS, DROPS_PER_KIND] (0 where the roll failed or the entry is
    unused) and zeny int32 [MAX_MONSTERS], uniform inclusive in the kind's range.
    """
    k_drop, k_zeny = jax.random.split(key)
    kind = jnp.clip(world["monsters"]["kind"], 0, N_DROP_KINDS - 1)
    u = jax.random.uniform(k_drop, (MAX_MONSTERS, DROPS_PER_KIND))
    item = DROP_ITEM[kind]
    hit = (u * 1000.0 < DROP_CHANCE[kind]) & (item != 0)
    items = jnp.where(hit, item, 0).astype(jnp.int32)
    counts = jnp.where(hit, DROP_COUNT[kind], 0).astype(jnp.int32)
    lo, hi = ZENY_MIN[kind], ZENY_MAX[kind]
    zeny = jax.random.randint(k_zeny, (MAX_MONSTERS,), lo, hi + 1, jnp.int32)
    return items, counts, zeny


def place_drops(world, items, counts, zeny):
    """Create ground drops for monsters with died set, in free drop slots.

    One drop per rolled entry plus one GOLD_COIN drop per monster with zeny > 0 (count = max(1, zeny //
    ZENY_PER_COIN)). Candidates are ranked by cumsum and assigned to free slots like loot._spawn_drops; ranks
    beyond the free count are discarded. NOTE: counts above 1 are lost until the lead adds drops["count"]
    (each ground drop is one unit); `counts` is accepted and the coin count is computed, but not stored yet.
    """
    mons, drops = world["monsters"], world["drops"]
    per = DROPS_PER_KIND + 1
    coin_count = jnp.maximum(1, zeny // item_data.ZENY_PER_COIN).astype(jnp.int32)
    coin_item = jnp.where(zeny > 0, item_data.ITEM_GOLD_COIN, 0).astype(jnp.int32)
    all_items = jnp.concatenate([items, coin_item[:, None]], axis=1)  # [M, per]
    del coin_count, counts  # not stored until drops["count"] exists
    n = MAX_MONSTERS * per
    flat_item = all_items.reshape(n)
    flat_pos = jnp.repeat(mons["pos"], per, axis=0)
    spawn = jnp.repeat(mons["died"], per) & (flat_item != 0)

    c_rank = jnp.cumsum(spawn) - 1
    free = ~drops["active"]
    f_rank = jnp.cumsum(free) - 1
    slot_of_rank = jnp.full((MAX_DROPS,), MAX_DROPS, jnp.int32).at[
        jnp.where(free, f_rank, MAX_DROPS)
    ].set(jnp.arange(MAX_DROPS, dtype=jnp.int32), mode="drop")
    slot = slot_of_rank.at[jnp.where(spawn, c_rank, MAX_DROPS)].get(
        mode="fill", fill_value=MAX_DROPS
    )
    new = {
        **drops,
        "pos": drops["pos"].at[slot].set(flat_pos, mode="drop"),
        "item": drops["item"].at[slot].set(flat_item, mode="drop"),
        "active": drops["active"].at[slot].set(True, mode="drop"),
        "ttl": drops["ttl"].at[slot].set(DROP_TTL, mode="drop"),
    }
    return {**world, "drops": new}
