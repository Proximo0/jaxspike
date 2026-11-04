"""Monster drop rolls and ground-drop placement (M3), fixed-shape JAX."""
import jax
import jax.numpy as jnp

from game.config import DROP_TTL, ITEM_NONE, MAP, MAX_DROPS, MAX_MONSTERS, MAX_PLAYERS
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
    unused) and zeny int32 [MAX_MONSTERS], uniform inclusive in the kind's range. Chances are per 10000.
    """
    k_drop, k_zeny = jax.random.split(key)
    kind = jnp.clip(world["monsters"]["kind"], 0, N_DROP_KINDS - 1)
    u = jax.random.uniform(k_drop, (MAX_MONSTERS, DROPS_PER_KIND))
    item = DROP_ITEM[kind]
    hit = (u * 10000.0 < DROP_CHANCE[kind]) & (item != 0)
    items = jnp.where(hit, item, 0).astype(jnp.int32)
    counts = jnp.where(hit, DROP_COUNT[kind], 0).astype(jnp.int32)
    lo, hi = ZENY_MIN[kind], ZENY_MAX[kind]
    zeny = jax.random.randint(k_zeny, (MAX_MONSTERS,), lo, hi + 1, jnp.int32)
    return items, counts, zeny


def drop_fields():
    """Fresh extra arrays for world["drops"]: count, refine, cards (the lead adds them in state.py)."""
    return {
        "count": jnp.zeros((MAX_DROPS,), jnp.int32),
        "refine": jnp.zeros((MAX_DROPS,), jnp.int32),
        "cards": jnp.zeros((MAX_DROPS, item_data.MAX_CARDS), jnp.int32),
    }


def place_drops(world, items, counts, zeny):
    """Create ground drops for monsters with died set, in free drop slots.

    One drop per rolled entry plus one GOLD_COIN drop per monster with zeny > 0. Candidates are ranked by
    cumsum and assigned to free slots like loot._spawn_drops; ranks beyond the free count are discarded. When
    the drops block has "count" the slot stores the rolled count (the coin: the full zeny, ZENY_PER_COIN = 1) and
    refine/cards 0; without the fields every drop is one unit.
    """
    mons, drops = world["monsters"], world["drops"]
    per = DROPS_PER_KIND + 1
    coin_count = jnp.maximum(1, zeny // item_data.ZENY_PER_COIN).astype(jnp.int32)
    coin_item = jnp.where(zeny > 0, item_data.ITEM_GOLD_COIN, 0).astype(jnp.int32)
    all_items = jnp.concatenate([items, coin_item[:, None]], axis=1)  # [M, per]
    all_counts = jnp.concatenate([counts, coin_count[:, None]], axis=1)  # [M, per]
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
    if "count" in drops:
        new["count"] = drops["count"].at[slot].set(jnp.maximum(all_counts.reshape(n), 1), mode="drop")
        new["refine"] = drops["refine"].at[slot].set(0, mode="drop")
        new["cards"] = drops["cards"].at[slot].set(0, mode="drop")
    return {**world, "drops": new}


def pickup(world):
    """Living active players on a drop tile take the lowest active drop into their first empty slot.

    Same rules as the old loot._pickup (lowest player wins a drop; a full inventory takes nothing). Worlds with
    an "items" block also get inv_count (the drop's count, 1 without the field), inv_refine and inv_cards written
    at the slot, and the taken drop's count/refine/cards cleared. Returns (world, picked) with picked bool
    [MAX_PLAYERS]: the player took a drop this tick. Fixed-shape, jit and vmap safe.
    """
    players, drops = world["players"], world["drops"]
    inv = players["inv"]
    didx = jnp.arange(MAX_DROPS, dtype=jnp.int32)
    pidx = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)

    dtile = drops["pos"][:, 0] * MAP + drops["pos"][:, 1]
    tile_min = jnp.full((MAP * MAP,), MAX_DROPS, jnp.int32).at[
        jnp.where(drops["active"], dtile, MAP * MAP)
    ].min(didx, mode="drop")
    ptile = jnp.clip(players["pos"][:, 0], 0, MAP - 1) * MAP + jnp.clip(players["pos"][:, 1], 0, MAP - 1)
    cand = tile_min[ptile]

    empty = inv == ITEM_NONE
    has_empty = empty.any(axis=1)
    first_empty = jnp.argmax(empty, axis=1)
    claim = players["active"] & ~players["dead"] & has_empty & (cand < MAX_DROPS)

    winner = jnp.full((MAX_DROPS,), MAX_PLAYERS, jnp.int32).at[
        jnp.where(claim, cand, MAX_DROPS)
    ].min(pidx, mode="drop")
    got = claim & (winner[jnp.where(claim, cand, 0)] == pidx)

    row = jnp.where(got, pidx, MAX_PLAYERS)
    src = jnp.where(got, cand, 0)
    new_inv = inv.at[row, first_empty].set(drops["item"][src], mode="drop")
    taken = winner < MAX_PLAYERS
    new_drops = {
        **drops,
        "active": drops["active"] & ~taken,
        "item": jnp.where(taken, ITEM_NONE, drops["item"]),
    }
    has_fields = "count" in drops
    if has_fields:
        new_drops["count"] = jnp.where(taken, 0, drops["count"])
        new_drops["refine"] = jnp.where(taken, 0, drops["refine"])
        new_drops["cards"] = jnp.where(taken[:, None], 0, drops["cards"])
    out = {**world, "players": {**players, "inv": new_inv}, "drops": new_drops}
    if "items" in world:
        it = world["items"]
        cnt = jnp.maximum(drops["count"][src], 1) if has_fields else jnp.ones((MAX_PLAYERS,), jnp.int32)
        ref = drops["refine"][src] if has_fields else jnp.zeros((MAX_PLAYERS,), jnp.int32)
        crd = drops["cards"][src] if has_fields else jnp.zeros((MAX_PLAYERS, item_data.MAX_CARDS), jnp.int32)
        out["items"] = {
            **it,
            "inv_count": it["inv_count"].at[row, first_empty].set(cnt, mode="drop"),
            "inv_refine": it["inv_refine"].at[row, first_empty].set(ref, mode="drop"),
            "inv_cards": it["inv_cards"].at[row, first_empty].set(crd, mode="drop"),
        }
    return out, got
