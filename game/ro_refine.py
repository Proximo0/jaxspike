"""Refining and card compounding (M3 items). Pure JAX, fixed shapes, no loops over players or slots.

World["items"] holds inv_count, inv_refine, inv_cards, equip, zeny; the item id of a slot is players["inv"].
"""
import jax
import jax.numpy as jnp

from game.config import INV_SLOTS, MAX_PLAYERS
from game.data import items as item_data

N_LEVELS = 5          # weapon level 0 (armor) .. 4
N_REFINE_COLS = 11    # refine 0..10
_SAFE = (4, 7, 6, 5, 4)  # armor, lvl1..lvl4: highest refine reached with certainty
# Per-mille chance by target level (+5..+10 etc.), pre-renewal.
_ABOVE = (
    (600, 400, 400, 200, 200, 90),   # armor +5..+10
    (600, 400, 190),                 # lvl1 +8..+10
    (600, 400, 200, 190),            # lvl2 +7..+10
    (600, 500, 200, 200, 190),       # lvl3 +6..+10
    (600, 400, 400, 200, 200, 90),   # lvl4 +5..+10
)


def _build_chance():
    rows = []
    for lvl in range(N_LEVELS):
        row = [1000] * _SAFE[lvl] + list(_ABOVE[lvl])
        assert len(row) == N_REFINE_COLS - 1, (lvl, len(row))
        rows.append(row + [0])  # column 10 = 0
    return rows


REFINE_CHANCE = jnp.array(_build_chance(), jnp.int32)

_PIDX = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)


def refine(world, inputs, key):
    """Refine the slot named by inputs["refine"] (-1 none). Success +1, failure destroys the item."""
    players, it = world["players"], world["items"]
    inv, count = players["inv"], it["inv_count"]
    slot = inputs["refine"].astype(jnp.int32)
    in_range = (slot >= 0) & (slot < INV_SLOTS)
    sc = jnp.clip(slot, 0, INV_SLOTS - 1)
    iid = inv[_PIDX, sc]
    r = it["inv_refine"][_PIDX, sc]
    kind = item_data.KIND[iid]
    wl = item_data.WEAPON_LEVEL[iid]
    ore = item_data.REFINE_ORE[wl]
    cost = item_data.REFINE_ZENY[wl]

    has_ore = (inv == ore[:, None]) & (count > 0) & (ore[:, None] != 0)
    ore_slot = jnp.argmax(has_ore, axis=1).astype(jnp.int32)  # lowest slot holding the ore
    alive = players["active"] & ~players["dead"]
    ok = (
        in_range & alive
        & ((kind == item_data.KIND_WEAPON) | (kind == item_data.KIND_ARMOR))
        & (count[_PIDX, sc] > 0)
        & (r < item_data.MAX_REFINE)
        & has_ore.any(axis=1)
        & (it["zeny"] >= cost)
    )

    chance = REFINE_CHANCE[wl, jnp.clip(r, 0, N_REFINE_COLS - 1)]
    u = jax.random.uniform(key, (MAX_PLAYERS,))
    win = ok & (u * 1000.0 < chance)
    lose = ok & ~win

    # Pay zeny and one ore unit (cleared when it reaches 0).
    zeny = it["zeny"] - jnp.where(ok, cost, 0)
    drop_i = INV_SLOTS  # out-of-range index: scatter is dropped
    o_idx = jnp.where(ok, ore_slot, drop_i)
    ore_left = count[_PIDX, ore_slot] - 1
    count = count.at[_PIDX, o_idx].set(ore_left, mode="drop")
    ore_gone = jnp.where(ok & (ore_left <= 0), ore_slot, drop_i)
    inv = inv.at[_PIDX, ore_gone].set(0, mode="drop")
    inv_refine = it["inv_refine"].at[_PIDX, ore_gone].set(0, mode="drop")
    inv_cards = it["inv_cards"].at[_PIDX, ore_gone].set(0, mode="drop")
    equip = jnp.where(it["equip"] == ore_gone[:, None], -1, it["equip"])

    # Success: +1 refine.
    w_idx = jnp.where(win, sc, drop_i)
    inv_refine = inv_refine.at[_PIDX, w_idx].add(1, mode="drop")

    # Failure: destroy the item and clear equip pointers to it.
    l_idx = jnp.where(lose, sc, drop_i)
    inv = inv.at[_PIDX, l_idx].set(0, mode="drop")
    count = count.at[_PIDX, l_idx].set(0, mode="drop")
    inv_refine = inv_refine.at[_PIDX, l_idx].set(0, mode="drop")
    inv_cards = inv_cards.at[_PIDX, l_idx].set(0, mode="drop")
    equip = jnp.where(lose[:, None] & (equip == sc[:, None]), -1, equip)

    new_it = {**it, "inv_count": count, "inv_refine": inv_refine, "inv_cards": inv_cards,
              "equip": equip, "zeny": zeny}
    return {**world, "players": {**players, "inv": inv}, "items": new_it}


def compound_card(world, slot, ok):
    """Socket the card in inventory `slot` into the matching worn item. Returns (world, done).

    The caller decrements the card count.
    """
    players, it = world["players"], world["items"]
    slot = slot.astype(jnp.int32)
    in_range = (slot >= 0) & (slot < INV_SLOTS)
    sc = jnp.clip(slot, 0, INV_SLOTS - 1)
    card = players["inv"][_PIDX, sc]
    is_card = (
        in_range & ok
        & (item_data.KIND[card] == item_data.KIND_CARD)
        & (it["inv_count"][_PIDX, sc] > 0)
    )

    equip = it["equip"]                                   # [P, 8]
    worn = equip >= 0
    ec = jnp.clip(equip, 0, INV_SLOTS - 1)
    wid = jnp.take_along_axis(players["inv"], ec, axis=1)  # [P, 8] item ids
    wid = jnp.where(worn, wid, 0)
    wcards = jnp.take_along_axis(it["inv_cards"], ec[:, :, None], axis=1)  # [P, 8, MAX_CARDS]
    k = jnp.arange(item_data.MAX_CARDS, dtype=jnp.int32)
    free = (
        worn[:, :, None] & (wid[:, :, None] != 0)
        & (k[None, None, :] < item_data.SLOTS[wid][:, :, None])
        & (wcards == 0)
    )                                                      # [P, 8, MAX_CARDS]
    slot_free = free.any(-1)                               # [P, 8]
    first_k = jnp.argmax(free, axis=-1).astype(jnp.int32)  # lowest free socket

    cls = item_data.CLASS[card]
    is_acc = cls == item_data.SLOT_ACCESSORY
    eq_slot = jnp.where(
        is_acc,
        jnp.where(slot_free[:, item_data.EQ_ACC1], item_data.EQ_ACC1, item_data.EQ_ACC2),
        jnp.clip(cls, 0, item_data.EQ_SHOES),  # SLOT_WEAPON..SLOT_SHOES == EQ_WEAPON..EQ_SHOES
    ).astype(jnp.int32)
    done = is_card & slot_free[_PIDX, eq_slot]

    target = jnp.where(done, equip[_PIDX, eq_slot], INV_SLOTS)
    kk = first_k[_PIDX, eq_slot]
    inv_cards = it["inv_cards"].at[_PIDX, target, kk].set(card, mode="drop")
    return {**world, "items": {**it, "inv_cards": inv_cards}}, done
