"""Bot item policy: the four item inputs (use_item, equip, refine, drop) for every player slot.

Pure JAX over fixed shapes, read-only on the world. Merge with the movement policy as
`{**bot_inputs(world, key), **item_inputs(world, key)}`.

A living bot, each tick:
- equips the highest-ATK weapon it holds when that beats what it wears (one equip per tick; when no
  weapon is better it wears any armor piece whose equip slot is still free);
- uses its first healing consumable (never MEAT, survival.py eats that) when hp < HUNT_RETREAT_HP;
  otherwise it uses a held card that fits a free socket of a worn item (one use_item per tick);
- refines its worn weapon when it holds that weapon level's ore and the zeny and the next step is
  still a guaranteed one (REFINE_CHANCE == 1000, i.e. below the safe limit).
It never drops.
"""
import jax.numpy as jnp

from game.config import HUNT_RETREAT_HP, INV_SLOTS, ITEM_MEAT, MAX_PLAYERS
from game.data import items as I
from game.ro_refine import REFINE_CHANCE

_SLOTS = jnp.arange(INV_SLOTS, dtype=jnp.int32)
_NONE = jnp.full((MAX_PLAYERS,), -1, jnp.int32)
_PIDX = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)


def _first(mask):
    """Lowest True column of a [MAX_PLAYERS, INV_SLOTS] mask as (slot, any)."""
    return jnp.argmax(mask, axis=1).astype(jnp.int32), mask.any(axis=1)


def item_inputs(world, key):
    """Return {"use_item", "equip", "refine", "drop"}, each int32 [MAX_PLAYERS], -1 = none."""
    del key  # the policy is deterministic
    pl, it = world["players"], world["items"]
    live = pl["active"] & ~pl["dead"]
    inv, count, equip, cards = pl["inv"], it["inv_count"], it["equip"], it["inv_cards"]
    held = (inv != 0) & (count > 0)
    kind = I.KIND[inv]
    worn_slot = jnp.any(equip[:, :, None] == _SLOTS[None, None, :], axis=1)  # [P, S] slot is worn

    # --- equip: best weapon, else an armor piece for a free equip slot -------------------------
    weapon = held & (kind == I.KIND_WEAPON)
    w_atk = jnp.where(weapon, I.ATK[inv], -1)
    best = jnp.argmax(w_atk, axis=1).astype(jnp.int32)
    best_atk = w_atk.max(axis=1)
    cur = equip[:, I.EQ_WEAPON]
    cur_atk = jnp.where(cur >= 0, I.ATK[inv[_PIDX, jnp.clip(cur, 0, INV_SLOTS - 1)]], -1)
    want_weapon = live & (best_atk > cur_atk)

    armor = held & (kind == I.KIND_ARMOR) & ~worn_slot
    cls = I.CLASS[inv]
    free_eq = equip < 0  # [P, 8]
    # equip slot an armor piece would go to is free: slot types 1..5 map to the same equip slot index
    # (ACC1 when ACC1 is free, else ACC2 is the only other accessory slot)
    fits = jnp.where(
        cls == I.SLOT_ACCESSORY,
        free_eq[:, I.EQ_ACC1][:, None] | free_eq[:, I.EQ_ACC2][:, None],
        jnp.take_along_axis(free_eq, jnp.clip(cls, 0, I.N_EQUIP_SLOTS - 1), axis=1),
    )
    a_slot, a_any = _first(armor & fits)
    equip_in = jnp.where(want_weapon, best, jnp.where(live & a_any, a_slot, -1)).astype(jnp.int32)

    # --- use_item: heal when hurt, else compound a card that fits -------------------------------
    heal = held & (kind == I.KIND_CONSUMABLE) & (I.HEAL_HP[inv] > 0) & (inv != ITEM_MEAT)
    h_slot, h_any = _first(heal)
    hurt = live & (pl["hp"] < HUNT_RETREAT_HP) & h_any

    worn_id = jnp.where(equip >= 0, jnp.take_along_axis(inv, jnp.clip(equip, 0, INV_SLOTS - 1), axis=1), 0)
    worn_cards = jnp.take_along_axis(cards, jnp.clip(equip, 0, INV_SLOTS - 1)[:, :, None], axis=1)  # [P, 8, 4]
    socket_ok = jnp.arange(cards.shape[2])[None, None, :] < I.SLOTS[worn_id][:, :, None]
    has_free = (worn_id != 0) & jnp.any((worn_cards == 0) & socket_ok, axis=2)  # [P, 8]
    card = held & (kind == I.KIND_CARD)
    c_target = jnp.clip(I.CLASS[inv], 0, I.N_EQUIP_SLOTS - 1)  # card target slot type == equip slot for 0..5
    target_free = jnp.where(
        I.CLASS[inv] == I.SLOT_ACCESSORY,
        has_free[:, I.EQ_ACC1][:, None] | has_free[:, I.EQ_ACC2][:, None],
        jnp.take_along_axis(has_free, c_target, axis=1),
    )
    c_slot, c_any = _first(card & target_free)
    use_in = jnp.where(hurt, h_slot, jnp.where(live & c_any, c_slot, -1)).astype(jnp.int32)

    # --- refine: worn weapon below its safe limit with ore and zeny in hand ---------------------
    w_slot = jnp.clip(cur, 0, INV_SLOTS - 1)
    w_id = jnp.where(cur >= 0, inv[_PIDX, w_slot], 0)
    wl = I.WEAPON_LEVEL[w_id]
    r_now = jnp.clip(it["inv_refine"][_PIDX, w_slot], 0, I.MAX_REFINE)
    safe_step = REFINE_CHANCE[wl, r_now] >= 1000
    ore = I.REFINE_ORE[wl]
    has_ore = jnp.any(held & (inv == ore[:, None]), axis=1)
    can_pay = it["zeny"] >= I.REFINE_ZENY[wl]
    do_refine = live & (cur >= 0) & (w_id != 0) & (r_now < I.MAX_REFINE) & safe_step & has_ore & can_pay
    refine_in = jnp.where(do_refine, cur, -1).astype(jnp.int32)

    return {"use_item": use_in, "equip": equip_in, "refine": refine_in, "drop": _NONE}
