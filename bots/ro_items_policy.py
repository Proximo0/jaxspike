"""Bot item policy: the item inputs (use_item, equip, refine, drop; buy / sell when the engine has them).

Pure JAX over fixed shapes, read-only on the world. Merge with the movement policy as
`{**bot_inputs(world, key), **item_inputs(world, key)}`.

A living bot, each tick:
- equips the best weapon its job may wield (items.EQUIP_JOBS, else JOB_WEAPONS, RO pre-renewal classes; score
  ATK, plus MATK for Mages and Acolytes) when it beats what it wears; else the armor piece with the largest DEF gain over what is worn in
  its equip slot (a free slot counts as DEF -1, so any piece fills it); one equip per tick;
- uses its first healing consumable (never MEAT, survival.py eats that) below POTION_PCT of max_hp, or as soon as
  the hp it is missing is at least what that consumable heals (no heal wasted; ro6-balance); otherwise it
  uses a held card that fits a free socket of a worn item (one use_item per tick);
- refines its worn weapon when it holds that weapon level's ore and the ryn and the next step is still a
  guaranteed one (REFINE_CHANCE == 1000, i.e. below the safe limit);
- at a shop (game.ro_shop.at_shop), when the engine has "buy" / "sell" inputs: sells one etc. loot slot, then buys
  red potions while it holds fewer than POTION_STOCK and keeps RYN_RESERVE (item id to buy, -1 none; sell =
  inventory slot). bots.ro_nav_policy walks bots that want potions to their town's shop row.
It never drops.
"""
import jax.numpy as jnp
import numpy as np

from game.config import INV_SLOTS, ITEM_MEAT, MAX_PLAYERS, PLAYER_MAX_HP
from game.data import items as I
from game.data.skills import JOB_ACOLYTE, JOB_MAGE, N_JOBS
from game.ro_refine import REFINE_CHANCE

try:  # imported here, never inside a traced function: a module's import-time int() calls must not run under jit
    from game import ro_shop
except ImportError:  # pragma: no cover
    ro_shop = None

POTION_PCT = 45         # above bots.ro_nav_policy.RETREAT_PCT: a bot holding potions drinks instead of resting
POTION_STOCK = 10
RYN_RESERVE = 0         # ro6-balance: was 200, which a bot first held after ~3500 ticks (no potion bought in 1000)
BUY_ITEM = I.ITEM_RED_POTION

# Weapon classes each job may wield (RO pre-renewal; Novice ... Merchant), index = job id.
_WC = {
    0: (I.WC_DAGGER, I.WC_SWORD, I.WC_AXE, I.WC_MACE, I.WC_STAFF),
    1: (I.WC_DAGGER, I.WC_SWORD, I.WC_AXE, I.WC_MACE),
    2: (I.WC_DAGGER, I.WC_STAFF),
    3: (I.WC_DAGGER, I.WC_BOW),
    4: (I.WC_MACE, I.WC_STAFF),
    5: (I.WC_DAGGER, I.WC_SWORD, I.WC_BOW),
    6: (I.WC_DAGGER, I.WC_SWORD, I.WC_AXE, I.WC_MACE),
}
_jw = np.zeros((N_JOBS, 8), np.bool_)
for _j, _cls in _WC.items():
    _jw[_j, list(_cls)] = True
JOB_WEAPONS = jnp.asarray(_jw)  # [N_JOBS, n weapon classes]

_SLOTS = jnp.arange(INV_SLOTS, dtype=jnp.int32)
_NONE = jnp.full((MAX_PLAYERS,), -1, jnp.int32)
_PIDX = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)


def _first(mask):
    """Lowest True column of a [MAX_PLAYERS, INV_SLOTS] mask as (slot, any)."""
    return jnp.argmax(mask, axis=1).astype(jnp.int32), mask.any(axis=1)


def _jobs(world):
    if "stats" in world and "job" in world["stats"]:
        return world["stats"]["job"].astype(jnp.int32)
    return jnp.zeros((MAX_PLAYERS,), jnp.int32)


def _max_hp(world):
    if "stats" in world:
        from game import ro_stats
        return jnp.maximum(ro_stats.derived(world)["max_hp"].astype(jnp.int32), 1)
    return jnp.full((MAX_PLAYERS,), PLAYER_MAX_HP, jnp.int32)


def may_wear(item, job):
    """bool: the job may equip `item`: the item table's EQUIP_JOBS when it exists (ro-items2), else JOB_WEAPONS
    for weapons and anything for armor."""
    if hasattr(I, "EQUIP_JOBS"):
        return I.EQUIP_JOBS[item, job]
    is_w = I.KIND[item] == I.KIND_WEAPON
    return ~is_w | JOB_WEAPONS[job, jnp.clip(I.CLASS[item], 0, JOB_WEAPONS.shape[1] - 1)]


def weapon_score(item, job):
    """int32: how much a job wants weapon `item` (ATK, plus MATK for casters); -1 when it may not wield it."""
    caster = (job == JOB_MAGE) | (job == JOB_ACOLYTE)
    ok = (I.KIND[item] == I.KIND_WEAPON) & may_wear(item, job)
    return jnp.where(ok, I.ATK[item] + jnp.where(caster, I.MATK[item], 0), -1)


def wants_potions(world):
    """bool [P]: a living bot with fewer than POTION_STOCK red potions and the ryn for one plus RYN_RESERVE."""
    pl, it = world["players"], world["items"]
    held = (pl["inv"] == BUY_ITEM) & (it["inv_count"] > 0)
    n_pot = jnp.sum(jnp.where(held, it["inv_count"], 0), axis=1)
    rich = it["ryn"] >= I.BUY[BUY_ITEM] + RYN_RESERVE
    return pl["active"] & ~pl["dead"] & rich & (n_pot < POTION_STOCK)


def _at_shop(world):
    if ro_shop is None:  # pragma: no cover - before ro-items2's shop module
        return world["pz"][world["players"]["pos"][:, 0], world["players"]["pos"][:, 1]]
    return ro_shop.at_shop(world)


def engine_inputs():
    """Set of input keys the engine's empty_inputs() has (static)."""
    from game.state import empty_inputs
    return set(empty_inputs())


def item_inputs(world, key):
    """Return {"use_item", "equip", "refine", "drop"} (+ "buy", "sell" when the engine has them), int32 [P], -1 none."""
    del key  # the policy is deterministic
    pl, it = world["players"], world["items"]
    live = pl["active"] & ~pl["dead"]
    inv, count, equip, cards = pl["inv"], it["inv_count"], it["equip"], it["inv_cards"]
    held = (inv != 0) & (count > 0)
    kind = I.KIND[inv]
    job = _jobs(world)
    worn_slot = jnp.any(equip[:, :, None] == _SLOTS[None, None, :], axis=1)  # [P, S] slot is worn

    # --- equip: best allowed weapon, else the largest armor DEF gain ------------------------------------------
    w_score = jnp.where(held, weapon_score(inv, job[:, None]), -1)
    best = jnp.argmax(w_score, axis=1).astype(jnp.int32)
    best_score = w_score.max(axis=1)
    cur = equip[:, I.EQ_WEAPON]
    cur_id = jnp.where(cur >= 0, inv[_PIDX, jnp.clip(cur, 0, INV_SLOTS - 1)], 0)
    cur_score = jnp.where(cur >= 0, weapon_score(cur_id, job), -1)
    want_weapon = live & (best_score >= 0) & (best_score > cur_score)

    worn_id = jnp.where(equip >= 0, jnp.take_along_axis(inv, jnp.clip(equip, 0, INV_SLOTS - 1), axis=1), 0)
    worn_def = jnp.where(equip >= 0, I.DEF[worn_id] + I.BONUS_VALUE[worn_id], -1)  # [P, 8]
    cls = I.CLASS[inv]
    is_acc = cls == I.SLOT_ACCESSORY
    acc_def = jnp.minimum(worn_def[:, I.EQ_ACC1], worn_def[:, I.EQ_ACC2])[:, None]  # the weaker accessory
    eq_slot = jnp.clip(cls, 0, I.N_EQUIP_SLOTS - 1)
    slot_def = jnp.where(is_acc, acc_def, jnp.take_along_axis(worn_def, eq_slot, axis=1))
    # an accessory replaces ACC2 when ACC1 is taken: only worth it when ACC2 is the weaker (or free) one
    acc_ok = ~is_acc | (equip[:, I.EQ_ACC1] < 0)[:, None] | (worn_def[:, I.EQ_ACC2] <= worn_def[:, I.EQ_ACC1])[:, None]
    gain = (I.DEF[inv] + I.BONUS_VALUE[inv]) - slot_def
    armor = (held & (kind == I.KIND_ARMOR) & ~worn_slot & (cls >= I.SLOT_HEAD) & acc_ok & (gain > 0)
             & may_wear(inv, job[:, None]))
    a_gain = jnp.where(armor, gain, -1)
    a_slot = jnp.argmax(a_gain, axis=1).astype(jnp.int32)
    a_any = armor.any(axis=1)
    equip_in = jnp.where(want_weapon, best, jnp.where(live & a_any, a_slot, -1)).astype(jnp.int32)

    # --- use_item: heal when hurt, else compound a card that fits -------------------------------------------
    heal = held & (kind == I.KIND_CONSUMABLE) & (I.HEAL_HP[inv] > 0) & (inv != ITEM_MEAT)
    h_slot, h_any = _first(heal)
    mx = _max_hp(world)
    h_amt = I.HEAL_HP[inv[_PIDX, h_slot]]
    hurt = live & h_any & ((pl["hp"] * 100 < POTION_PCT * mx) | (mx - pl["hp"] >= h_amt))

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

    # --- refine: worn weapon below its safe limit with ore and ryn in hand ---------------------------------
    w_slot = jnp.clip(cur, 0, INV_SLOTS - 1)
    w_id = jnp.where(cur >= 0, inv[_PIDX, w_slot], 0)
    wl = I.WEAPON_LEVEL[w_id]
    r_now = jnp.clip(it["inv_refine"][_PIDX, w_slot], 0, I.MAX_REFINE)
    safe_step = REFINE_CHANCE[wl, r_now] >= 1000
    ore = I.REFINE_ORE[wl]
    has_ore = jnp.any(held & (inv == ore[:, None]), axis=1)
    can_pay = it["ryn"] >= I.REFINE_RYN[wl]
    do_refine = live & (cur >= 0) & (w_id != 0) & (r_now < I.MAX_REFINE) & safe_step & has_ore & can_pay
    refine_in = jnp.where(do_refine, cur, -1).astype(jnp.int32)

    out = {"use_item": use_in, "equip": equip_in, "refine": refine_in, "drop": _NONE}

    # --- shop: sell etc. loot, buy potions (only when the engine has the inputs) ----------------------------
    keys = engine_inputs()
    if "sell" in keys or "buy" in keys:
        shop = live & _at_shop(world)
    if "sell" in keys:
        etc = held & (kind == I.KIND_ETC) & (inv != I.ITEM_GOLD_COIN)
        s_slot, s_any = _first(etc)
        out["sell"] = jnp.where(shop & s_any, s_slot, -1).astype(jnp.int32)
    if "buy" in keys:
        out["buy"] = jnp.where(shop & wants_potions(world), BUY_ITEM, -1).astype(jnp.int32)
    return out
