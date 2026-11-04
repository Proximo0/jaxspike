"""Playtest cheats (server flag --cheats): one jitted writer over the world's own tables.

`apply(world, slot, op_code, args)` is a single jitted function (op_code and slot traced, so one compile for every
op and slot): a `lax.switch` over the ops below, each a fixed-shape `.at[slot].set` write through the engine's
tables (stats.STAT_POINTS_PER_LEVEL, MAX_JOB_LEVEL, items.EQUIP_JOBS, skills.SKILL_JOB / SKILL_MAX_LEVEL).
Derived values (max HP/SP, ATK, ...) are never written: ro_stats.derived recomputes them from the stats each tick.
`heal_mask(world, mask)` refills HP/SP of every masked living player (god mode, once per tick, host-side flag).

args is int32 [N_ARGS]; per op:
  level     [base, job]               absolute levels, 0 = keep; clamped to 1..99 and 1..job cap; stat points
                                      follow the engine's per-level table, skill points one per job level gained
  ryn       [amount]                  added, clamped to 0..MAX_RYN
  item      [item, count, refine]     first free inventory slots (a stackable takes one slot, gear one slot each)
  heal      []                        hp = max_hp, sp = max_sp (living players)
  teleport  [x, y]                    the host picks a free temple tile of the town
  kit       []                        best job weapon + armour set worn, 20 red / 10 blue potions, 100 flit feathers
  allskills []                        every Novice and current-job skill at max, stat points of the level granted
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from game import ro_stats
from game.config import INV_SLOTS, ITEM_NONE
from game.data import items as item_data
from game.data import skills as skill_data
from game.data import stats as stats_data

OPS = ("level", "ryn", "item", "heal", "teleport", "kit", "allskills")
OP_CODE = {name: i for i, name in enumerate(OPS)}
N_ARGS = 4
I32 = jnp.int32

# ---- the kit, built once from the item table (plain Python at import, a constant array in the jit) ---------------
# Weapon class each job fights with (the usual starter class of stats.JOB_ASPD_BASE's comment); the best item of that
# class the job may wear is the one with the highest ATK + MATK.
_JOB_WEAPON_CLASS = (item_data.WC_DAGGER, item_data.WC_SWORD, item_data.WC_STAFF, item_data.WC_BOW,
                     item_data.WC_MACE, item_data.WC_DAGGER, item_data.WC_AXE)
_ARMOR_SLOTS = ((item_data.SLOT_HEAD, item_data.EQ_HEAD), (item_data.SLOT_BODY, item_data.EQ_BODY),
                (item_data.SLOT_SHIELD, item_data.EQ_SHIELD), (item_data.SLOT_GARMENT, item_data.EQ_GARMENT),
                (item_data.SLOT_SHOES, item_data.EQ_SHOES))
FLY_WING = getattr(item_data, "ITEM_FLIT_FEATHER", 0)  # RO's Fly Wing (random teleport); 0 = no such item
KIT_SUPPLIES = ((item_data.ITEM_RED_POTION, 20), (item_data.ITEM_BLUE_POTION, 10), (FLY_WING, 100))
N_KIT = 1 + len(_ARMOR_SLOTS) + len(KIT_SUPPLIES)


def _best(job: int, pred) -> int:
    k = np.asarray(item_data.KIND)
    jobs = np.asarray(item_data.EQUIP_JOBS)
    score = np.asarray(item_data.ATK) + np.asarray(item_data.MATK) + np.asarray(item_data.DEF)
    ids = [i for i in range(1, item_data.N_ITEMS) if k[i] != item_data.KIND_NONE and jobs[i, job] and pred(i)]
    return max(ids, key=lambda i: (int(score[i]), int(np.asarray(item_data.BUY)[i]))) if ids else 0


def _kit_rows():
    """int32 [NUM_JOBS, N_KIT, 3]: (item id, count, equip slot or -1) per job; id 0 = nothing in that row."""
    k, c = np.asarray(item_data.KIND), np.asarray(item_data.CLASS)
    rows = []
    for job in range(stats_data.NUM_JOBS):
        wc = _JOB_WEAPON_CLASS[job]
        r = [(_best(job, lambda i: k[i] == item_data.KIND_WEAPON and c[i] == wc), 1, item_data.EQ_WEAPON)]
        for slot_type, eq in _ARMOR_SLOTS:
            two_handed = wc == item_data.WC_BOW and slot_type == item_data.SLOT_SHIELD  # a bow takes both hands
            best = 0 if two_handed else _best(job, lambda i: k[i] == item_data.KIND_ARMOR and c[i] == slot_type)
            r.append((best, 1, eq))
        r += [(i, n if i else 0, -1) for i, n in KIT_SUPPLIES]
        rows.append(r)
    return np.asarray(rows, np.int32)


KIT = _kit_rows()
KIT_J = jnp.asarray(KIT)


def kit_items(job: int) -> list[tuple[int, int, int]]:
    """The kit of a job as (item id, count, equip slot) tuples (for logs and tests)."""
    return [tuple(int(v) for v in r) for r in KIT[job] if r[0] > 0]


# ---- helpers ----------------------------------------------------------------------------------------------------
_SIDX = jnp.arange(INV_SLOTS, dtype=I32)
_CUM_STAT = jnp.cumsum(stats_data.STAT_POINTS_PER_LEVEL_J)  # points gained up to base level L
_CUM_COST = jnp.cumsum(stats_data.STAT_COST_J)  # points spent to raise a stat from 0 to value v (cum[v-1])


def _put(world, blk, key, val, slot):
    old = world[blk][key]
    return {**world, blk: {**world[blk], key: old.at[slot].set(jnp.asarray(val, old.dtype))}}


def _place(world, slot, ids, counts, refines):
    """Put up to K items (ids [K], 0 = skip) into the first free inventory slots of `slot`, in order.

    Returns (world, where [K]): the inventory index each item went to, -1 when skipped or no room."""
    inv = world["players"]["inv"][slot]
    it = world["items"]
    free = inv == ITEM_NONE
    want = ids > 0
    rank_item = jnp.cumsum(want.astype(I32)) - 1  # k-th wanted item
    rank_free = jnp.cumsum(free.astype(I32)) - 1  # s-th free slot
    hit = free[None, :] & want[:, None] & (rank_free[None, :] == rank_item[:, None])  # [K, S]
    has = jnp.any(hit, axis=1)
    where = jnp.where(has, jnp.argmax(hit, axis=1), -1).astype(I32)
    any_hit = jnp.any(hit, axis=0)  # [S]
    k_of = jnp.argmax(hit, axis=0)  # item index per slot
    new_inv = jnp.where(any_hit, ids[k_of], inv)
    new_cnt = jnp.where(any_hit, counts[k_of], it["inv_count"][slot])
    new_ref = jnp.where(any_hit, refines[k_of], it["inv_refine"][slot])
    new_cards = jnp.where(any_hit[:, None], 0, it["inv_cards"][slot])
    world = _put(world, "players", "inv", new_inv, slot)
    world = _put(world, "items", "inv_count", new_cnt, slot)
    world = _put(world, "items", "inv_refine", new_ref, slot)
    world = _put(world, "items", "inv_cards", new_cards, slot)
    return world, where


def _stat_points_of_level(world, slot):
    """Status points a character of this base level and these stats has left when nothing is wasted."""
    s = world["stats"]
    lvl = jnp.clip(s["base_level"][slot], 1, stats_data.MAX_BASE_LEVEL)
    total = stats_data.STAT_POINTS_START + _CUM_STAT[lvl] - _CUM_STAT[1]
    vals = jnp.stack([s[k][slot] for k in stats_data.STAT_NAMES])
    spent = jnp.sum(_CUM_COST[jnp.clip(vals - 1, 0, stats_data.MAX_STAT)] - _CUM_COST[0])
    return jnp.maximum(total - spent, 0)


# ---- ops (each: (world, slot, args) -> world) -------------------------------------------------------------------
def _op_level(world, slot, a):
    s = world["stats"]
    job = jnp.clip(s["job"][slot], 0, stats_data.NUM_JOBS - 1)
    b0, j0 = s["base_level"][slot], s["job_level"][slot]
    b1 = jnp.where(a[0] > 0, jnp.clip(a[0], 1, stats_data.MAX_BASE_LEVEL), b0)
    j1 = jnp.where(a[1] > 0, jnp.clip(a[1], 1, stats_data.MAX_JOB_LEVEL_J[job]), j0)
    cb = lambda L: _CUM_STAT[jnp.clip(L, 0, stats_data.MAX_BASE_LEVEL)]
    world = _put(world, "stats", "stat_points", jnp.maximum(s["stat_points"][slot] + cb(b1) - cb(b0), 0), slot)
    world = _put(world, "stats", "skill_points",
                 s["skill_points"][slot] + jnp.maximum(j1 - j0, 0) * stats_data.SKILL_POINTS_PER_JOB_LEVEL, slot)
    world = _put(world, "stats", "base_level", b1, slot)
    world = _put(world, "stats", "job_level", j1, slot)
    world = _put(world, "stats", "base_exp", jnp.where(b1 != b0, 0, s["base_exp"][slot]), slot)
    return _put(world, "stats", "job_exp", jnp.where(j1 != j0, 0, s["job_exp"][slot]), slot)


def _op_ryn(world, slot, a):
    r = world["items"]["ryn"][slot]
    amt = jnp.clip(a[0], -item_data.MAX_RYN, item_data.MAX_RYN)
    new = jnp.where(amt > 0, r + jnp.minimum(amt, item_data.MAX_RYN - r), jnp.maximum(r + amt, 0))  # no overflow
    return _put(world, "items", "ryn", new, slot)


def _op_item(world, slot, a):
    item = jnp.clip(a[0], 0, item_data.N_ITEMS - 1)
    ok = (a[0] > 0) & (a[0] < item_data.N_ITEMS) & (item_data.KIND[item] != item_data.KIND_NONE)
    stack = item_data.STACKABLE[item]
    gear = (item_data.KIND[item] == item_data.KIND_WEAPON) | (item_data.KIND[item] == item_data.KIND_ARMOR)
    count = jnp.clip(a[1], 1, item_data.MAX_STACK)
    n_slots = jnp.where(stack, 1, jnp.minimum(count, INV_SLOTS))  # gear and other non-stackables: one per slot
    ids = jnp.where(ok & (_SIDX < n_slots), item, 0)
    counts = jnp.full((INV_SLOTS,), jnp.where(stack, count, 1), I32)
    refines = jnp.full((INV_SLOTS,), jnp.where(gear, jnp.clip(a[2], 0, item_data.MAX_REFINE), 0), I32)
    world, _ = _place(world, slot, ids, counts, refines)
    return world


def _full_hp(world, mask):
    d = ro_stats.derived(world)
    p, s = world["players"], world["stats"]
    live = mask & p["active"] & ~p["dead"]
    hp = jnp.where(live, d["max_hp"], p["hp"]).astype(p["hp"].dtype)
    sp = jnp.where(live, d["max_sp"], s["sp"]).astype(s["sp"].dtype)
    return {**world, "players": {**p, "hp": hp}, "stats": {**s, "sp": sp}}


def _op_heal(world, slot, a):
    return _full_hp(world, jnp.arange(world["players"]["hp"].shape[0]) == slot)


def _op_teleport(world, slot, a):
    pos = world["players"]["pos"]
    return _put(world, "players", "pos", jnp.stack([a[0], a[1]]).astype(pos.dtype), slot)


def _op_kit(world, slot, a):
    job = jnp.clip(world["stats"]["job"][slot], 0, stats_data.NUM_JOBS - 1)
    rows = KIT_J[job]  # [N_KIT, 3]
    zeros = jnp.zeros((N_KIT,), I32)
    world, where = _place(world, slot, rows[:, 0], jnp.maximum(rows[:, 1], 1), zeros)
    eq = rows[:, 2]
    n_eq = world["items"]["equip"].shape[1]
    worn = (eq >= 0) & (where >= 0)
    equip = world["items"]["equip"][slot].at[jnp.where(worn, eq, n_eq)].set(where, mode="drop")
    return _put(world, "items", "equip", equip, slot)


def _op_allskills(world, slot, a):
    job = world["stats"]["job"][slot]
    sj = skill_data.SKILL_JOB
    mine = (sj == skill_data.JOB_NOVICE) | (sj == job)
    mine = mine & (jnp.arange(sj.shape[0]) > 0)
    learned = world["skills"]["learned"][slot]
    world = _put(world, "skills", "learned", jnp.where(mine, jnp.maximum(learned, skill_data.SKILL_MAX_LEVEL), learned),
                 slot)
    sp = jnp.maximum(world["stats"]["stat_points"][slot], _stat_points_of_level(world, slot))
    return _put(world, "stats", "stat_points", sp, slot)


_BRANCHES = (_op_level, _op_ryn, _op_item, _op_heal, _op_teleport, _op_kit, _op_allskills)
assert len(_BRANCHES) == len(OPS)


def _apply(world, slot, op_code, args):
    args = jnp.asarray(args, I32)
    branches = [lambda w, f=f: f(w, slot, args) for f in _BRANCHES]
    return lax.switch(jnp.clip(op_code, 0, len(OPS) - 1), branches, world)


apply = jax.jit(_apply)
heal_mask = jax.jit(_full_hp)


def args_of(msg: dict, teleport_tile=None) -> tuple[int, np.ndarray]:
    """(op_code, int32 [N_ARGS]) for a validated cheat message (net.protocol); godmode is host-side, not here."""
    op = msg["op"]
    a = np.zeros(N_ARGS, np.int32)
    if op == "level":
        a[:2] = msg["base"], msg["job"]
    elif op == "ryn":
        a[0] = msg["amount"]
    elif op == "item":
        a[:3] = msg["item"], msg["count"], msg["refine"]
    elif op == "teleport":
        a[:2] = teleport_tile
    return OP_CODE[op], a
