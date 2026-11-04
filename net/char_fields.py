"""Per-character persistence of the RO blocks: extract from a host world, validate a loaded file, apply at login.

Flat dict of npz keys (block prefix + field). Extraction and validation are numpy only; `apply_character` scatters
into a JAX world. Docs: docs/persist_notes.md, docs/stats_wire.md "Persistence note", docs/items_persist.md.
"""
from functools import lru_cache

import numpy as np

from game.config import INV_SLOTS
from game.data import items as item_data
from game.data.craft import CRAFT_MAX_LEVEL, LEVEL_EXP, N_CRAFT_SKILLS
from game.data.skills import N_SKILLS, SKILL_MAX_LEVEL
from game.data.stats import BASE_EXP_TABLE, JOB_EXP_TABLE, MAX_BASE_LEVEL, MAX_JOB_LEVEL, MAX_STAT, NUM_JOBS

I32_MAX = 2**31 - 1
STAT_SCALARS = ("str", "agi", "vit", "int", "dex", "luk", "base_level", "job_level", "base_exp", "job_exp", "job",
                "stat_points", "skill_points", "sp", "max_sp", "was_dead")
_STAT6 = STAT_SCALARS[:6]
STATS_KEYS = tuple(f"stats_{k}" for k in STAT_SCALARS)
SKILLS_KEYS = ("skills_learned", "skills_sp")
ITEMS_KEYS = ("items_inv_count", "items_inv_refine", "items_inv_cards", "items_equip", "items_ryn")
CRAFT_KEYS = ("craft_level", "craft_exp", "craft_gathers", "craft_crafts")
PVP_KEYS = ("pvp_kills", "pvp_deaths")  # ro5-pvp2: world["pvp"] rows of the same names
RO_KEYS = STATS_KEYS + SKILLS_KEYS + ITEMS_KEYS + CRAFT_KEYS + PVP_KEYS
LEGACY_FIELDS = ("pos", "spawn", "hp", "inv", "kills", "deaths")

_LEVEL_EXP = np.asarray(LEVEL_EXP, np.int64)
_KIND = np.asarray(item_data.KIND)
_CLASS = np.asarray(item_data.CLASS)
_SLOTS = np.asarray(item_data.SLOTS)
_SKILL_MAX = np.asarray(SKILL_MAX_LEVEL)


@lru_cache(maxsize=1)
def _fresh_rows():
    """Slot 0 of each freshly built block, as numpy (what make_empty_world gives every slot)."""
    from game import ro_craft, ro_items, ro_skills, ro_stats
    st, sk, it, cr = ro_stats.make_state(), ro_skills.make_state(), ro_items.make_state(), ro_craft.make_state()
    return {
        "stats": {f"stats_{k}": int(st[k][0]) for k in STAT_SCALARS},
        "skills": {"skills_learned": np.asarray(sk["learned"])[0].astype(np.int32), "skills_sp": int(sk["sp"][0])},
        "items": {"items_inv_count": np.asarray(it["inv_count"])[0],
                  "items_inv_refine": np.asarray(it["inv_refine"])[0],
                  "items_inv_cards": np.asarray(it["inv_cards"])[0], "items_equip": np.asarray(it["equip"])[0],
                  "items_ryn": int(it["ryn"][0])},
        "craft": {"craft_level": np.asarray(cr["level"])[0], "craft_exp": np.asarray(cr["exp"])[0],
                  "craft_gathers": int(cr["gathers"][0]), "craft_crafts": int(cr["crafts"][0])},
        "pvp": {k: 0 for k in PVP_KEYS},
    }


def fresh_group(group: str, inv=None) -> dict:
    """Fresh-character values of one RO group ('stats', 'skills', 'items', 'craft', 'pvp'), as copies. For 'items' with
    `inv`, inv_count is 1 where inv != 0 (the migration rule)."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in _fresh_rows()[group].items()}
    if group == "items" and inv is not None:
        out["items_inv_count"] = (np.asarray(inv) != 0).astype(np.int32)
    return out


def fresh_ro_fields(inv=None) -> dict:
    out = {}
    for g in ("stats", "skills", "items", "craft", "pvp"):
        out.update(fresh_group(g, inv))
    return out


# --- extract -------------------------------------------------------------------------------------------------------
def character_fields(world_np, slot, town=None, storage=None) -> dict:
    """Every persisted field of player `slot`, read from host numpy blocks (e.g. `jax.device_get(world)`).

    Returns the flat npz-key dict (arrays are copies, scalars Python ints). `town` and `storage` are not world
    state; pass them to include them in the dict (the server keeps both per client)."""
    p, st, sk, it, cr = (world_np[b] for b in ("players", "stats", "skills", "items", "craft"))
    s = int(slot)
    out = {
        "pos": np.array(p["pos"][s], np.int32), "spawn": np.array(p["spawn"][s], np.int32),
        "hp": int(p["hp"][s]), "inv": np.array(p["inv"][s], np.int32),
        "kills": int(p["kills"][s]), "deaths": int(p["deaths"][s]),
    }
    for k in STAT_SCALARS:
        out[f"stats_{k}"] = int(st[k][s])
    out["skills_learned"] = np.array(sk["learned"][s], np.int32)
    out["skills_sp"] = int(sk["sp"][s])
    out["items_inv_count"] = np.array(it["inv_count"][s], np.int32)
    out["items_inv_refine"] = np.array(it["inv_refine"][s], np.int32)
    out["items_inv_cards"] = np.array(it["inv_cards"][s], np.int32)
    out["items_equip"] = np.array(it["equip"][s], np.int32)
    out["items_ryn"] = int(it["ryn"][s])
    out["craft_level"] = np.array(cr["level"][s], np.int32)
    out["craft_exp"] = np.array(cr["exp"][s], np.int32)
    out["craft_gathers"] = int(cr["gathers"][s])
    out["craft_crafts"] = int(cr["crafts"][s])
    if "pvp" in world_np:  # ro5-pvp2 (a block without it keeps the fresh 0 / 0 on load)
        for k in PVP_KEYS:
            out[k] = int(world_np["pvp"][k][s])
    if town is not None:
        out["town"] = int(town)
    if storage is not None:
        out["storage"] = np.array(storage, np.int32)
    return out


# --- apply ---------------------------------------------------------------------------------------------------------
def apply_character(world, slot, fields) -> dict:
    """Write a loaded character into `slot` of a (device) world with `.at[slot].set`. Returns a new world.

    Writes pos/spawn/hp/inv/kills/deaths and every RO field that is present in `fields`. A missing field keeps what
    the slot holds, which is the fresh value in a world from `make_empty_world` (and `characters.load` fills every
    RO key anyway). Not touched: `active`, `dead`, and the stats ledgers `kills_seen`/`deaths_seen`; the server
    sets those (docs/persist_notes.md)."""
    import jax.numpy as jnp
    s = int(slot)

    def put(block, key, fkey):
        nonlocal world
        if fkey in fields:
            old = world[block][key]
            val = jnp.asarray(np.asarray(fields[fkey]), old.dtype)
            world = {**world, block: {**world[block], key: old.at[s].set(val)}}

    for k in LEGACY_FIELDS:
        put("players", k, k)
    for k in STAT_SCALARS:
        put("stats", k, f"stats_{k}")
    put("skills", "learned", "skills_learned")
    put("skills", "sp", "skills_sp")
    for k in ("inv_count", "inv_refine", "inv_cards", "equip", "ryn"):
        put("items", k, f"items_{k}")
    for k in ("level", "exp", "gathers", "crafts"):
        put("craft", k, f"craft_{k}")
    if "pvp" in world:
        for k in PVP_KEYS:
            put("pvp", k, k)
    return world


# --- validate (used by characters.load) ------------------------------------------------------------------------------
def _scalar(v, lo, hi):
    """Clamp an integer-like 0-d value into [lo, hi]; None if it is not an integer scalar."""
    a = np.asarray(v)
    if a.shape != () or a.dtype.kind not in "iub":
        return None
    return int(min(max(int(a), lo), hi))


def _array(v, shape, lo, hi):
    """Int32 copy clamped into [lo, hi]; None if the shape or dtype is wrong."""
    a = np.asarray(v)
    if a.shape != shape or a.dtype.kind not in "iub":
        return None
    return np.clip(a.astype(np.int64), lo, hi).astype(np.int32)


def _stats(raw):
    if not all(k in raw for k in STATS_KEYS):
        return None
    g = {k: _scalar(raw[f"stats_{k}"], 0, I32_MAX) for k in STAT_SCALARS}
    if any(v is None for v in g.values()):
        return None
    for k in _STAT6:
        g[k] = min(max(g[k], 1), MAX_STAT)
    g["base_level"] = min(max(g["base_level"], 1), MAX_BASE_LEVEL)
    g["job"] = min(g["job"], NUM_JOBS - 1)
    g["job_level"] = min(max(g["job_level"], 1), MAX_JOB_LEVEL[g["job"]])
    g["was_dead"] = int(g["was_dead"] != 0)
    g["max_sp"] = max(g["max_sp"], 1)
    g["sp"] = min(g["sp"], g["max_sp"])
    if g["base_exp"] >= BASE_EXP_TABLE[g["base_level"]] or g["job_exp"] >= JOB_EXP_TABLE[g["job"]][g["job_level"]]:
        return None
    return {f"stats_{k}": v for k, v in g.items()}


def _skills(raw):
    if not all(k in raw for k in SKILLS_KEYS):
        return None
    arr = np.asarray(raw["skills_learned"])
    if arr.ndim == 1 and arr.dtype.kind in "iub" and arr.size < N_SKILLS:  # older save: new skills start at 0
        arr = np.concatenate([arr, np.zeros(N_SKILLS - arr.size, arr.dtype)])
    lv = _array(arr, (N_SKILLS,), 0, I32_MAX)
    sp = _scalar(raw["skills_sp"], 0, I32_MAX)
    if lv is None or sp is None:
        return None
    return {"skills_learned": np.minimum(lv, _SKILL_MAX).astype(np.int32), "skills_sp": sp}


def _items(raw, inv):
    if not all(k in raw for k in ITEMS_KEYS):
        return None
    cnt = _array(raw["items_inv_count"], (INV_SLOTS,), 0, item_data.MAX_STACK)
    ref = _array(raw["items_inv_refine"], (INV_SLOTS,), 0, item_data.MAX_REFINE)
    cards = _array(raw["items_inv_cards"], (INV_SLOTS, item_data.MAX_CARDS), 0, item_data.N_ITEMS - 1)
    eq = _array(raw["items_equip"], (item_data.N_EQUIP_SLOTS,), -1, INV_SLOTS - 1)
    ryn = _scalar(raw["items_ryn"], 0, item_data.MAX_RYN)
    if any(x is None for x in (cnt, ref, cards, eq, ryn)):
        return None
    empty = inv == 0
    cnt = np.where(empty, 0, np.maximum(cnt, 1)).astype(np.int32)
    ref = np.where(empty, 0, ref).astype(np.int32)
    socket_ok = np.arange(item_data.MAX_CARDS)[None, :] < _SLOTS[inv][:, None]
    cards = np.where((_KIND[cards] == item_data.KIND_CARD) & socket_ok & ~empty[:, None], cards, 0).astype(np.int32)
    seen = set()
    for e in range(eq.size):
        ptr = int(eq[e])
        if ptr < 0:
            continue
        item = int(inv[ptr])
        kind, cls = int(_KIND[item]), int(_CLASS[item])
        right = (kind == item_data.KIND_WEAPON and e == item_data.EQ_WEAPON) or (
            kind == item_data.KIND_ARMOR and ((cls == item_data.SLOT_ACCESSORY and e >= item_data.EQ_ACC1)
                                              or (1 <= cls <= 5 and e == cls)))
        if not right or ptr in seen:
            eq[e] = -1
        else:
            seen.add(ptr)
    return {"items_inv_count": cnt, "items_inv_refine": ref, "items_inv_cards": cards, "items_equip": eq,
            "items_ryn": ryn}


def _craft(raw):
    if not all(k in raw for k in CRAFT_KEYS):
        return None
    exp = _array(raw["craft_exp"], (N_CRAFT_SKILLS,), 0, I32_MAX)
    lvl = _array(raw["craft_level"], (N_CRAFT_SKILLS,), 1, CRAFT_MAX_LEVEL)
    gat = _scalar(raw["craft_gathers"], 0, I32_MAX)
    cra = _scalar(raw["craft_crafts"], 0, I32_MAX)
    if any(x is None for x in (exp, lvl, gat, cra)):
        return None
    lvl = np.maximum((exp[:, None] >= _LEVEL_EXP[None, :]).sum(-1), 1).astype(np.int32)  # level derives from exp
    return {"craft_level": lvl, "craft_exp": exp, "craft_gathers": gat, "craft_crafts": cra}


def _pvp(raw):
    """ro5-pvp2: the PvP record, clamped to [0, I32_MAX]; None (fresh 0 / 0) when absent or not integer scalars."""
    if not all(k in raw for k in PVP_KEYS):
        return None
    g = {k: _scalar(raw[k], 0, I32_MAX) for k in PVP_KEYS}
    return None if any(v is None for v in g.values()) else g


def validate_ro(raw: dict, inv) -> dict:
    """Full RO key set for a loaded file's `raw` dict and its already validated `inv`.

    Each group (stats, skills, items, craft, pvp) is optional: absent, partial or unusable groups fall back to the fresh
    group (items: count 1 where inv != 0); clampable values are clamped. Never raises, never returns None."""
    inv = np.asarray(inv, np.int32)
    stats = _stats(raw) or fresh_group("stats")
    skills = _skills(raw) or fresh_group("skills")
    return {**stats, **skills, **(_items(raw, inv) or fresh_group("items", inv)),
            **(_craft(raw) or fresh_group("craft")),
            **(_pvp(raw) or fresh_group("pvp"))}
