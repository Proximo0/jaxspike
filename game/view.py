"""Per-viewer visibility and delta computation for protocol v2. Pure JAX, fixed shapes.

Entity ids are global: players 0..MAX_PLAYERS-1, monsters MAX_PLAYERS + i, drops MAX_PLAYERS + MAX_MONSTERS + i,
resource nodes MAX_PLAYERS + MAX_MONSTERS + MAX_DROPS + i (docs/craft_wire.md). E is the table's own length.
See docs/m0_brief.md Part A.
"""
import jax
import jax.numpy as jnp

from game import ro_anim, ro_stats

from game.config import (ITEM_ID_LIMIT, MAX_DROPS, MAX_PLAYERS, MONSTER_HP, N_ENTITIES, VIEW_H, VIEW_K,
                         VIEW_W)
from game.data.craft import MAX_NODES

DROP_KIND_BASE = 100
NODE_KIND_BASE = DROP_KIND_BASE + ITEM_ID_LIMIT  # nodes 400..407: drops use 100 + item id (< ITEM_ID_LIMIT)
N_VIEW_ENTITIES = N_ENTITIES + MAX_NODES  # the table length E (config.N_ENTITIES has no nodes yet)
_BIG = 2**30


def entity_table(world):
    """Flat per-entity arrays over all E ids (players, monsters, drops, nodes). Nodes: vis = placed (a depleted node
    stays visible), hp = active 0/1, max_hp 1, kind NODE_KIND_BASE + node kind."""
    p, m, d, n = world["players"], world["monsters"], world["drops"], world["craft"]["nodes"]
    dying = ro_anim.dying(world)  # corpse window: dead clip shown and died within DEAD_VISIBLE_MS
    mhp = jnp.asarray(MONSTER_HP, jnp.int32)[m["kind"]]
    zeros_d = jnp.zeros((MAX_DROPS,), jnp.int32)
    return {
        "pos": jnp.concatenate([p["pos"], m["pos"], d["pos"], n["pos"]]).astype(jnp.int32),
        "vis": jnp.concatenate([p["active"] & (~p["dead"] | dying[:MAX_PLAYERS]),
                                m["active"] | (dying[MAX_PLAYERS:] & (m["respawn_in"] > 0)), d["active"],
                                n["placed"]]),
        "hp": jnp.concatenate([p["hp"], m["hp"], zeros_d, n["active"]]).astype(jnp.int32),
        "max_hp": jnp.concatenate([ro_stats.derived(world)["max_hp"].astype(jnp.int32), mhp, zeros_d,
                                   jnp.ones((MAX_NODES,), jnp.int32)]),
        "kind": jnp.concatenate([jnp.zeros((MAX_PLAYERS,), jnp.int32), 1 + m["kind"].astype(jnp.int32),
                                 DROP_KIND_BASE + d["item"].astype(jnp.int32),
                                 NODE_KIND_BASE + n["kind"].astype(jnp.int32)]),
        **_anim_columns(world),
    }


# ro-anim (docs/anim_wire.md): small dtypes because the host pulls the whole table every tick; drops are all 0.
_ANIM_DTYPES = {"action": jnp.int8, "face": jnp.int8, "start_tick": jnp.int32, "dur_ms": jnp.int16,
                "speed_ms": jnp.int16, "hit": jnp.int32}


def _anim_columns(world):
    w = ro_anim.wire_fields(world)
    return {k: jnp.concatenate([w[k].astype(dt), jnp.zeros((MAX_DROPS + MAX_NODES,), dt)]) for k, dt in _ANIM_DTYPES.items()}


def changed_mask(prev, now):
    """True where pos, vis, hp or kind differ, or (ro-anim) action, face, start_tick or speed_ms changed or a hit
    landed this tick (hit != 0)."""
    base = ((prev["pos"] != now["pos"]).any(axis=-1) | (prev["vis"] != now["vis"])
            | (prev["hp"] != now["hp"]) | (prev["kind"] != now["kind"]))
    if "action" not in now:
        return base
    return (base | (prev["action"] != now["action"]) | (prev["face"] != now["face"])
            | (prev["start_tick"] != now["start_tick"]) | (prev["speed_ms"] != now["speed_ms"]) | (now["hit"] != 0))


def _compact(key):
    """key int32 [V, E] (>= _BIG = not a candidate; candidate keys unique) -> (ids int32 [V, VIEW_K] ascending key,
    -1 padded; thresh int32 [V], the VIEW_K-th smallest key, or _BIG - 1 when the row has fewer candidates)."""
    neg, idx = jax.lax.top_k(-key, VIEW_K)  # -key cannot overflow: keys <= _BIG or dist * n + ids < _BIG < 2**31
    vals = -neg
    ids = jnp.where(vals < _BIG, idx, -1).astype(jnp.int32)
    return ids, jnp.minimum(vals[:, -1], _BIG - 1)  # thresh: a candidate is emitted iff key <= thresh


def view_update(now, changed, viewer_slot, known):
    """Returns (send_ids, gone_ids, known_next); see docs/m0_brief.md Part A."""
    valid = viewer_slot >= 0
    slot = jnp.where(valid, viewer_slot, 0)
    vpos = now["pos"][slot]                                              # [V, 2]
    d = jnp.abs(now["pos"][None, :, :] - vpos[:, None, :])               # [V, E, 2]
    inside = (d[..., 0] <= VIEW_W // 2) & (d[..., 1] <= VIEW_H // 2)
    n = now["pos"].shape[0]
    ids = jnp.arange(n, dtype=jnp.int32)[None, :]
    is_self = valid[:, None] & (ids == slot[:, None])
    visible = valid[:, None] & ((now["vis"][None, :] & inside) | is_self)
    known = known & valid[:, None]

    dist = d[..., 0] + d[..., 1]
    send = visible & (changed[None, :] | ~known)
    gone = known & ~visible
    key_send = jnp.where(send, dist * n + ids, _BIG)
    key_gone = jnp.where(gone, ids, _BIG)
    send_ids, thresh_send = _compact(key_send)
    gone_ids, thresh_gone = _compact(key_gone)

    # Unsent overflow stays unknown (retried); unsent gone stays known (retried).
    emitted_send = send & (key_send <= thresh_send[:, None])
    emitted_gone = gone & (key_gone <= thresh_gone[:, None])
    known_next = (known & ~emitted_gone) | emitted_send
    return send_ids, gone_ids, known_next
