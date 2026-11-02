"""Per-viewer visibility and delta computation for protocol v2. Pure JAX, fixed shapes.

Entity ids are global: players 0..MAX_PLAYERS-1, monsters MAX_PLAYERS + i, drops MAX_PLAYERS + MAX_MONSTERS + i.
See docs/m0_brief.md Part A.
"""
import jax.numpy as jnp

from game import ro_stats

from game.config import (MAX_DROPS, MAX_PLAYERS, MONSTER_HP, N_ENTITIES, VIEW_H, VIEW_K,
                         VIEW_W)

DROP_KIND_BASE = 100
_BIG = 2**30


def entity_table(world):
    """Flat per-entity arrays over all E ids (players, monsters, drops)."""
    p, m, d = world["players"], world["monsters"], world["drops"]
    mhp = jnp.asarray(MONSTER_HP, jnp.int32)[m["kind"]]
    zeros_d = jnp.zeros((MAX_DROPS,), jnp.int32)
    return {
        "pos": jnp.concatenate([p["pos"], m["pos"], d["pos"]]).astype(jnp.int32),
        "vis": jnp.concatenate([p["active"] & ~p["dead"], m["active"], d["active"]]),
        "hp": jnp.concatenate([p["hp"], m["hp"], zeros_d]).astype(jnp.int32),
        "max_hp": jnp.concatenate([ro_stats.derived(world)["max_hp"].astype(jnp.int32), mhp, zeros_d]),
        "kind": jnp.concatenate([jnp.zeros((MAX_PLAYERS,), jnp.int32), 1 + m["kind"].astype(jnp.int32),
                                 DROP_KIND_BASE + d["item"].astype(jnp.int32)]),
    }


def changed_mask(prev, now):
    """True where pos, vis, hp or kind differ."""
    return ((prev["pos"] != now["pos"]).any(axis=-1) | (prev["vis"] != now["vis"])
            | (prev["hp"] != now["hp"]) | (prev["kind"] != now["kind"]))


def _compact(key):
    """key int32 [V, E] (>= _BIG = not a candidate) -> ids int32 [V, VIEW_K], ascending key, -1 padded."""
    order = jnp.argsort(key, axis=1)[:, :VIEW_K]
    ok = jnp.take_along_axis(key, order, axis=1) < _BIG
    return jnp.where(ok, order, -1).astype(jnp.int32)


def _scatter(ids):
    """Bool [V, E] marking the emitted ids (-1 ignored)."""
    rows = jnp.arange(ids.shape[0])[:, None]
    cols = jnp.where(ids >= 0, ids, N_ENTITIES)  # out of bounds, dropped
    return jnp.zeros((ids.shape[0], N_ENTITIES), jnp.bool_).at[rows, cols].set(True, mode="drop")


def view_update(now, changed, viewer_slot, known):
    """Returns (send_ids, gone_ids, known_next); see docs/m0_brief.md Part A."""
    valid = viewer_slot >= 0
    slot = jnp.where(valid, viewer_slot, 0)
    vpos = now["pos"][slot]                                              # [V, 2]
    d = jnp.abs(now["pos"][None, :, :] - vpos[:, None, :])               # [V, E, 2]
    inside = (d[..., 0] <= VIEW_W // 2) & (d[..., 1] <= VIEW_H // 2)
    ids = jnp.arange(N_ENTITIES, dtype=jnp.int32)[None, :]
    is_self = valid[:, None] & (ids == slot[:, None])
    visible = valid[:, None] & ((now["vis"][None, :] & inside) | is_self)
    known = known & valid[:, None]

    dist = d[..., 0] + d[..., 1]
    send = visible & (changed[None, :] | ~known)
    send_ids = _compact(jnp.where(send, dist * N_ENTITIES + ids, _BIG))
    gone_ids = _compact(jnp.where(known & ~visible, ids, _BIG))

    # Unsent overflow stays unknown (retried); unsent gone stays known (retried).
    known_next = (known & ~_scatter(gone_ids)) | _scatter(send_ids)
    return send_ids, gone_ids, known_next
