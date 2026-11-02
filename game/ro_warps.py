"""Warp phase: players standing on a warp source tile are moved to its destination tile.

Pure JAX, fixed shapes: a [MAX_PLAYERS, N_WARPS] match against the padded warp table, then the
sort-based occupancy of game/movement.py (no MAP x MAP table). Owns players.pos only.
"""
import jax
import jax.numpy as jnp

from game import map_gen
from game.config import MAP, MAX_PLAYERS
from game.movement import _inverse_permutation, _occupied

_NO_CLAIM = jnp.iinfo(jnp.int32).max


def apply_warps(world, table=None):
    """Move active, living players on a warp src tile to its dst; blocked or losing players retry next tick.

    table: optional (src, dst) int32 [W, 2] pair, -1 rows are padding; defaults to map_gen.warp_table.
    """
    src, dst = map_gen.warp_table(world["terrain"]) if table is None else table
    players, monsters = world["players"], world["monsters"]
    pos = players["pos"]

    real = src[:, 0] >= 0  # [W]
    match = (
        jnp.all(pos[:, None, :] == src[None, :, :], axis=2) & real[None, :]
    )  # [P, W]
    hit = jnp.any(match, axis=1)
    target = dst[jnp.argmax(match, axis=1)]  # [P, 2]
    valid = hit & jnp.all((target >= 0) & (target < MAP), axis=1)
    tflat = target[:, 0] * MAP + target[:, 1]

    live = players["active"] & ~players["dead"]
    player_keys = jnp.where(live, pos[:, 0] * MAP + pos[:, 1], -1)
    monster_keys = jnp.where(monsters["active"], monsters["pos"][:, 0] * MAP + monsters["pos"][:, 1], -1)
    occupied = _occupied(jnp.where(valid, tflat, -2), jnp.concatenate([player_keys, monster_keys]))

    want = live & valid & ~occupied
    slots = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
    claim = jnp.where(want, tflat * MAX_PLAYERS + slots, _NO_CLAIM)
    order = jnp.argsort(claim)
    sorted_claim = claim[order]
    sorted_tile = sorted_claim // MAX_PLAYERS
    first = jnp.concatenate([jnp.array([True]), sorted_tile[1:] != sorted_tile[:-1]])
    win = (first & (sorted_claim < _NO_CLAIM))[_inverse_permutation(order)]

    new_pos = jnp.where(win[:, None], target, pos)
    return {**world, "players": {**players, "pos": new_pos}}
