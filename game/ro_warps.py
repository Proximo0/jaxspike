"""Warp phase: players standing on a warp source tile are moved to its destination tile.

Pure JAX, fixed shapes: a [MAX_PLAYERS, N_WARPS] match against the padded warp table, then the
sort-based occupancy and tile claims of game/ro_move.py (no MAP x MAP table). Owns players.pos only.

`claim_moves` is the shared "jump a player to a tile" rule (warps here, the waybell teleport in game/ro_waybell.py):
the tile must be free of living players and active monsters at the start of the phase, and the lowest slot wins a
tile two players want. Keys are plain tile indices (x * MAP + y < MAP * MAP), never `tile * MAX_PLAYERS + slot`:
that product reached 2^31 at MAP 2048 (docs/world_scale.md); the slot order comes from the stable sort instead.
"""
import jax.numpy as jnp

from game import map_gen
from game.config import MAP
from game.ro_move import _NO_TILE, _first_claims, _occupied


def claim_moves(world, target, valid):
    """bool [MAX_PLAYERS]: which living players may jump to target [MAX_PLAYERS, 2] this tick.

    valid marks the players that want to jump (already checked for their own rule); the target must be in bounds,
    not hold a living player or an active monster at the start of the phase, and not be wanted by a lower slot.
    """
    players, monsters = world["players"], world["monsters"]
    pos = players["pos"]
    live = players["active"] & ~players["dead"]
    valid = live & valid & jnp.all((target >= 0) & (target < MAP), axis=1)
    tc = jnp.clip(target, 0, MAP - 1)
    tflat = jnp.where(valid, tc[:, 0] * MAP + tc[:, 1], _NO_TILE)
    player_keys = jnp.where(live, pos[:, 0] * MAP + pos[:, 1], _NO_TILE)
    monster_keys = jnp.where(monsters["active"], monsters["pos"][:, 0] * MAP + monsters["pos"][:, 1], _NO_TILE)
    occupied = _occupied(tflat, jnp.concatenate([player_keys, monster_keys]))
    return _first_claims(jnp.where(valid & ~occupied, tflat, _NO_TILE))


def apply_warps(world, table=None):
    """Move active, living players on a warp src tile to its dst; blocked or losing players retry next tick.

    table: optional (src, dst) int32 [W, 2] pair, -1 rows are padding; defaults to map_gen.warp_table.
    """
    src, dst = map_gen.warp_table(world["terrain"]) if table is None else table
    players = world["players"]
    pos = players["pos"]

    real = src[:, 0] >= 0  # [W]
    match = jnp.all(pos[:, None, :] == src[None, :, :], axis=2) & real[None, :]  # [P, W]
    hit = jnp.any(match, axis=1)
    target = dst[jnp.argmax(match, axis=1)]  # [P, 2]

    won = claim_moves(world, target, hit)
    new_pos = jnp.where(won[:, None], target, pos)
    return {**world, "players": {**players, "pos": new_pos}}
