"""Player movement phase.

Occupancy and tile claims are resolved by sorting tile keys, not by scattering
into a MAP x MAP grid. Batched scatters into full-map arrays blew up GPU memory
under vmap (found while bisecting GPU memory use), and sorting is O(N log N) per world.
"""
import jax
import jax.numpy as jnp

from game.config import DIR_DELTAS, MAP, MAX_PLAYERS

_DELTAS = jnp.asarray(DIR_DELTAS, jnp.int32)
_NO_TILE = MAP * MAP  # sentinel tile key for inactive entities; sorts after every real tile
_NO_CLAIM = _NO_TILE * MAX_PLAYERS  # sentinel claim key; sorts after every real claim


def _occupied(target_flat, occupant_flat):
    """True where a target tile key matches an occupant key (sentinels never match a real target).

    One sort over occupants and targets together, then a running max of occupant keys.
    Occupants sort before targets with the same key, so a target is occupied exactly when
    the latest occupant key seen so far equals its own key.
    """
    n_occ = occupant_flat.shape[0]
    keys = jnp.concatenate([occupant_flat, target_flat])
    is_occ = jnp.concatenate([jnp.ones((n_occ,), jnp.bool_), jnp.zeros((target_flat.shape[0],), jnp.bool_)])
    order = jnp.argsort(keys * 2 + (~is_occ).astype(jnp.int32))
    k_sorted = keys[order]
    occ_sorted = is_occ[order]
    latest_occ = jax.lax.cummax(jnp.where(occ_sorted, k_sorted, -1), axis=0)
    hit_sorted = ~occ_sorted & (latest_occ == k_sorted)
    hit = hit_sorted[_inverse_permutation(order)]
    return hit[n_occ:]


def _inverse_permutation(order):
    """Inverse of a permutation, computed with a sort instead of a scatter (scatters with batched indices expand badly under vmap)."""
    return jnp.argsort(order)


def move_players(world, inputs):
    """Move active, living players one tile if free; lowest slot wins ties."""
    players = world["players"]
    monsters = world["monsters"]
    pos = players["pos"]
    target = pos + _DELTAS[jnp.clip(inputs["move"], 0, _DELTAS.shape[0] - 1)]

    in_bounds = jnp.all((target >= 0) & (target < MAP), axis=1)
    tc = jnp.clip(target, 0, MAP - 1)
    tflat = tc[:, 0] * MAP + tc[:, 1]

    # Occupancy from start-of-tick positions. Dead players are not physical: they are
    # waiting to reappear at their spawn tile and never block a tile (see combat._respawn_players).
    player_keys = jnp.where(players["active"] & ~players["dead"], pos[:, 0] * MAP + pos[:, 1], _NO_TILE)
    monster_keys = jnp.where(
        monsters["active"], monsters["pos"][:, 0] * MAP + monsters["pos"][:, 1], _NO_TILE
    )
    occupant_keys = jnp.concatenate([player_keys, monster_keys])
    occupied = _occupied(tflat, occupant_keys)

    moving = jnp.any(target != pos, axis=1)
    want = (
        players["active"]
        & ~players["dead"]
        & moving
        & in_bounds
        & (world["walls"][tc[:, 0], tc[:, 1]] == 0)
        & ~occupied
    )

    # Claim each contested tile for its lowest requesting slot: sort by (tile, slot) and keep the first of each tile.
    slots = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
    claim = jnp.where(want, tflat * MAX_PLAYERS + slots, _NO_CLAIM)
    order = jnp.argsort(claim)
    sorted_claim = claim[order]
    sorted_tile = sorted_claim // MAX_PLAYERS
    first_of_tile = jnp.concatenate([jnp.array([True]), sorted_tile[1:] != sorted_tile[:-1]])
    win_sorted = first_of_tile & (sorted_claim < _NO_CLAIM)
    win = win_sorted[_inverse_permutation(order)]

    new_pos = jnp.where(win[:, None], target, pos)
    return {**world, "players": {**players, "pos": new_pos}}
