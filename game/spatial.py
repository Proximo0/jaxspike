"""Spatial grid for nearest-entity lookups. Owned by the lead.

Entities are bucketed into CELL x CELL tile cells. A query only sees entities in
its own cell and the 8 neighbouring cells, so the effective search radius is
about one cell (CELL tiles). Callers must keep their ranges <= CELL.
"""
import jax
import jax.numpy as jnp

from game.config import CELL, G, K_CELL, MAP, NCELLS

BIG = 10**6
_OFFSETS = jnp.array([[dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)], dtype=jnp.int32)


def build_grid(pos, active):
    """Bucket entities. Returns (table int32 [NCELLS, K_CELL] of indices or -1, overflow count).

    One single-operand sort of `cell * n + index` and a cummax rank (ro4-perf, the form of
    ro_monster_modes._grid_tables; was argsort + searchsorted). Within a cell entities keep index order, the first
    K_CELL are kept and the rest counted as overflow, as before.
    """
    n = pos.shape[0]
    assert (NCELLS + 1) * n < 2 ** 31, "packed cell * n + index keys no longer fit int32"  # static shapes
    cell_xy = pos // CELL
    cell = jnp.where(active, cell_xy[:, 0] * G + cell_xy[:, 1], NCELLS)
    ar =jnp.arange(n, dtype=jnp.int32)
    key = jnp.sort(cell * n + ar)
    order = key % n
    sorted_cell = key // n
    is_first = jnp.concatenate([jnp.array([True]), sorted_cell[1:] != sorted_cell[:-1]])
    rank = ar - jax.lax.cummax(jnp.where(is_first, ar, 0), axis=0)
    live = sorted_cell < NCELLS
    keep = live & (rank < K_CELL)
    overflow = jnp.sum(live & (rank >= K_CELL))
    row = jnp.where(keep, sorted_cell, NCELLS)  # dropped entries write out of bounds
    table = jnp.full((NCELLS, K_CELL), -1, dtype=jnp.int32)
    table = table.at[row, jnp.where(keep, rank, 0)].set(order, mode="drop")
    return table, overflow


def block_candidates(table, query_pos):
    """For each query position, the entity indices in its 3x3 cell block. Shape [Q, 9*K_CELL], -1 = empty."""
    qcell = query_pos // CELL
    ncx = jnp.clip(qcell[:, 0:1] + _OFFSETS[None, :, 0], 0, G - 1)
    ncy = jnp.clip(qcell[:, 1:2] + _OFFSETS[None, :, 1], 0, G - 1)
    ncell = ncx * G + ncy
    return table[ncell].reshape(query_pos.shape[0], 9 * K_CELL)


def nearest_in_block(table, query_pos, targets_pos):
    """Nearest manhattan target in each query's 3x3 block.

    Returns (idx int32 [Q], dist int32 [Q]). idx is a valid target index even when
    dist == BIG (no target in range); callers must check dist.
    """
    cand = block_candidates(table, query_pos)
    valid = cand >= 0
    safe = jnp.maximum(cand, 0)
    if query_pos.shape[0] * 9 > NCELLS:
        # Many queries (ro4-perf): gather the positions once per table entry into a packed x * MAP + y table, then
        # every query reads 9 contiguous rows of it instead of a random [Q, 9 * K_CELL, 2] slice of targets_pos.
        packed = targets_pos[:, 0] * MAP + targets_pos[:, 1]
        tpos = block_candidates(packed[jnp.maximum(table, 0)], query_pos)
        d = jnp.abs(tpos // MAP - query_pos[:, 0:1]) + jnp.abs(tpos % MAP - query_pos[:, 1:2])
    else:
        d = jnp.abs(query_pos[:, None, :] - targets_pos[safe]).sum(-1)
    d = jnp.where(valid, d, BIG)
    col = jnp.argmin(d, axis=1)
    dist = jnp.min(d, axis=1)
    idx = jnp.take_along_axis(safe, col[:, None], axis=1)[:, 0]
    return idx, dist
