"""Tests for game/spatial.py: grid build, overflow accounting, 3x3 nearest lookup."""
import jax
import jax.numpy as jnp
import numpy as np

from game.config import CELL, G, K_CELL, MAP, MAX_PLAYERS, NCELLS
from game.spatial import BIG, build_grid, nearest_in_block

TOWN_LO = 2 * CELL           # town square starts at tile 16
TOWN_SIZE = 2 * CELL         # 16 x 16 tiles
N_TOWN = 300


def cell_ids(pos_np):
    return (pos_np[:, 0] // CELL) * G + pos_np[:, 1] // CELL


def per_cell_overflow(pos, active):
    """numpy: dict cell_id -> number of active entities dropped from that cell."""
    pos, active = np.asarray(pos), np.asarray(active)
    counts = np.bincount(cell_ids(pos)[active], minlength=NCELLS)
    return {int(c): int(n - K_CELL) for c, n in enumerate(counts) if n > K_CELL}


def brute_nearest(table_ok_mask, pos, qpos):
    """numpy nearest manhattan dist per query within the 3x3 cell block, among masked entities."""
    pc = pos // CELL
    out = np.full(len(qpos), BIG, dtype=np.int64)
    for i, q in enumerate(qpos):
        qc = q // CELL
        near = table_ok_mask & (np.abs(pc - qc).max(axis=1) <= 1)
        if near.any():
            out[i] = np.abs(pos[near] - q).sum(axis=1).min()
    return out


def uniform_scene(seed=0, n_cells=300):
    """One entity in each of n_cells distinct random cells, plus a few extras per cell."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(NCELLS, size=n_cells, replace=False)
    base = np.stack([cells // G, cells % G], axis=1) * CELL
    pos = base + rng.integers(0, CELL, size=(n_cells, 2))
    return pos.astype(np.int32), rng


def town_scene(seed=1):
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, MAP, size=(MAX_PLAYERS, 2)).astype(np.int32)
    pos[:N_TOWN] = rng.integers(TOWN_LO, TOWN_LO + TOWN_SIZE, size=(N_TOWN, 2))
    active = np.zeros(MAX_PLAYERS, dtype=bool)
    active[:N_TOWN] = True
    return pos, active


def test_uniform_scatter_matches_bruteforce():
    pos, rng = uniform_scene()
    active = np.ones(len(pos), dtype=bool)
    table, overflow = build_grid(jnp.asarray(pos), jnp.asarray(active))
    assert int(overflow) == 0
    q = rng.integers(0, MAP, size=(200, 2)).astype(np.int32)
    idx, dist = nearest_in_block(table, jnp.asarray(q), jnp.asarray(pos))
    idx, dist = np.asarray(idx), np.asarray(dist)
    expected = brute_nearest(active, pos, q)
    assert (dist == expected).all()
    found = dist < BIG
    assert found.any() and (~found).any()  # scene exercises both outcomes
    chosen = np.abs(pos[idx[found]] - q[found]).sum(axis=1)
    assert (chosen == dist[found]).all()


def test_clustered_town_overflow_and_table_integrity():
    pos, active = town_scene()
    table, overflow = build_grid(jnp.asarray(pos), jnp.asarray(active))
    table = np.asarray(table)
    counts = np.bincount(cell_ids(pos)[active], minlength=NCELLS)
    expected = int(np.maximum(counts - K_CELL, 0).sum())
    report = per_cell_overflow(pos, active)
    assert int(overflow) == expected == sum(report.values())
    print(f"\n[town overflow] {N_TOWN} players in {TOWN_SIZE}x{TOWN_SIZE} tiles: "
          f"cells overflowing={len(report)}, max dropped in one cell={max(report.values())}, "
          f"total dropped={sum(report.values())}")
    assert expected > 0

    entries = table[table >= 0]
    assert len(entries) == len(set(entries.tolist()))  # no duplicates
    assert active[entries].all()
    cells = cell_ids(pos)
    for c in range(NCELLS):
        row = table[c][table[c] >= 0]
        assert len(row) == min(counts[c], K_CELL)
        assert (cells[row] == c).all()


def test_town_queries_still_see_something():
    pos, active = town_scene()
    table, _ = build_grid(jnp.asarray(pos), jnp.asarray(active))
    q = pos[:N_TOWN]  # every town player queries from its own spot
    idx, dist = nearest_in_block(table, jnp.asarray(q), jnp.asarray(pos))
    idx, dist = np.asarray(idx), np.asarray(dist)
    assert (dist < BIG).all()
    assert ((idx >= 0) & (idx < MAX_PLAYERS)).all()
    assert active[idx].all()
    assert (np.abs(pos[idx] - q).sum(axis=1) == dist).all()


def test_inactive_excluded_and_not_counted():
    pos, active = town_scene()
    extra = ~active
    pos[extra] = pos[0]  # park all inactive slots on top of the crowd
    table, overflow = build_grid(jnp.asarray(pos), jnp.asarray(active))
    table = np.asarray(table)
    entries = table[table >= 0]
    assert active[entries].all()
    assert int(overflow) == sum(per_cell_overflow(pos, active).values())
    # all-inactive: empty table, zero overflow
    table2, overflow2 = build_grid(jnp.asarray(pos), jnp.zeros(MAX_PLAYERS, dtype=bool))
    assert int(overflow2) == 0 and (np.asarray(table2) == -1).all()


def test_jit_matches_eager():
    pos, active = town_scene()
    pos_j, active_j = jnp.asarray(pos), jnp.asarray(active)
    t_e, o_e = build_grid(pos_j, active_j)
    t_j, o_j = jax.jit(build_grid)(pos_j, active_j)
    assert (np.asarray(t_e) == np.asarray(t_j)).all() and int(o_e) == int(o_j)
    i_e, d_e = nearest_in_block(t_e, pos_j, pos_j)
    i_j, d_j = jax.jit(nearest_in_block)(t_j, pos_j, pos_j)
    assert (np.asarray(i_e) == np.asarray(i_j)).all()
    assert (np.asarray(d_e) == np.asarray(d_j)).all()
