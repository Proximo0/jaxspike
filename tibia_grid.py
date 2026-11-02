"""Tile-world step with a spatial grid replacing the all-pairs distance check.

Players are bucketed into CELL x CELL tile cells each tick. Each monster only
looks at the 3x3 block of cells around it, so it chases players within about
one cell of its own cell (monster chase range becomes a design rule, not an
accident of the algorithm).

Cost per tick: O(P log P) for the sort, plus M x 9K for the lookups, instead of
M x P for the all-pairs table. K is the per-cell player capacity.
"""
import jax
import jax.numpy as jnp

from tibia_spike import DIRS, MAP, MAX_MONSTERS, MAX_PLAYERS, make_world

CELL = 8
G = MAP // CELL
NCELLS = G * G
K = 8  # max players stored per cell; overflow is dropped and counted
OFFSETS = jnp.array([[dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)], dtype=jnp.int32)
BIG = 10**6


def build_grid(ppos, pactive):
    """Return (table[NCELLS, K] of player indices or -1, overflow count)."""
    cell = (ppos // CELL)[:, 0] * G + (ppos // CELL)[:, 1]
    cell = jnp.where(pactive, cell, NCELLS)  # inactive players go to a sentinel
    order = jnp.argsort(cell)
    sorted_cell = cell[order]
    first = jnp.searchsorted(sorted_cell, sorted_cell, side="left")
    rank = jnp.arange(MAX_PLAYERS) - first
    keep = (sorted_cell < NCELLS) & (rank < K)
    overflow = jnp.sum((sorted_cell < NCELLS) & (rank >= K))
    row = jnp.where(keep, sorted_cell, NCELLS)  # dropped entries write out of bounds
    table = jnp.full((NCELLS, K), -1, dtype=jnp.int32)
    table = table.at[row, jnp.where(keep, rank, 0)].set(order.astype(jnp.int32), mode="drop")
    return table, overflow


def nearest_player(mpos, ppos, pactive):
    table, overflow = build_grid(ppos, pactive)
    mcell = mpos // CELL
    ncx = jnp.clip(mcell[:, 0:1] + OFFSETS[None, :, 0], 0, G - 1)
    ncy = jnp.clip(mcell[:, 1:2] + OFFSETS[None, :, 1], 0, G - 1)
    ncell = ncx * G + ncy  # [M, 9]
    cand = table[ncell].reshape(mpos.shape[0], 9 * K)  # [M, 9K] player indices
    valid = cand >= 0
    safe = jnp.maximum(cand, 0)
    d = jnp.abs(mpos[:, None, :] - ppos[safe]).sum(-1)
    d = jnp.where(valid, d, BIG)
    col = jnp.argmin(d, axis=1)
    dist = jnp.min(d, axis=1)
    nearest = jnp.take_along_axis(safe, col[:, None], axis=1)[:, 0]
    return nearest, dist, overflow


@jax.jit
def step_grid(world, inputs):
    walls = world["walls"]
    p, m = world["players"], world["monsters"]

    target = jnp.clip(p["pos"] + DIRS[inputs], 0, MAP - 1)
    blocked = walls[target[:, 0], target[:, 1]] == 1
    can_move = p["active"] & ~blocked
    new_ppos = jnp.where(can_move[:, None], target, p["pos"])

    nearest, dist, _ = nearest_player(m["pos"], new_ppos, p["active"])
    delta = new_ppos[nearest] - m["pos"]
    horiz = jnp.abs(delta[:, 0]) >= jnp.abs(delta[:, 1])
    step_dir = jnp.where(
        horiz[:, None],
        jnp.stack([jnp.sign(delta[:, 0]), jnp.zeros_like(delta[:, 0])], -1),
        jnp.stack([jnp.zeros_like(delta[:, 1]), jnp.sign(delta[:, 1])], -1),
    )
    mtarget = jnp.clip(m["pos"] + step_dir, 0, MAP - 1)
    mblocked = walls[mtarget[:, 0], mtarget[:, 1]] == 1
    adjacent = dist <= 1
    new_mpos = jnp.where((m["active"] & ~mblocked & ~adjacent)[:, None], mtarget, m["pos"])

    dmg = jnp.where(m["active"] & adjacent, 3, 0)
    hp_loss = jnp.zeros((MAX_PLAYERS,), jnp.int32).at[nearest].add(dmg)
    new_php = jnp.where(p["active"], jnp.maximum(p["hp"] - hp_loss + 1, 0), p["hp"])
    alive = p["active"] & (new_php > 0)

    new_mhp = m["hp"] - jnp.where(adjacent & alive[nearest], 1, 0)
    new_mactive = m["active"] & (new_mhp > 0)

    return dict(
        walls=walls,
        players=dict(pos=new_ppos, hp=new_php, active=alive),
        monsters=dict(pos=new_mpos, hp=new_mhp, active=new_mactive),
    )


@jax.jit
def brute_nearest(mpos, ppos, pactive):
    d = jnp.abs(mpos[:, None, :] - ppos[None, :, :]).sum(-1)
    d = jnp.where(pactive[None, :], d, BIG)
    return jnp.argmin(d, axis=1), jnp.min(d, axis=1)
