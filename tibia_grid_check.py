"""Correctness and timing for the spatial-grid step.

1. Correctness: where a monster has a player in its 3x3 cell neighbourhood,
   the grid's nearest-player distance must equal the brute-force distance.
   Monsters with no such player are reported as out-of-range (expected).
2. Overflow: count of players dropped because a cell exceeded K.
3. Timing: grid step at several batch sizes on the current backend.
"""
import sys
import time

import jax
import jax.numpy as jnp

from tibia_grid import BIG, CELL, K, brute_nearest, nearest_player, step_grid
from tibia_spike import MAX_PLAYERS, make_world


def neighbourhood_brute(mpos, ppos, pactive):
    """Brute-force nearest restricted to players in the monster's 3x3 cell block."""
    mcell = mpos // CELL
    pcell = ppos // CELL
    dx = jnp.abs(mcell[:, None, 0] - pcell[None, :, 0])
    dy = jnp.abs(mcell[:, None, 1] - pcell[None, :, 1])
    in_block = (dx <= 1) & (dy <= 1)
    d = jnp.abs(mpos[:, None, :] - ppos[None, :, :]).sum(-1)
    d = jnp.where(in_block & pactive[None, :], d, BIG)
    return jnp.min(d, axis=1)


def check_correctness(seed=0, worlds=8):
    keys = jax.random.split(jax.random.PRNGKey(seed), worlds)
    agree = found_total = out_of_block = overflow_total = 0
    for w in keys:
        world = make_world(w)
        p, m = world["players"], world["monsters"]
        _, g_dist, overflow = nearest_player(m["pos"], p["pos"], p["active"])
        ref = neighbourhood_brute(m["pos"], p["pos"], p["active"])
        full = brute_nearest(m["pos"], p["pos"], p["active"])[1]
        g_dist, ref, full, overflow = map(jax.device_get, (g_dist, ref, full, overflow))
        active = jax.device_get(m["active"])
        found = (g_dist < BIG) & active
        found_total += int(found.sum())
        agree += int(((g_dist == ref) & found).sum())
        out_of_block += int(((ref >= BIG) & (full < BIG) & active).sum())
        overflow_total += int(overflow)
    print(
        f"grid vs brute-in-3x3 over {worlds} worlds: "
        f"{agree}/{found_total} active monsters with a player in block match exactly; "
        f"{out_of_block} monsters have a player only outside their 3x3 block (chase range limit); "
        f"overflow dropped {overflow_total} players"
    )


def time_grid(batch, ticks=20):
    keys = jax.random.split(jax.random.PRNGKey(1), batch)
    worlds = jax.vmap(make_world)(keys)
    base = jnp.arange(MAX_PLAYERS)[None, :] + jnp.arange(batch)[:, None]
    vstep = jax.vmap(step_grid)
    worlds = vstep(worlds, base % 5)
    jax.block_until_ready(worlds)
    t0 = time.perf_counter()
    for i in range(ticks):
        worlds = vstep(worlds, (base + i) % 5)
    jax.block_until_ready(worlds)
    dt = (time.perf_counter() - t0) / ticks
    print(
        f"grid batch {batch:5d}: {dt*1000:9.2f} ms/tick total, "
        f"{dt*1000/batch:.4f} ms/world, platform={jax.devices()[0].platform}"
    )


if __name__ == "__main__":
    print("platform:", jax.devices()[0].platform, "K =", K)
    check_correctness()
    for b in (256, 1024):
        time_grid(b)
