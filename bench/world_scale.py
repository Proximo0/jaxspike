"""World-build scale bench (CPU): time terrain, distance fields and make_world_batch; report memory.

Run from the repo root: python bench/world_scale.py [--batches 1 4]
MAP is a config constant, so 2048 is estimated in docs/world_scale.md, not run here.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from bots.runner import make_world_batch
from game import map_gen
from game.config import MAP


def timed(fn, *args):
    t = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    return out, time.perf_counter() - t


def live_mb():
    return sum(a.nbytes for a in jax.live_arrays()) / 2**20


def mb(a):
    return a.nbytes / 2**20


def relaxation_iterations(d):
    """Iterations of the relaxation = max finite distance + 1 (the last pass finds no change)."""
    d = jax.device_get(d)
    finite = d[d != map_gen.DIST_UNREACHABLE]
    return int(finite.max()) + 1 if finite.size else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"MAP={MAP} devices={jax.devices()}")

    key = jax.random.PRNGKey(args.seed)
    mk = jax.jit(map_gen.make_terrain)
    terrain, t_first = timed(mk, key)
    _, t_second = timed(mk, key)
    print(f"make_terrain: first(with compile) {t_first:.2f}s  second {t_second:.2f}s")

    df = jax.jit(map_gen.distance_fields)
    d, t_first = timed(df, terrain)
    _, t_second = timed(df, terrain)
    print(f"distance_fields: first {t_first:.2f}s  second {t_second:.2f}s  "
          f"relaxation iterations {relaxation_iterations(d)}")

    if hasattr(map_gen, "distance_fields_ro"):
        dro = jax.jit(map_gen.distance_fields_ro)
        ter_ro = jax.jit(map_gen.make_terrain_ro)(key) if hasattr(map_gen, "make_terrain_ro") else terrain
        d2, t_first = timed(dro, ter_ro)
        _, t_second = timed(dro, ter_ro)
        print(f"distance_fields_ro: first {t_first:.2f}s  second {t_second:.2f}s  "
              f"relaxation iterations {relaxation_iterations(d2)}")
    else:
        print("distance_fields_ro: not present in game/map_gen.py yet, skipped")

    print(f"per-world arrays (MB): dist_town {mb(d):.2f}  terrain {mb(terrain):.2f}  "
          f"walls {mb(map_gen.blocking(terrain)):.2f}  pz {mb(map_gen.protection_zone(terrain)):.2f}")
    del d, terrain

    for batch in args.batches:
        worlds, t_first = timed(make_world_batch, args.seed, batch)
        live = live_mb()
        _, t_second = timed(make_world_batch, args.seed, batch)
        total = sum(mb(a) for a in jax.tree_util.tree_leaves(worlds))
        print(f"make_world_batch({batch}): first {t_first:.2f}s ({t_first / batch:.2f}s/world)  "
              f"second {t_second:.2f}s ({t_second / batch:.2f}s/world)")
        print(f"  world pytree {total:.1f} MB ({total / batch:.2f} MB/world); jax live buffers {live:.1f} MB")
        for name in ("dist_town", "terrain", "walls", "pz"):
            print(f"  {name}: {mb(worlds[name]) / batch:.2f} MB/world")
        big = sorted(((mb(v) / batch, k) for k, v in worlds.items() if hasattr(v, "nbytes")), reverse=True)[:3]
        print("  largest world arrays: " + ", ".join(f"{k} {m:.2f}" for m, k in big))
        del worlds


if __name__ == "__main__":
    main()
