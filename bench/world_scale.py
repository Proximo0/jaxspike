"""World-build scale bench (CPU): time terrain, distance fields and make_world_batch; report memory.

Run from the repo root: python bench/world_scale.py [--batches 1 4] [--no-base] [--census TICKS]
MAP is a config constant: to measure 2048, run this from a copy of the repo with MAP = 2048 in its
game/config.py (docs/world_scale.md has the commands and the numbers). --census plays TICKS bot ticks
and prints where the players and monsters are, per zone (world_layout.ZONES).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from bots.runner import make_world_batch, run_games
from game import map_gen, world_layout
from game.config import MAP


def timed(fn, *args):
    t = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    return out, time.perf_counter() - t


def live_mb():
    return sum(a.nbytes for a in jax.live_arrays()) / 2**20


def peak_rss_mb():
    """Peak resident memory of this process so far (Windows working set, or Linux ru_maxrss)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        c = PMC()
        c.cb = ctypes.sizeof(c)
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10


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
    ap.add_argument("--no-base", action="store_true", help="skip the plain (warp-less) distance_fields timing")
    ap.add_argument("--census", type=int, default=0, help="bot ticks before a per-zone census (0 = none)")
    args = ap.parse_args()
    print(f"MAP={MAP} devices={jax.devices()}", flush=True)

    key = jax.random.PRNGKey(args.seed)
    mk = jax.jit(map_gen.make_terrain)
    terrain, t_first = timed(mk, key)
    _, t_second = timed(mk, key)
    print(f"make_terrain: first(with compile) {t_first:.2f}s  second {t_second:.2f}s", flush=True)

    if not args.no_base:
        df = jax.jit(map_gen.distance_fields)
        d, t_first = timed(df, terrain)
        _, t_second = timed(df, terrain)
        print(f"distance_fields: first {t_first:.2f}s  second {t_second:.2f}s  "
              f"relaxation iterations {relaxation_iterations(d)}", flush=True)

    if hasattr(map_gen, "distance_fields_ro"):
        dro = jax.jit(map_gen.distance_fields_ro)
        ter_ro = jax.jit(map_gen.make_terrain_ro)(key) if hasattr(map_gen, "make_terrain_ro") else terrain
        d2, t_first = timed(dro, ter_ro)
        _, t_second = timed(dro, ter_ro)
        print(f"distance_fields_ro: first {t_first:.2f}s  second {t_second:.2f}s  "
              f"relaxation iterations {relaxation_iterations(d2)}  peak RSS {peak_rss_mb():.0f} MB", flush=True)
        d = d2
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
        print(f"  world pytree {total:.1f} MB ({total / batch:.2f} MB/world); jax live buffers {live:.1f} MB; "
              f"peak RSS {peak_rss_mb():.0f} MB")
        for name in ("dist_town", "terrain", "walls", "pz"):
            print(f"  {name}: {mb(worlds[name]) / batch:.2f} MB/world")
        big = sorted(((mb(v) / batch, k) for k, v in worlds.items() if hasattr(v, "nbytes")), reverse=True)[:3]
        print("  largest world arrays: " + ", ".join(f"{k} {m:.2f}" for m, k in big), flush=True)
        if args.census:
            census(worlds, args.census)
        del worlds


def census(worlds, ticks):
    """Play `ticks` bot ticks, then print living players and active monsters per zone (summed over worlds)."""
    t = time.perf_counter()
    worlds, _ = jax.block_until_ready(run_games(worlds, ticks))
    print(f"census after {ticks} ticks ({time.perf_counter() - t:.1f}s):")
    zone_of = jax.jit(jax.vmap(world_layout.zone_of))
    p, m = worlds["players"], worlds["monsters"]
    pz = np.asarray(zone_of(p["pos"]))[np.asarray(p["active"] & ~p["dead"])]
    mz = np.asarray(zone_of(m["pos"]))[np.asarray(m["active"])]
    pc, mc = np.bincount(pz, minlength=world_layout.N_ZONES), np.bincount(mz, minlength=world_layout.N_ZONES)
    for i, z in enumerate(world_layout.ZONES):
        print(f"  {i:2d} {z[0]:<11} {world_layout.ZONE_KINDS[z[1]]:<7} players {pc[i]:5d}  monsters {mc[i]:5d}")


if __name__ == "__main__":
    main()
