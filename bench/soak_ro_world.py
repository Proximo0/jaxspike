"""Soak of the RO world as if the lead had wired it: RO terrain, warp-aware distance fields, warp phase.

Nothing lead-owned is edited. The script swaps module attributes before building worlds:
  map_gen.make_terrain     -> map_gen.make_terrain_ro      (bots/runner.make_world_batch reads the attribute)
  map_gen.distance_fields  -> map_gen.distance_fields_ro   (same)
  game.step.step           -> the same phase order with ro_warps.apply_warps after movement
Monster kinds stay rat/orc: config's MONSTER_HP / MONSTER_DAMAGE have two rows until the lead switches them
to game/data/monsters.py, and monsters.py / combat.py index those tables by kind.

Checks soak_check's invariants after every chunk, then prints where the bots ARE (town, field band,
dungeon, warp tile) and where the monsters are, per the handoff lesson "print where entities are".
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from game import combat, loot, map_gen, monsters, movement, ro_warps, survival  # noqa: E402
from game import step as step_mod  # noqa: E402
from game.config import MAP, MONSTER_HP  # noqa: E402
from game.data.terrain import T_CAVE, T_STATION, T_WARP  # noqa: E402
from game.map_gen import DIST_UNREACHABLE  # noqa: E402
from game.world_layout import DUNGEONS, FIELD_BANDS, TOWN_NAMES, dungeon_index  # noqa: E402


def step_ro(world, inputs):
    """step.py's phase order with the warp phase right after movement (the line the lead adds)."""
    world = {**world, "monsters": {**world["monsters"], "died": jnp.zeros_like(world["monsters"]["died"])}}
    world = movement.move_players(world, inputs)
    world = ro_warps.apply_warps(world)
    world = monsters.monster_ai(world)
    world = combat.combat(world, inputs)
    world = loot.loot(world)
    world = survival.survival(world, inputs)
    return {**world, "tick": world["tick"] + 1}


_BASE_TERRAIN = map_gen.make_terrain


def _terrain_ro(key):
    """make_terrain_ro, with the module's make_terrain pointing at the base generator while it traces."""
    map_gen.make_terrain = _BASE_TERRAIN
    try:
        return map_gen.make_terrain_ro(key)
    finally:
        map_gen.make_terrain = _terrain_ro


def patch():
    map_gen.make_terrain = _terrain_ro
    map_gen.distance_fields = map_gen.distance_fields_ro
    step_mod.step = step_ro


def where_are(worlds):
    """Print counts of living bots and active monsters by region, summed over the batch."""
    w = jax.device_get(worlds)
    p, m = w["players"], w["monsters"]
    terrain = np.asarray(w["terrain"])
    pz = np.asarray(w["pz"])
    gate = np.asarray(w["dist_town"]).min(axis=1)  # [B, MAP, MAP]
    batch = terrain.shape[0]
    bands = [f"band{k} ({lo:.2f}-{hi:.2f} MAP)" for k, (lo, hi, _) in enumerate(FIELD_BANDS)]
    for label, ent, alive in (("bots", p, np.asarray(p["active"]) & ~np.asarray(p["dead"])),
                              ("monsters", m, np.asarray(m["active"]))):
        counts = {"in town (pz)": 0, "on warp": 0, "on station": 0, "unreachable field": 0}
        for b_ in bands:
            counts[b_] = 0
        for k, d in enumerate(DUNGEONS):
            counts[f"dungeon {k} {d[3]}"] = 0
        for b in range(batch):
            pos = np.asarray(ent["pos"][b])[alive[b]]
            x, y = pos[:, 0], pos[:, 1]
            t = terrain[b][x, y]
            dix = np.asarray(dungeon_index(jnp.asarray(pos)))
            g = gate[b][x, y]
            counts["in town (pz)"] += int(pz[b][x, y].sum())
            counts["on warp"] += int((t == T_WARP).sum())
            counts["on station"] += int((t == T_STATION).sum())
            for k, d in enumerate(DUNGEONS):
                counts[f"dungeon {k} {d[3]}"] += int((dix == k).sum())
            field = (dix < 0) & ~pz[b][x, y]
            counts["unreachable field"] += int((field & (g >= DIST_UNREACHABLE)).sum())
            for k, (lo, hi, _) in enumerate(FIELD_BANDS):
                sel = field & (g < DIST_UNREACHABLE) & (g >= lo * MAP) & (g < hi * MAP)
                counts[bands[k]] += int(sel.sum())
        total = int(alive.sum())
        print(f"where the {label} are (total {total}, {batch} worlds):")
        for k_, v in counts.items():
            print(f"  {k_:36s} {v:6d}  ({100.0 * v / max(total, 1):5.1f}%)")
    cave = (terrain == T_CAVE).sum() / batch
    print(f"cave tiles per world: {cave:.0f}; towns: {', '.join(TOWN_NAMES)}")


def main(batch=4, chunks=5, ticks_per_chunk=100, seed=0):
    patch()
    from bench.soak_check import check_invariants
    from bots.runner import make_world_batch, run_games

    print("platform:", jax.devices()[0].platform, "(RO world: make_terrain_ro, distance_fields_ro, warps)", flush=True)
    t0 = time.perf_counter()
    worlds = make_world_batch(seed, batch)
    jax.block_until_ready(worlds)
    print(f"built {batch} worlds in {time.perf_counter() - t0:.1f}s", flush=True)
    kinds = np.asarray(worlds["monsters"]["kind"])[np.asarray(worlds["monsters"]["active"])]
    print(f"monsters per world by kind: {np.bincount(kinds, minlength=len(MONSTER_HP)) / batch}", flush=True)
    where_are(worlds)
    total_problems = []
    kills0 = 0
    t0 = time.perf_counter()
    for c in range(chunks):
        worlds, stats = run_games(worlds, ticks_per_chunk)
        jax.block_until_ready(worlds)
        probs = check_invariants(worlds)
        total_problems += probs
        kills = int(jnp.sum(stats["kills"]))
        deaths = int(jnp.sum(stats["deaths"]))
        print(
            f"chunk {c + 1:2d} (tick {(c + 1) * ticks_per_chunk:5d}): kills={kills} deaths={deaths} "
            f"alive_players={int(jnp.sum(stats['alive']))} items_held={int(jnp.sum(stats['items']))} "
            f"invariant_violations={len(probs)}",
            flush=True,
        )
        kills0 = kills
    dt = time.perf_counter() - t0
    ticks = chunks * ticks_per_chunk
    n_bots = int(np.asarray(worlds["players"]["active"]).sum())
    print(f"total {ticks} ticks x {batch} worlds in {dt:.1f}s; "
          f"{1000.0 * kills0 / max(n_bots * ticks, 1):.2f} kills per bot per 1000 ticks")
    where_are(worlds)
    if total_problems:
        print("FIRST VIOLATIONS:")
        for p_ in total_problems[:15]:
            print("  ", p_)
    else:
        print("all invariants held")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=100, help="ticks per chunk")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(batch=a.batch, chunks=a.chunks, ticks_per_chunk=a.ticks, seed=a.seed)
