"""Soak for the M4 gather/craft phase: bot games with the craft block, checking world and craft invariants.

Invariants after every chunk: everything bench/soak_check.check_invariants checks (its NUM_ITEMS inventory rule is
replaced by "id in 0..3 or in CRAFT_ITEM_IDS"), plus node, respawn, station and level rules (see check_craft).
Prints per-chunk totals and, at the end, gather/craft rates, skill level histograms and where the gatherer bots stand.
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from bench.soak_check import check_invariants  # noqa: E402
from bots.runner import POP_PLAYERS, make_world_batch  # noqa: E402
from game import ro_craft  # noqa: E402
from game.config import MAP  # noqa: E402
from game.data.craft import (  # noqa: E402
    CRAFT_ITEM_IDS, CRAFT_MAX_LEVEL, N_CRAFT_SKILLS, N_GATHER_KINDS, SKILL_NAMES)
from game.map_gen import DIST_UNREACHABLE  # noqa: E402
from game.state import next_key  # noqa: E402
from game.step import step  # noqa: E402

from bots.ro_craft_policy import GATHER_EVERY, craft_bot_inputs  # noqa: E402

POLICY = "bots.ro_craft_policy"

_ALLOWED_INV = np.array([0, 1, 2, 3, *CRAFT_ITEM_IDS])


def build_worlds(seed, batch):
    worlds = make_world_batch(seed, batch)
    worlds = jax.vmap(lambda w: {**w, "craft": ro_craft.make_state()})(worlds)
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), batch)
    worlds = jax.vmap(ro_craft.place_nodes)(worlds, keys)
    dist = jax.lax.map(ro_craft.node_fields, worlds)  # sequential over worlds
    return {**worlds, "craft": {**worlds["craft"], "dist_node": dist}}


def make_runner(ticks):
    def tick(ws, _):
        keys, ws = jax.vmap(next_key)(ws)
        inputs = jax.vmap(craft_bot_inputs)(ws, keys)
        ws = jax.vmap(step)(ws, inputs)
        return jax.vmap(ro_craft.phase)(ws, inputs), None

    @jax.jit
    def run(ws):
        return jax.lax.scan(tick, ws, None, length=ticks)[0]
    return run


def check_craft(worlds):
    """Craft-specific and adapted base invariants; returns violation strings."""
    problems = [p for p in check_invariants(worlds) if "inventory value out of range" not in p]
    w = jax.device_get(worlds)
    walls, pz = np.asarray(w["walls"]), np.asarray(w["pz"])
    cr = w["craft"]
    for b in range(walls.shape[0]):
        n = {k: np.asarray(v[b]) for k, v in cr["nodes"].items()}
        placed, active, kind, pos, rin = n["placed"], n["active"], n["kind"], n["pos"], n["respawn_in"]
        inv = np.asarray(w["players"]["inv"][b])
        if not np.isin(inv, _ALLOWED_INV).all():
            problems.append(f"world {b}: inventory holds an unknown item id")
        gp = placed & (kind < N_GATHER_KINDS)
        if gp.any():
            gx, gy = pos[gp, 0], pos[gp, 1]
            if (walls[b][gx, gy] != 0).any():
                problems.append(f"world {b}: gather node on a wall")
            if pz[b][gx, gy].any():
                problems.append(f"world {b}: gather node on a protection-zone tile")
        pp = pos[placed]
        if len({tuple(t) for t in pp}) != len(pp):
            problems.append(f"world {b}: two placed nodes share a tile")
        if (rin < 0).any():
            problems.append(f"world {b}: respawn_in negative")
        if (placed & ~active & (rin <= 0)).any():
            problems.append(f"world {b}: inactive node without respawn_in > 0")
        if (placed & (kind >= N_GATHER_KINDS) & ~active).any():
            problems.append(f"world {b}: station inactive")
        lvl, exp = np.asarray(cr["level"][b]), np.asarray(cr["exp"][b])
        if not (lvl == np.asarray(ro_craft.level_from_exp(jnp.asarray(exp)))).all():
            problems.append(f"world {b}: level != level_from_exp(exp)")
        if ((lvl < 1) | (lvl > CRAFT_MAX_LEVEL)).any():
            problems.append(f"world {b}: level out of range")
    return problems


def _near(m, x, y):
    """True if the boolean map m is set within manhattan 1 of (x, y)."""
    p = np.pad(m, 1)
    return bool(p[x + 1, y + 1] or p[x, y + 1] or p[x + 2, y + 1] or p[x + 1, y] or p[x + 1, y + 2])


def gatherer_report(worlds, ticks_total):
    w = jax.device_get(worlds)
    batch = np.asarray(w["walls"]).shape[0]
    slots = np.arange(POP_PLAYERS)
    gslots = slots[slots % GATHER_EVERY == 1]
    kinds = (gslots // GATHER_EVERY) % N_GATHER_KINDS
    cr, pl = w["craft"], w["players"]
    act = np.asarray(pl["active"])[:, gslots]
    gathers = np.asarray(cr["gathers"])[:, gslots][act].sum()
    crafts = np.asarray(cr["crafts"])[:, gslots][act].sum()
    ng = max(int(act.sum()), 1)
    print(f"gatherers: {ng} bots ({len(gslots)} slots x {batch} worlds)")
    print(f"gathers per gatherer per 1000 ticks: {gathers / ng / ticks_total * 1000:.2f}")
    print(f"crafts  per gatherer per 1000 ticks: {crafts / ng / ticks_total * 1000:.2f}")
    print("level histogram per skill (level: players, active population):")
    lv = np.asarray(cr["level"])[:, :POP_PLAYERS]
    for s in range(N_CRAFT_SKILLS):
        h = np.bincount(lv[:, :, s].ravel(), minlength=CRAFT_MAX_LEVEL + 1)[1:]
        print(f"  {SKILL_NAMES[s]:10s} {({i + 1: int(c) for i, c in enumerate(h) if c})}")
    on_pz = near_act = near_inact = elsewhere = full = 0
    dsum, dn, dunr = 0.0, 0, 0
    for b in range(batch):
        n = {k: np.asarray(v[b]) for k, v in cr["nodes"].items()}
        gp = n["placed"] & (n["kind"] < N_GATHER_KINDS)
        amap = np.zeros((MAP, MAP), bool)
        imap = np.zeros((MAP, MAP), bool)
        for m, sel in ((amap, gp & n["active"]), (imap, gp & ~n["active"])):
            m[n["pos"][sel, 0], n["pos"][sel, 1]] = True
        dist = np.asarray(cr["dist_node"][b])
        pzb = np.asarray(w["pz"][b])
        for i, k in zip(gslots, kinds):
            if not (pl["active"][b][i] and not pl["dead"][b][i]):
                continue
            x, y = (int(v) for v in np.asarray(pl["pos"][b][i]))
            if (np.asarray(pl["inv"][b][i]) != 0).all():
                full += 1
            d = int(dist[k, x, y])
            if d >= DIST_UNREACHABLE:
                dunr += 1
            else:
                dsum += d
                dn += 1
            if pzb[x, y]:
                on_pz += 1
            elif _near(amap, x, y):
                near_act += 1
            elif _near(imap, x, y):
                near_inact += 1
            else:
                elsewhere += 1
    tot = on_pz + near_act + near_inact + elsewhere
    print(f"gatherer positions (living, n={tot}): on pz={on_pz} near active node={near_act} "
          f"near inactive node={near_inact} elsewhere={elsewhere}")
    print(f"mean dist_node[own kind] at gatherer positions: {dsum / max(dn, 1):.1f} "
          f"(reachable n={dn}, unreachable n={dunr})")
    print(f"gatherers with a full inventory: {full}")


def main(batch=4, chunks=10, ticks_per_chunk=100, seed=0):
    print("platform:", jax.devices()[0].platform, "| policy:", POLICY, flush=True)
    t0 = time.perf_counter()
    worlds = build_worlds(seed, batch)
    jax.block_until_ready(worlds)
    print(f"built {batch} worlds with nodes in {time.perf_counter() - t0:.1f}s", flush=True)
    run = make_runner(ticks_per_chunk)
    total_problems = []
    t0 = time.perf_counter()
    prev_g = prev_c = 0
    for c in range(chunks):
        worlds = run(worlds)
        jax.block_until_ready(worlds)
        probs = check_craft(worlds)
        total_problems += probs
        p, cr = worlds["players"], worlds["craft"]
        kills, deaths = int(p["kills"].sum()), int(p["deaths"].sum())
        g, cf = int(cr["gathers"].sum()), int(cr["crafts"].sum())
        print(f"chunk {c + 1:2d} (tick {(c + 1) * ticks_per_chunk:5d}): kills={kills} deaths={deaths} "
              f"gathers={g} (+{g - prev_g}) crafts={cf} (+{cf - prev_c}) invariant_violations={len(probs)}", flush=True)
        prev_g, prev_c = g, cf
    dt = time.perf_counter() - t0
    print(f"total {chunks * ticks_per_chunk} ticks x {batch} worlds in {dt:.1f}s "
          "(kills/deaths/gathers/crafts are cumulative)")
    gatherer_report(worlds, chunks * ticks_per_chunk)
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
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=100, help="ticks per chunk")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(batch=a.batch, chunks=a.chunks, ticks_per_chunk=a.ticks, seed=a.seed)
