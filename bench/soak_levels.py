"""Leveling-curve soak: how fast RO bots level, where they die, and where they are.

Builds RO worlds (bots.runner.make_world_batch), adds the dungeon entrance fields the bots dive with
(bots.ro_nav_policy.dungeon_fields as world["dist_dungeon"]; --no-dungeons leaves them out), plays the composed bot
policy in chunks and prints after each chunk: base / job level histogram, kills and deaths per bot per 1000 ticks,
where living bots are (town, field band 0..2, each dungeon), and the bench/soak_check invariants. At the end:
ticks until the first / median bot reached base 10 and Novice job 10, ticks to first-job job 10 per job, deaths per
base level, compared with classic RO (Novice to job 10 takes a new player about 1-2 hours; at TICK_MS per tick that
is the RO_* tick range printed).

    venv\\Scripts\\python.exe bench/soak_levels.py --batch 2 --chunks 20 --ticks 250
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from bench.soak_check import check_invariants  # noqa: E402
from bots import ro_nav_policy as nav  # noqa: E402
from bots.ro_policies import ro_bot_inputs  # noqa: E402
from bots.runner import POP_PLAYERS, make_world_batch  # noqa: E402
from game import ro_spawns  # noqa: E402
from game.config import TICK_MS  # noqa: E402
from game.data.skills import N_JOBS  # noqa: E402
from game.state import next_key  # noqa: E402
from game.step import step  # noqa: E402
from game.world_layout import DUNGEONS, dungeon_index  # noqa: E402

JOB_NAMES = ("Novice", "Swordsman", "Mage", "Archer", "Acolyte", "Thief", "Merchant")
RO_NOVICE_JOB10_HOURS = (1.0, 2.0)
TARGET_LEVEL = 10


def play(worlds, ticks):
    def tick(ws, _):
        keys, ws = jax.vmap(next_key)(ws)
        return jax.vmap(step)(ws, jax.vmap(ro_bot_inputs)(ws, keys)), None

    out, _ = jax.lax.scan(tick, worlds, None, length=ticks)
    return out


play_jit = jax.jit(play, static_argnums=1)


def where_bots_are(worlds):
    """Fractions of living bots: town, band 0..2, dungeon k."""
    def one(w):
        pl = w["players"]
        live = pl["active"] & ~pl["dead"]
        pos = pl["pos"]
        town = w["pz"][pos[:, 0], pos[:, 1]]
        dun = dungeon_index(pos)
        band = ro_spawns.band_of(w, pos)
        cols = [town] + [~town & (dun < 0) & (band == b) for b in range(3)] + [dun == k for k in range(len(DUNGEONS))]
        return jnp.stack([(c & live).sum() for c in cols]), live.sum()

    counts, live = jax.vmap(one)(worlds)
    return np.asarray(counts.sum(0)) / max(int(live.sum()), 1)


def main(batch=2, chunks=20, ticks=250, seed=0, dungeons=True):
    print("platform:", jax.devices()[0].platform, flush=True)
    t0 = time.perf_counter()
    worlds = make_world_batch(seed, batch)
    if dungeons:
        worlds = {**worlds, "dist_dungeon": jax.lax.map(nav.dungeon_fields, worlds["terrain"])}
    jax.block_until_ready(worlds)
    print(f"built {batch} worlds in {time.perf_counter() - t0:.1f}s (dungeon fields: {dungeons})", flush=True)

    n = POP_PLAYERS
    first_base10 = np.full((batch, n), -1)
    first_nov_job10 = np.full((batch, n), -1)
    first_job10_by_job = {j: [] for j in range(1, N_JOBS)}
    seen_first_job10 = np.zeros((batch, n), bool)
    deaths_at_level = np.zeros(100, np.int64)
    bot_chunks_at_level = np.zeros(100, np.int64)
    prev_deaths = np.zeros((batch, n), np.int64)
    problems = []
    names = ["town", "band0", "band1", "band2"] + [d[3] for d in DUNGEONS]
    t0 = time.perf_counter()
    for c in range(chunks):
        lvl_before = np.asarray(worlds["stats"]["base_level"])[:, :n]
        worlds = play_jit(worlds, ticks)
        jax.block_until_ready(worlds)
        tick = (c + 1) * ticks
        st = jax.device_get(worlds["stats"])
        pl = jax.device_get(worlds["players"])
        blv, jlv, job = st["base_level"][:, :n], st["job_level"][:, :n], st["job"][:, :n]
        deaths = pl["deaths"][:, :n].astype(np.int64)
        np.add.at(deaths_at_level, lvl_before.ravel(), (deaths - prev_deaths).ravel())
        np.add.at(bot_chunks_at_level, lvl_before.ravel(), 1)
        prev_deaths = deaths
        first_base10[(first_base10 < 0) & (blv >= TARGET_LEVEL)] = tick
        # a first job means the Novice reached job 10 at some point (bot_job changes jobs at Novice job 10)
        first_nov_job10[(first_nov_job10 < 0) & (((job == 0) & (jlv >= 10)) | (job > 0))] = tick
        newly = ~seen_first_job10 & (job > 0) & (jlv >= TARGET_LEVEL)
        for b, i in zip(*np.nonzero(newly)):
            first_job10_by_job[int(job[b, i])].append(tick - first_nov_job10[b, i])
        seen_first_job10 |= newly
        probs = check_invariants(worlds)
        problems += probs
        kills = pl["kills"][:, :n].sum() / (batch * n) * 1000 / tick
        dpb = deaths.sum() / (batch * n) * 1000 / tick
        where = where_bots_are(worlds)
        jobs = np.bincount(job.ravel(), minlength=N_JOBS)
        print(f"tick {tick:6d}: base lv mean {blv.mean():5.2f} max {blv.max():2d} | job lv mean {jlv.mean():5.2f} | "
              f"jobs {dict(zip(JOB_NAMES, jobs.tolist()))} | kills/bot/1k {kills:4.2f} deaths/bot/1k {dpb:5.3f} | "
              f"where {' '.join(f'{k}={v:.1%}' for k, v in zip(names, where))} | violations {len(probs)}",
              flush=True)
    dt = time.perf_counter() - t0
    total = chunks * ticks
    print(f"\n{total} ticks x {batch} worlds x {n} bots in {dt:.1f}s; TICK_MS {TICK_MS}")
    print("base level histogram:", dict(enumerate(np.bincount(blv.ravel(), minlength=blv.max() + 1).tolist())))

    def summary(arr, label):
        hit = arr[arr >= 0]
        if hit.size:
            print(f"{label}: {hit.size}/{arr.size} bots; first at tick {hit.min()}, median {int(np.median(hit))} "
                  f"(= {np.median(hit) * TICK_MS / 3.6e6:.2f} h of game time)")
        else:
            print(f"{label}: no bot within {total} ticks")

    summary(first_base10, f"base {TARGET_LEVEL}")
    summary(first_nov_job10, "Novice job 10")
    for j, v in first_job10_by_job.items():
        if v:
            print(f"  {JOB_NAMES[j]}: job 10 after the job change in median {int(np.median(v))} ticks ({len(v)} bots)")
    lo, hi = (h * 3.6e6 / TICK_MS for h in RO_NOVICE_JOB10_HOURS)
    print(f"RO reference: Novice to job 10 in {RO_NOVICE_JOB10_HOURS[0]}-{RO_NOVICE_JOB10_HOURS[1]} h "
          f"= {lo:.0f}-{hi:.0f} ticks at {TICK_MS} ms")
    print("deaths per 1000 bot-ticks by base level (level: deaths, bot-ticks):")
    for lv in np.nonzero(bot_chunks_at_level)[0]:
        bt = bot_chunks_at_level[lv] * ticks
        print(f"  {lv:2d}: {deaths_at_level[lv]:4d} deaths over {bt:8d} bot-ticks = {deaths_at_level[lv] * 1000 / bt:.3f}")
    print("all invariants held" if not problems else f"VIOLATIONS ({len(problems)}): {problems[:10]}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--ticks", type=int, default=250, help="ticks per chunk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dungeons", action="store_true", help="leave world['dist_dungeon'] out: bots never dive")
    a = ap.parse_args()
    main(a.batch, a.chunks, a.ticks, a.seed, not a.no_dungeons)
