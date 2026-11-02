"""M1 soak: bot games with the RO stats phase after every step, checking the world and stats invariants.

Same CLI as soak_check.py (--batch --chunks --ticks --seed). Reuses make_world_batch and check_invariants.
Prints per chunk: mean/max base level, mean job level, job changes, total exp.
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from bench.soak_check import check_invariants  # noqa: E402
from bots.ro_stats_policy import bot_allot, bot_job  # noqa: E402
from bots.runner import make_world_batch  # noqa: E402
from game import ro_stats  # noqa: E402
from game.config import MONSTER_HP  # noqa: E402
from game.data import stats as D  # noqa: E402
from game.state import next_key  # noqa: E402


def add_stats(worlds):
    """Insert a fresh stats block into every world of the batch (broadcast of make_state)."""
    batch = worlds["tick"].shape[0]
    block = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (batch,) + a.shape), ro_stats.make_state())
    return {**worlds, "stats": block}


def _stats_summary(worlds):
    s, act = worlds["stats"], worlds["players"]["active"]
    n = jnp.maximum(act.sum(), 1)
    return {
        "mean_level": (s["base_level"] * act).sum() / n,
        "max_level": jnp.where(act, s["base_level"], 0).max(),
        "mean_job_level": (s["job_level"] * act).sum() / n,
        "job_changes": ((s["job"] != D.JOB_NOVICE) & act).sum(),
        "total_exp": (s["base_exp"].astype(jnp.float32) * act).sum(),
    }


def run_games(worlds, ticks):
    """Play `ticks` ticks of every world with bots: step, then ro_stats.phase with allot/job inputs."""
    from bots.policies import bot_inputs
    from game.step import step

    def one(w, key):
        inputs = bot_inputs(w, key)
        w2 = step(w, inputs)
        extra = {**inputs, "allot": bot_allot(w, key), "job": bot_job(w, key)}
        return ro_stats.phase(w2, extra)

    def tick(ws, _):
        keys, ws = jax.vmap(next_key)(ws)
        return jax.vmap(one)(ws, keys), None

    final, _ = jax.lax.scan(tick, worlds, None, length=ticks)
    p = final["players"]
    return final, {
        "kills": p["kills"].sum(axis=1).astype(jnp.int32),
        "deaths": p["deaths"].sum(axis=1).astype(jnp.int32),
    }


def check_stats(worlds):
    """Return violation strings for the stats block across the batch."""
    w = jax.device_get(worlds)
    s = {k: np.asarray(v) for k, v in w["stats"].items()}
    act = np.asarray(w["players"]["active"])
    probs = []
    base_tab = np.asarray(D.BASE_EXP_TABLE_J)
    job_tab = np.asarray(D.JOB_EXP_TABLE_J)
    max_job = np.asarray(D.MAX_JOB_LEVEL_J)
    for b in range(act.shape[0]):
        a = act[b]
        if not a.any():
            continue

        def g(k):
            return s[k][b][a]

        for k in D.STAT_NAMES:
            v = g(k)
            if ((v < 1) | (v > D.MAX_STAT)).any():
                probs.append(f"world {b}: stat {k} out of 1..{D.MAX_STAT}")
        bl, jl, job = g("base_level"), g("job_level"), g("job")
        if ((bl < 1) | (bl > D.MAX_BASE_LEVEL)).any():
            probs.append(f"world {b}: base_level out of range")
        if ((job < 0) | (job >= D.NUM_JOBS)).any():
            probs.append(f"world {b}: job out of range")
            continue
        if ((jl < 1) | (jl > max_job[job])).any():
            probs.append(f"world {b}: job_level out of range for job")
        be = g("base_exp")
        need = base_tab[np.clip(bl, 1, D.MAX_BASE_LEVEL)]
        if ((be < 0) | ((need > 0) & (be >= need)) | ((need == 0) & (be != 0))).any():
            probs.append(f"world {b}: base_exp outside [0, table[level])")
        je = g("job_exp")
        jneed = job_tab[job, np.clip(jl, 1, job_tab.shape[1] - 1)]
        if ((je < 0) | ((jneed > 0) & (je >= jneed)) | ((jneed == 0) & (je != 0))).any():
            probs.append(f"world {b}: job_exp outside [0, table[job_level])")
        if (g("stat_points") < 0).any():
            probs.append(f"world {b}: stat_points negative")
        if (g("skill_points") < 0).any():
            probs.append(f"world {b}: skill_points negative")
        sp, msp = g("sp"), g("max_sp")
        if ((sp < 0) | (sp > msp)).any():
            probs.append(f"world {b}: sp outside [0, max_sp]")
    return probs


def main(batch=8, chunks=10, ticks_per_chunk=100, seed=0):
    print("platform:", jax.devices()[0].platform, flush=True)
    worlds = add_stats(make_world_batch(seed, batch))
    kinds = np.asarray(worlds["monsters"]["kind"])[np.asarray(worlds["monsters"]["active"])]
    print(f"monsters per world by kind: {np.bincount(kinds, minlength=len(MONSTER_HP)) / batch}", flush=True)
    total_problems = []
    t0 = time.perf_counter()
    for c in range(chunks):
        worlds, st = run_games(worlds, ticks_per_chunk)
        jax.block_until_ready(worlds)
        probs = check_invariants(worlds) + check_stats(worlds)
        total_problems += probs
        sm = jax.device_get(_stats_summary(worlds))
        print(
            f"chunk {c + 1:2d} (tick {(c + 1) * ticks_per_chunk:5d}): "
            f"kills={int(st['kills'].sum())} deaths={int(st['deaths'].sum())} "
            f"mean_base_level={float(sm['mean_level']):.2f} max_base_level={int(sm['max_level'])} "
            f"mean_job_level={float(sm['mean_job_level']):.2f} job_changes={int(sm['job_changes'])} "
            f"total_base_exp={float(sm['total_exp']):.0f} invariant_violations={len(probs)}",
            flush=True,
        )
    dt = time.perf_counter() - t0
    print(f"total {chunks * ticks_per_chunk} ticks x {batch} worlds in {dt:.1f}s")
    if total_problems:
        print("FIRST VIOLATIONS:")
        for p_ in total_problems[:15]:
            print("  ", p_)
    else:
        print("all invariants held")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=100, help="ticks per chunk")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(batch=a.batch, chunks=a.chunks, ticks_per_chunk=a.ticks, seed=a.seed)
