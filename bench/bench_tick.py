"""Per-phase timing of the jitted tick on a populated world batch (round 3, team ro3-perf).

Mirrors the phase list of game/step.py (lead-owned; keep PHASES in step order when it changes). For every phase:
  alone   p50 ms of the phase jitted + vmapped over the batch, fed the same warmed-up worlds every call and
          returning only the world leaves the phase changes (so an untouched MAP x MAP array costs nothing here)
  prefix  p50 ms of step truncated after this phase minus the previous truncation (what the phase adds inside
          the fused step; sums to the whole step within noise)
Then the whole `step` (jit + vmap) and its per-world cost. CPU by default (JAX_PLATFORMS=cpu unless set).

  venv\\Scripts\\python.exe bench/bench_tick.py --batch 4 --warm-ticks 50 --calls 30
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _phases():
    import jax.numpy as jnp
    from game import (combat, loot, monsters, movement, ro_anim, ro_craft, ro_items, ro_kafra, ro_mvp, ro_party,
                      ro_shop, ro_skills, ro_stats, ro_warps, survival)

    def clear_died(w, i):
        return {**w, "monsters": {**w["monsters"], "died": jnp.zeros_like(w["monsters"]["died"])}}

    def tick_inc(w, i):
        return {**w, "tick": w["tick"] + 1}

    return [
        ("clear_died", clear_died),
        ("movement", movement.move_players),
        ("warps", lambda w, i: ro_warps.apply_warps(w)),
        ("monsters", lambda w, i: monsters.monster_ai(w)),
        ("skills", ro_skills.phase),
        ("combat", combat.combat),
        ("loot", lambda w, i: loot.loot(w)),
        ("stats", ro_stats.phase),
        ("items", ro_items.phase),
        ("shop", ro_shop.phase),
        ("survival", survival.survival),
        ("craft", ro_craft.phase),
        ("party", ro_party.phase),
        ("mvp", lambda w, i: ro_mvp.phase(w)),
        ("kafra", ro_kafra.phase),
        ("anim", lambda w, i: ro_anim.phase(w)),
        ("tick", tick_inc),
    ]


def _changed_paths(fn, world1, inputs1):
    """Key paths of the leaves `fn` replaces, found by one eager call on a single world (unchanged leaves are the
    same objects, since phases build `{**world, ...}`)."""
    import jax
    out = fn(world1, inputs1)
    before = dict(jax.tree_util.tree_flatten_with_path(world1)[0])
    changed = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(out)[0]:
        if path not in before or before[path] is not leaf:
            changed.append(path)
    return changed


def _pick(tree, paths):
    import jax
    flat = dict(jax.tree_util.tree_flatten_with_path(tree)[0])
    return [flat[p] for p in paths]


def _time(fn, args, calls, warm):
    import jax
    import numpy as np
    out = fn(*args)
    jax.block_until_ready(out)
    ts = []
    for i in range(warm + calls):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        if i >= warm:
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm-ticks", type=int, default=50, help="bot ticks played before timing (a live world)")
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--warm", type=int, default=5, help="untimed calls per phase after compile")
    ap.add_argument("--no-prefix", action="store_true", help="skip the prefix-delta column (faster)")
    args = ap.parse_args()

    import jax
    from game.state import next_key
    from game.step import step
    from bots.ro_policies import ro_bot_inputs
    from bots.runner import make_world_batch, run_games

    worlds = make_world_batch(seed=args.seed, batch=args.batch)
    if args.warm_ticks:
        worlds, _ = run_games(worlds, args.warm_ticks)
    keys, worlds = jax.vmap(next_key)(worlds)
    inputs = jax.vmap(ro_bot_inputs)(worlds, keys)
    jax.block_until_ready((worlds, inputs))
    world1 = jax.tree_util.tree_map(lambda a: a[0], worlds)
    inputs1 = jax.tree_util.tree_map(lambda a: a[0], inputs)
    phases = _phases()
    B = args.batch

    rows = []
    # Baseline: a jit that returns one small leaf (dispatch + vmap overhead).
    ident = jax.jit(jax.vmap(lambda w, i: w["tick"]))
    base_p50, _ = _time(ident, (worlds, inputs), args.calls, args.warm)

    for name, fn in phases:
        paths = _changed_paths(fn, world1, inputs1)
        if paths:
            alone = jax.jit(jax.vmap(lambda w, i, fn=fn, paths=paths: _pick(fn(w, i), paths)))
            p50, p95 = _time(alone, (worlds, inputs), args.calls, args.warm)
        else:
            p50, p95 = base_p50, base_p50
        rows.append([name, p50, p95, len(paths)])

    prefix = []
    if not args.no_prefix:
        prev = 0.0
        for k in range(1, len(phases) + 1):
            def trunc(w, i, k=k):
                for _, fn in phases[:k]:
                    w = fn(w, i)
                return w
            p50, _ = _time(jax.jit(jax.vmap(trunc)), (worlds, inputs), args.calls, args.warm)
            prefix.append(p50 - prev)
            prev = p50
    step_p50, step_p95 = _time(jax.jit(jax.vmap(step)), (worlds, inputs), args.calls, args.warm)

    print(f"platform={jax.default_backend()} batch={B} warm_ticks={args.warm_ticks} calls={args.calls} "
          f"dispatch baseline {base_p50:.2f} ms")
    head = f"{'phase':10s} | {'alone p50':>9s} | {'alone p95':>9s} | {'per world':>9s}"
    head += f" | {'prefix d':>9s}" if prefix else ""
    head += " | leaves"
    print(head)
    sum_alone = 0.0
    for i, (name, p50, p95, n) in enumerate(rows):
        sum_alone += p50
        line = f"{name:10s} | {p50:9.2f} | {p95:9.2f} | {p50 / B:9.3f}"
        line += f" | {prefix[i]:9.2f}" if prefix else ""
        print(line + f" | {n}")
    print(f"{'sum alone':10s} | {sum_alone:9.2f} | {'':9s} | {sum_alone / B:9.3f}")
    print(f"{'step':10s} | {step_p50:9.2f} | {step_p95:9.2f} | {step_p50 / B:9.3f}   ms per world-tick")


if __name__ == "__main__":
    main()
