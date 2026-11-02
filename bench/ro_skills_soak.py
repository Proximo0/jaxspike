"""Soak for the skills phase: bots learn and cast skills for many ticks, checking world and skills invariants.

Uses a local step (game/step.py's order with ro_skills.phase between monster_ai and combat) and the skills bot
policy. Extra invariants: sp in [0, max_sp], cooldown / buffs / cast_left >= 0, casting in [0, N_SKILLS),
learned in [0, MAX_SKILL_LEVEL], no active monster with hp <= 0. Damage split: monster hp removed by the skills
phase vs by combat, measured around each phase in the scan carry (not in the world dict).
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from bench.soak_check import check_invariants  # noqa: E402
from bots.ro_skills_policy import bot_inputs  # noqa: E402
from bots.runner import make_world_batch  # noqa: E402
from game import combat, loot, monsters, movement, ro_skills, survival  # noqa: E402
from game.config import MAX_PLAYERS, MONSTER_HP  # noqa: E402
from game.data.skills import MAX_SKILL_LEVEL, N_JOBS, N_SKILLS  # noqa: E402
from game.state import next_key  # noqa: E402

START_SKILL_POINTS = 20


def _removed(before, after, active):
    """Monster hp removed between two hp arrays, over monsters active before."""
    return jnp.sum(jnp.where(active, jnp.maximum(before - after, 0), 0)).astype(jnp.int32)


def step(world, inputs):
    """game.step.step with the skills phase between monster_ai and combat.

    Returns (world, skill_hp_removed, auto_hp_removed) for this tick.
    """
    world = {**world, "monsters": {**world["monsters"], "died": jnp.zeros_like(world["monsters"]["died"])}}
    world = movement.move_players(world, inputs)
    world = monsters.monster_ai(world)
    m0 = world["monsters"]
    world = ro_skills.phase(world, inputs, derived=ro_skills.fallback_derived(world))
    m1 = world["monsters"]
    world = combat.combat(world, inputs)
    m2 = world["monsters"]
    skill_rm = _removed(m0["hp"], m1["hp"], m0["active"])
    auto_rm = _removed(m1["hp"], m2["hp"], m1["active"])
    world = loot.loot(world)
    world = survival.survival(world, inputs)
    return {**world, "tick": world["tick"] + 1}, skill_rm, auto_rm


def add_skills(worlds):
    """Add the batched skills block and a stub stats block to a world batch."""
    batch = worlds["tick"].shape[0]
    skills = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape), ro_skills.make_state())
    stats = {
        "job": jnp.broadcast_to(jnp.arange(MAX_PLAYERS, dtype=jnp.int32) % N_JOBS, (batch, MAX_PLAYERS)),
        "skill_points": jnp.full((batch, MAX_PLAYERS), START_SKILL_POINTS, jnp.int32),
    }
    return {**worlds, "skills": skills, "stats": stats}


def run_games(worlds, ticks):
    """Play `ticks` ticks with the skills bot; return (worlds, stats incl. skill / auto monster hp removed)."""

    def tick(carry, _):
        ws, skill_acc, auto_acc = carry
        keys, ws = jax.vmap(next_key)(ws)
        inputs = jax.vmap(bot_inputs)(ws, keys)
        ws, s_rm, a_rm = jax.vmap(step)(ws, inputs)
        return (ws, skill_acc + s_rm, auto_acc + a_rm), None

    zero = jnp.zeros(worlds["tick"].shape[0], jnp.int32)
    (final, skill_rm, auto_rm), _ = jax.lax.scan(tick, (worlds, zero, zero), None, length=ticks)
    p = final["players"]
    stats = {
        "kills": p["kills"].sum(axis=1).astype(jnp.int32),
        "deaths": p["deaths"].sum(axis=1).astype(jnp.int32),
        "casts": final["skills"]["casts"].sum(axis=1),
        "skill_dmg": final["skills"]["skill_dmg"].sum(axis=1),
        "skill_hp_removed": skill_rm,
        "auto_hp_removed": auto_rm,
    }
    return final, stats


def check_skills_invariants(worlds):
    """Return violation strings for the skills block and monsters across the batch."""
    w = jax.device_get(worlds)
    sk, m = w["skills"], w["monsters"]
    d = jax.device_get(jax.vmap(ro_skills.fallback_derived)(worlds))
    problems = []
    for b in range(np.asarray(m["hp"]).shape[0]):
        sp = np.asarray(sk["sp"][b])
        if ((sp < 0) | (sp > np.asarray(d["max_sp"][b]))).any():
            problems.append(f"world {b}: sp out of [0, max_sp]")
        for name in ("cooldown", "buffs", "cast_left"):
            if (np.asarray(sk[name][b]) < 0).any():
                problems.append(f"world {b}: {name} negative")
        c = np.asarray(sk["casting"][b])
        if ((c < 0) | (c >= N_SKILLS)).any():
            problems.append(f"world {b}: casting out of range")
        lr = np.asarray(sk["learned"][b])
        if ((lr < 0) | (lr > MAX_SKILL_LEVEL)).any():
            problems.append(f"world {b}: learned out of range")
        if (np.asarray(m["active"][b]) & (np.asarray(m["hp"][b]) <= 0)).any():
            problems.append(f"world {b}: active monster with hp <= 0")
    return problems


def main(batch=4, chunks=5, ticks_per_chunk=100, seed=0):
    print("platform:", jax.devices()[0].platform, flush=True)
    worlds = add_skills(make_world_batch(seed, batch))
    kinds = np.asarray(worlds["monsters"]["kind"])[np.asarray(worlds["monsters"]["active"])]
    print(f"monsters per world by kind: {np.bincount(kinds, minlength=len(MONSTER_HP)) / batch}", flush=True)
    n_bots = int(np.asarray(worlds["players"]["active"]).sum())  # across the whole batch
    total_problems, total_ticks = [], 0
    skill_rm = auto_rm = skill_dmg = casts = 0
    t0 = time.perf_counter()
    run = jax.jit(run_games, static_argnums=1)
    for c in range(chunks):
        worlds, stats = run(worlds, ticks_per_chunk)
        jax.block_until_ready(worlds)
        probs = check_invariants(worlds) + check_skills_invariants(worlds)
        total_problems += probs
        total_ticks += ticks_per_chunk
        skill_rm += int(jnp.sum(stats["skill_hp_removed"]))
        auto_rm += int(jnp.sum(stats["auto_hp_removed"]))
        # kills / deaths / casts / skill_dmg are cumulative counters in the world state
        skill_dmg = int(jnp.sum(stats["skill_dmg"]))
        casts = int(jnp.sum(stats["casts"]))
        print(
            f"chunk {c + 1:2d} (tick {total_ticks:5d}): kills={int(jnp.sum(stats['kills']))} "
            f"deaths={int(jnp.sum(stats['deaths']))} casts_resolved={casts} skill_damage={skill_dmg} "
            f"violations={len(probs)}",
            flush=True,
        )
    dt = time.perf_counter() - t0
    print(f"total {total_ticks} ticks x {batch} worlds in {dt:.1f}s")
    per_bot = casts / max(n_bots, 1) / total_ticks * 1000
    print(f"casts per bot per 1000 ticks: {per_bot:.2f}")
    print(f"monster hp removed: skills={skill_rm} auto={auto_rm} (measured around each phase)")
    print(f"skill damage share: {skill_rm / max(skill_rm + auto_rm, 1):.3f} (raw skill_dmg counter {skill_dmg})")
    if total_problems:
        print("FIRST VIOLATIONS:")
        for p_ in total_problems[:15]:
            print("  ", p_)
    print("invariants: all held" if not total_problems else f"invariants: {len(total_problems)} violations")
    print("SOAK PASS" if not total_problems else "SOAK FAIL")
    return not total_problems


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=100, help="ticks per chunk")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.exit(0 if main(batch=a.batch, chunks=a.chunks, ticks_per_chunk=a.ticks, seed=a.seed) else 1)
