"""Tibia-style tile world step in JAX: timing and vmapped-bot spike.

World: one floor, MAP x MAP tiles, int8 walkability.
Players: fixed slots, each with pos, hp, active flag, and a move input.
Monsters: fixed slots, each with pos, hp, active flag.
Step: players move from their inputs, monsters chase nearest active
player, monsters adjacent to a player deal damage, and hp regenerates.
"""
import time
from functools import partial

import jax
import jax.numpy as jnp

MAP = 256
MAX_PLAYERS = 512
MAX_MONSTERS = 4096
DIRS = jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32)


def make_world(key):
    k1, k2, k3 = jax.random.split(key, 3)
    walls = (jax.random.uniform(k1, (MAP, MAP)) < 0.15).astype(jnp.int8)
    players = dict(
        pos=jax.random.randint(k2, (MAX_PLAYERS, 2), 0, MAP, dtype=jnp.int32),
        hp=jnp.full((MAX_PLAYERS,), 100, dtype=jnp.int32),
        active=jnp.arange(MAX_PLAYERS) < MAX_PLAYERS // 2,
    )
    monsters = dict(
        pos=jax.random.randint(k3, (MAX_MONSTERS, 2), 0, MAP, dtype=jnp.int32),
        hp=jnp.full((MAX_MONSTERS,), 50, dtype=jnp.int32),
        active=jnp.arange(MAX_MONSTERS) < MAX_MONSTERS * 3 // 4,
    )
    return dict(walls=walls, players=players, monsters=monsters)


@jax.jit
def step(world, inputs):
    walls = world["walls"]
    p, m = world["players"], world["monsters"]

    # Players: move by input direction if target tile is free and in bounds.
    target = jnp.clip(p["pos"] + DIRS[inputs], 0, MAP - 1)
    blocked = walls[target[:, 0], target[:, 1]] == 1
    can_move = p["active"] & ~blocked
    new_ppos = jnp.where(can_move[:, None], target, p["pos"])

    # Monsters: chase the nearest active player (MAX_MONSTERS x MAX_PLAYERS).
    d = jnp.abs(m["pos"][:, None, :] - new_ppos[None, :, :]).sum(-1)
    d = jnp.where(p["active"][None, :], d, 10**6)
    nearest = jnp.argmin(d, axis=1)
    dist = jnp.min(d, axis=1)
    delta = new_ppos[nearest] - m["pos"]
    step_dir = jnp.sign(delta) * (jnp.abs(delta[:, 0:1]) >= jnp.abs(delta[:, 1:2]))
    step_dir = jnp.where(
        jnp.abs(delta[:, 0:1]) >= jnp.abs(delta[:, 1:2]),
        jnp.stack([jnp.sign(delta[:, 0]), jnp.zeros_like(delta[:, 0])], -1),
        jnp.stack([jnp.zeros_like(delta[:, 1]), jnp.sign(delta[:, 1])], -1),
    )
    mtarget = jnp.clip(m["pos"] + step_dir, 0, MAP - 1)
    mblocked = walls[mtarget[:, 0], mtarget[:, 1]] == 1
    adjacent = dist <= 1
    new_mpos = jnp.where((m["active"] & ~mblocked & ~adjacent)[:, None], mtarget, m["pos"])

    # Monsters adjacent to a player hit it: scatter-add damage.
    dmg = jnp.where(m["active"] & adjacent, 3, 0)
    hp_loss = jnp.zeros((MAX_PLAYERS,), jnp.int32).at[nearest].add(dmg)
    new_php = jnp.where(p["active"], jnp.maximum(p["hp"] - hp_loss + 1, 0), p["hp"])
    alive = p["active"] & (new_php > 0)

    # Monsters take damage from adjacent players (one hit each, simple rule).
    hit_by_player = jnp.zeros((MAX_MONSTERS,), jnp.int32)
    new_mhp = m["hp"] - hit_by_player - jnp.where(adjacent & alive[nearest], 1, 0)
    new_mactive = m["active"] & (new_mhp > 0)

    return dict(
        walls=walls,
        players=dict(pos=new_ppos, hp=new_php, active=alive),
        monsters=dict(pos=new_mpos, hp=new_mhp, active=new_mactive),
    )


def time_single(steps=200):
    world = make_world(jax.random.PRNGKey(0))
    inputs = jnp.zeros((MAX_PLAYERS,), jnp.int32)
    world = step(world, inputs)  # compile
    jax.block_until_ready(world)
    t0 = time.perf_counter()
    for i in range(steps):
        inputs = (jnp.arange(MAX_PLAYERS) + i) % 5
        world = step(world, inputs)
    jax.block_until_ready(world)
    dt = (time.perf_counter() - t0) / steps
    print(f"single world: {dt*1000:.2f} ms/tick at {MAX_PLAYERS} player slots, {MAX_MONSTERS} monster slots")


@partial(jax.jit, static_argnames=())
def batched_step(worlds, inputs):
    return jax.vmap(step)(worlds, inputs)


def time_batch(batch=64, steps=50):
    keys = jax.random.split(jax.random.PRNGKey(1), batch)
    worlds = jax.vmap(make_world)(keys)
    inputs = jnp.zeros((batch, MAX_PLAYERS), jnp.int32)
    worlds = batched_step(worlds, inputs)
    jax.block_until_ready(worlds)
    t0 = time.perf_counter()
    for i in range(steps):
        inputs = (jnp.arange(MAX_PLAYERS)[None, :] + i + jnp.arange(batch)[:, None]) % 5
        worlds = batched_step(worlds, inputs)
    jax.block_until_ready(worlds)
    dt = (time.perf_counter() - t0) / steps
    print(f"batch of {batch} worlds: {dt*1000:.2f} ms/tick total, {dt*1000/batch:.3f} ms per world")


if __name__ == "__main__":
    print("devices:", jax.devices())
    time_single()
    time_batch(batch=64)
    time_batch(batch=256, steps=20)
