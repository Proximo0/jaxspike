"""Timing and correctness check for the tile-world spike.

Runs a fixed number of ticks with fixed inputs from a fixed seed, then prints
a checksum of the final state. Run once on CPU and once on GPU; matching
checksums mean the two backends agree on the rules.
"""
import sys
import time

import jax
import jax.numpy as jnp

from tibia_spike import MAX_PLAYERS, batched_step, make_world, step


def checksum(world):
    p, m = world["players"], world["monsters"]
    return (
        int(jnp.sum(p["pos"]) + jnp.sum(p["hp"]) + jnp.sum(p["active"]))
        + int(jnp.sum(m["pos"]) + jnp.sum(m["hp"]) + jnp.sum(m["active"]))
    )


def run_single(ticks=50):
    world = make_world(jax.random.PRNGKey(0))
    for i in range(ticks):
        inputs = (jnp.arange(MAX_PLAYERS) + i) % 5
        world = step(world, inputs)
    jax.block_until_ready(world)
    print(f"single world checksum after {ticks} ticks: {checksum(world)}")


def run_batch(batch, ticks=20):
    keys = jax.random.split(jax.random.PRNGKey(1), batch)
    worlds = jax.vmap(make_world)(keys)
    base = jnp.arange(MAX_PLAYERS)[None, :] + jnp.arange(batch)[:, None]
    worlds = batched_step(worlds, base % 5)  # warm-up / compile
    jax.block_until_ready(worlds)
    t0 = time.perf_counter()
    for i in range(ticks):
        worlds = batched_step(worlds, (base + i) % 5)
    jax.block_until_ready(worlds)
    dt = (time.perf_counter() - t0) / ticks
    print(
        f"batch {batch:4d}: {dt*1000:8.2f} ms/tick total, "
        f"{dt*1000/batch:.4f} ms/world, devices={jax.devices()[0].platform}"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("platform:", jax.devices()[0].platform)
    if mode in ("all", "single"):
        run_single()
    if mode in ("all", "batch"):
        for b in (64, 128, 256):
            run_batch(b)
