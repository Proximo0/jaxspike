"""Brute-force vs grid step at larger entity counts.

Makes copies of tibia_spike.py and tibia_grid.py with larger slot counts in
a separate folder, then times both steps on the current backend.
"""
import os
import pathlib
import shutil
import sys
import time

SRC = pathlib.Path(__file__).parent
DST = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path.home() / "jaxbig"
PLAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
MONSTERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8192

DST.mkdir(parents=True, exist_ok=True)
for name in ("tibia_spike.py", "tibia_grid.py"):
    text = (SRC / name).read_text()
    text = text.replace("MAX_PLAYERS = 512\n", f"MAX_PLAYERS = {PLAYERS}\n")
    text = text.replace("MAX_MONSTERS = 4096\n", f"MAX_MONSTERS = {MONSTERS}\n")
    (DST / name).write_text(text)
sys.path.insert(0, str(DST))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tibia_grid import step_grid  # noqa: E402
from tibia_spike import MAX_PLAYERS, batched_step, make_world  # noqa: E402


def bench(fn, batch, ticks=10):
    keys = jax.random.split(jax.random.PRNGKey(1), batch)
    worlds = jax.vmap(make_world)(keys)
    base = jnp.arange(MAX_PLAYERS)[None, :] + jnp.arange(batch)[:, None]
    worlds = fn(worlds, base % 5)
    jax.block_until_ready(worlds)
    t0 = time.perf_counter()
    for i in range(ticks):
        worlds = fn(worlds, (base + i) % 5)
    jax.block_until_ready(worlds)
    return (time.perf_counter() - t0) / ticks * 1000


if __name__ == "__main__":
    grid_v = jax.vmap(step_grid)
    print(f"platform={jax.devices()[0].platform} P={MAX_PLAYERS} M={MONSTERS}", flush=True)
    for b in (16, 64):
        try:
            brute = bench(batched_step, b)
            grid = bench(grid_v, b)
            print(f"B={b}: brute {brute:.2f} ms/tick, grid {grid:.2f} ms/tick", flush=True)
        except Exception as e:  # report allocation failures instead of crashing
            print(f"B={b}: failed {type(e).__name__}", flush=True)
