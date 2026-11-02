"""Time the vmapped bot_inputs + step over batches of worlds."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

TICKS = 20
BATCHES = (1, 64, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["cpu", "gpu"], default=None)
    args = ap.parse_args()
    if args.platform:
        os.environ.setdefault("JAX_PLATFORMS", "cpu" if args.platform == "cpu" else "cuda,cpu")

    import jax
    from game import config
    from game.state import next_key
    from game.step import step
    from bots.policies import bot_inputs
    from bots.runner import make_world_batch

    platform = jax.default_backend()
    print(f"platform={platform} players={config.MAX_PLAYERS} monsters={config.MAX_MONSTERS}", flush=True)

    def tick(world, _):
        key, world = next_key(world)
        return step(world, bot_inputs(world, key)), None

    @jax.jit
    def run(worlds):
        def body(ws, _):
            return jax.vmap(lambda w: tick(w, None)[0])(ws), None
        return jax.lax.scan(body, worlds, None, length=TICKS)[0]

    for b in BATCHES:
        try:
            worlds = make_world_batch(seed=0, batch=b)
            jax.block_until_ready(run(worlds))  # warm-up / compile
            t0 = time.perf_counter()
            jax.block_until_ready(run(worlds))
            ms = (time.perf_counter() - t0) * 1000 / TICKS
            print(f"B={b} {ms:.3f} ms/tick total, {ms / b:.4f} ms/world, {platform}", flush=True)
        except Exception as e:
            print(f"B={b} failed {type(e).__name__}", flush=True)


if __name__ == "__main__":
    main()
