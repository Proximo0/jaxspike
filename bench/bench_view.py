"""Time one jitted tick: step + entity_table + changed_mask + view_update, for 512 / 64 / 512 valid viewers."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

CALLS = 50
WARM = 10
VIEWERS = (512, 64, 1)


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from game import config
    from game.state import next_key
    from game.step import step
    from game.view import changed_mask, entity_table, view_update
    from bots.policies import bot_inputs
    from bots.runner import make_world_batch

    platform = jax.default_backend()
    world0 = jax.tree_util.tree_map(lambda a: a[0], make_world_batch(seed=0, batch=1))

    @jax.jit
    def tick(world, prev, known, vs):
        key, world = next_key(world)
        world = step(world, bot_inputs(world, key))
        now = entity_table(world)
        out = view_update(now, changed_mask(prev, now), vs, known)
        return world, now, out

    rows = []
    for nv in sorted(set(VIEWERS), reverse=True):
        vs = jnp.full((config.MAX_VIEWERS,), -1, jnp.int32).at[:nv].set(jnp.arange(nv, dtype=jnp.int32))
        world, prev = world0, entity_table(world0)
        known = jnp.zeros((config.MAX_VIEWERS, config.N_ENTITIES), bool)
        ts = []
        for i in range(WARM + CALLS):
            t0 = time.perf_counter()
            world, prev, (s, g, known) = tick(world, prev, known, vs)
            jax.block_until_ready((s, g, known))
            if i >= WARM:
                ts.append((time.perf_counter() - t0) * 1000)
        rows.append((nv, np.percentile(ts, 50), np.percentile(ts, 95), min(ts)))
    print(f"platform={platform} E={config.N_ENTITIES} VIEW_K={config.VIEW_K} calls={CALLS}")
    print("viewers |  p50 ms |  p95 ms |  min ms")
    for nv, p50, p95, mn in rows:
        print(f"{nv:7d} | {p50:7.2f} | {p95:7.2f} | {mn:7.2f}")


if __name__ == "__main__":
    main()
