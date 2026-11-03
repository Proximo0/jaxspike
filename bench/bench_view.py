"""Time one jitted tick: step + entity_table + changed_mask + view_update, for 512 / 64 / 1 valid viewers.

Rows "E=..." with the table cut to the first config.N_ENTITIES ids show the cost without the resource nodes.
The "anim" line (ro-anim) times the 512-viewer tick with and without world["anim"] + ro_anim.phase."""
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
    from bots.ro_policies import ro_bot_inputs as bot_inputs  # the server's policy (all RO input keys)
    from bots.runner import make_world_batch

    platform = jax.default_backend()
    world0 = jax.tree_util.tree_map(lambda a: a[0], make_world_batch(seed=0, batch=1))

    def table(world, e):
        return {k: v[:e] for k, v in entity_table(world).items()}

    rows = []
    e_full = entity_table(world0)["kind"].shape[0]
    for e in sorted({e_full, config.N_ENTITIES}, reverse=True):
        @jax.jit
        def tick(world, prev, known, vs, e=e):
            key, world = next_key(world)
            world = step(world, bot_inputs(world, key))
            now = table(world, e)
            out = view_update(now, changed_mask(prev, now), vs, known)
            return world, now, out

        for nv in sorted(set(VIEWERS), reverse=True):
            vs = jnp.full((config.MAX_VIEWERS,), -1, jnp.int32).at[:nv].set(jnp.arange(nv, dtype=jnp.int32))
            world, prev = world0, table(world0, e)
            known = jnp.zeros((config.MAX_VIEWERS, e), bool)
            ts = []
            for i in range(WARM + CALLS):
                t0 = time.perf_counter()
                world, prev, (s, g, known) = tick(world, prev, known, vs)
                jax.block_until_ready((s, g, known))
                if i >= WARM:
                    ts.append((time.perf_counter() - t0) * 1000)
            rows.append((e, nv, np.percentile(ts, 50), np.percentile(ts, 95), min(ts)))
    print(f"platform={platform} E={e_full} (without nodes {config.N_ENTITIES}) VIEW_K={config.VIEW_K} calls={CALLS}")
    print("      E | viewers |  p50 ms |  p95 ms |  min ms")
    for e, nv, p50, p95, mn in rows:
        print(f"{e:7d} | {nv:7d} | {p50:7.2f} | {p95:7.2f} | {mn:7.2f}")
    _anim_cost(world0, step, next_key, bot_inputs, entity_table, changed_mask, view_update)


def _anim_cost(world0, step, next_key, bot_inputs, entity_table, changed_mask, view_update):
    """p50 ms of the full 512-viewer tick without and with the anim block (step + ro_anim.phase + wire columns)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from game import config, ro_anim

    def make(with_anim):
        @jax.jit
        def tick(world, prev, known, vs):
            key, world = next_key(world)
            world = step(world, bot_inputs(world, key))
            if with_anim:
                world = ro_anim.phase(world)
            now = entity_table(world)
            return world, now, view_update(now, changed_mask(prev, now), vs, known)
        return tick

    vs = jnp.arange(config.MAX_VIEWERS, dtype=jnp.int32)
    p50 = []
    for with_anim in (False, True):
        tick = make(with_anim)
        world = {**world0, "anim": ro_anim.make_state()} if with_anim else world0
        prev = entity_table(world)
        known = jnp.zeros((config.MAX_VIEWERS, prev["kind"].shape[0]), bool)
        ts = []
        for i in range(WARM + CALLS):
            t0 = time.perf_counter()
            world, prev, (s, g, known) = tick(world, prev, known, vs)
            jax.block_until_ready((s, g, known))
            if i >= WARM:
                ts.append((time.perf_counter() - t0) * 1000)
        p50.append(np.percentile(ts, 50))
    print(f"anim | 512 viewers p50 ms without {p50[0]:.2f} | with {p50[1]:.2f} | added {p50[1] - p50[0]:+.2f}")


if __name__ == "__main__":
    main()
