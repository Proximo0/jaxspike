"""Batch bot runner: build populated worlds and play them with scan + vmap."""
import jax
import jax.numpy as jnp

from game.config import (
    ITEM_NONE,
    MAP,
    MAX_MONSTERS,
    MAX_PLAYERS,
    MONSTER_HP,
    PLAYER_MAX_HP,
)
from game.state import make_empty_world, next_key

POP_PLAYERS = min(128, MAX_PLAYERS)
POP_MONSTERS = min(512, MAX_MONSTERS)


def _distinct_walkable(walls, key, n):
    """Pick n distinct open tiles (x, y) at random, as int32 [n, 2].

    Ranks every tile by a random score and takes the lowest n open ones. Tiles are
    never shared, so no two entities spawn on the same tile. Memory is one value
    per tile, which avoids the per-spawn tables that ran out of GPU memory at batch 16.
    """
    scores = jax.random.uniform(key, (MAP * MAP,))
    scores = jnp.where(walls.reshape(-1) == 0, scores, 2.0)
    idx = jnp.argsort(scores)[:n]
    return jnp.stack([idx // MAP, idx % MAP], axis=1).astype(jnp.int32)


def spawn_population(world, key):
    """Activate the first 128 players and 512 monsters on distinct random walkable tiles."""
    kt, kk = jax.random.split(key)
    walls = world["walls"]
    tiles = _distinct_walkable(walls, kt, POP_PLAYERS + POP_MONSTERS)
    ppos = tiles[:POP_PLAYERS]
    mpos = tiles[POP_PLAYERS:]
    kind = jax.random.randint(kk, (POP_MONSTERS,), 0, 2, dtype=jnp.int32)
    hp = jnp.asarray(MONSTER_HP, jnp.int32)[kind]

    p, m = world["players"], world["monsters"]
    players = {
        **p,
        "pos": p["pos"].at[:POP_PLAYERS].set(ppos),
        "spawn": p["spawn"].at[:POP_PLAYERS].set(ppos),
        "hp": p["hp"].at[:POP_PLAYERS].set(PLAYER_MAX_HP),
        "active": p["active"].at[:POP_PLAYERS].set(True),
    }
    monsters = {
        **m,
        "pos": m["pos"].at[:POP_MONSTERS].set(mpos),
        "spawn": m["spawn"].at[:POP_MONSTERS].set(mpos),
        "kind": m["kind"].at[:POP_MONSTERS].set(kind),
        "hp": m["hp"].at[:POP_MONSTERS].set(hp),
        "active": m["active"].at[:POP_MONSTERS].set(True),
    }
    return {**world, "players": players, "monsters": monsters}


def make_world_batch(seed, batch):
    """Build `batch` populated worlds, every field batched on axis 0."""
    from game import map_gen

    keys = jax.random.split(jax.random.PRNGKey(seed), batch)

    def one(key):
        k_world, k_walls, k_spawn = jax.random.split(key, 3)
        world = make_empty_world(k_world)
        world = {**world, "walls": map_gen.make_walls(k_walls)}
        return spawn_population(world, k_spawn)

    return jax.vmap(one)(keys)


def _stats(worlds):
    p = worlds["players"]
    return {
        "kills": p["kills"].sum(axis=1).astype(jnp.int32),
        "deaths": p["deaths"].sum(axis=1).astype(jnp.int32),
        "alive": (p["active"] & ~p["dead"]).sum(axis=1).astype(jnp.int32),
        "items": (p["inv"] != ITEM_NONE).sum(axis=(1, 2)).astype(jnp.int32),
    }


def run_games(worlds, ticks):
    """Play `ticks` ticks of every world with bots; return (worlds, stats)."""
    from bots.policies import bot_inputs
    from game.step import step

    def tick(ws, _):
        keys, ws = jax.vmap(next_key)(ws)
        inputs = jax.vmap(bot_inputs)(ws, keys)
        return jax.vmap(step)(ws, inputs), None

    final, _ = jax.lax.scan(tick, worlds, None, length=ticks)
    return final, _stats(final)
