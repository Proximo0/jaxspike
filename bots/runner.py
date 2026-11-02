"""Batch bot runner: build populated worlds and play them with scan + vmap."""
import jax
import jax.numpy as jnp

from game.config import (
    GATE_APRON_TILES,
    ITEM_NONE,
    MAP,
    MAX_MONSTERS,
    MAX_PLAYERS,
    MONSTER_HP,
    MONSTER_KIND_ORC,
    MONSTER_KIND_RAT,
    PLAYER_MAX_HP,
)
from game.state import make_empty_world, next_key
from game.world_layout import RAT_RING, TOWNS, temple_tiles

POP_PLAYERS = min(128, MAX_PLAYERS)
POP_MONSTERS = min(1024, MAX_MONSTERS)


def _distinct_tiles(allowed, key, n):
    """Pick n distinct allowed tiles (x, y) at random, as int32 [n, 2].

    Ranks every tile by a random score and takes the lowest n allowed ones. Tiles are
    never shared, so no two entities spawn on the same tile. Memory is one value
    per tile, which avoids the per-spawn tables that ran out of GPU memory at batch 16.
    """
    scores = jax.random.uniform(key, (MAP * MAP,))
    scores = jnp.where(allowed.reshape(-1), scores, 2.0)
    idx = jnp.argsort(scores)[:n]
    return jnp.stack([idx // MAP, idx % MAP], axis=1).astype(jnp.int32)


def spawn_population(world, key):
    """Activate players on their home temples and monsters on distinct walkable non-pz tiles away from gates.

    Player i gets town i % NT and temple tile i // NT. Monsters within RAT_RING walking distance of a gate are rats, the rest orcs.
    Needs world["dist_town"] filled (make_world_batch does this).
    """
    nt = len(TOWNS)
    i = jnp.arange(POP_PLAYERS)
    ppos = temple_tiles()[i % nt, i // nt]
    gate_dist = world["dist_town"].min(axis=0)  # walking distance to the nearest gate
    mpos = _distinct_tiles((world["walls"] == 0) & ~world["pz"] & (gate_dist > GATE_APRON_TILES), key, POP_MONSTERS)
    near = gate_dist[mpos[:, 0], mpos[:, 1]] < RAT_RING * MAP
    kind = jnp.where(near, MONSTER_KIND_RAT, MONSTER_KIND_ORC).astype(jnp.int32)
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

    def terrain_world(key):
        k_world, k_walls, k_spawn = jax.random.split(key, 3)
        world = make_empty_world(k_world)
        terrain = map_gen.make_terrain(k_walls)
        world = {
            **world,
            "terrain": terrain,
            "walls": map_gen.blocking(terrain),
            "pz": map_gen.protection_zone(terrain),
        }
        return world, k_spawn

    worlds, k_spawn = jax.vmap(terrain_world)(keys)
    # Sequential over worlds: the relaxation loop's temporaries would otherwise be batched too.
    fields = jax.lax.map(map_gen.distance_fields, worlds["terrain"])
    return jax.vmap(spawn_population)({**worlds, "dist_town": fields}, k_spawn)


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
