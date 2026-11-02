"""M4 resource nodes and crafting stations: placement, tile lookup, walking-distance fields.

Pure JAX, fixed shapes. See docs/craft_brief.md (Worker B). Tables come from game.data.craft.
"""
import jax
import jax.numpy as jnp
import numpy as np

from game.config import DIR_DELTAS, MAP, T_GRASS, T_ROCK, T_TREE, T_WATER
from game.data.craft import (MAX_NODES, N_GATHER_KINDS, N_NODE_KINDS, NODE_COAL, NODE_COPPER, NODE_COUNT, NODE_FIRE, NODE_FISH,
                             NODE_FURNACE, NODE_HERB, NODE_IRON, NODE_SLOT_START, NODE_TREE)
from game.map_gen import DIST_UNREACHABLE
from game.world_layout import TEMPLE_TILES, town_centers
from game.data.craft import STATION_GAP

_NT_TILES = MAP * MAP


def empty_nodes():
    """Return the zero `nodes` dict (nothing placed)."""
    return {
        "pos": jnp.zeros((MAX_NODES, 2), jnp.int32),
        "kind": jnp.zeros((MAX_NODES,), jnp.int32),
        "placed": jnp.zeros((MAX_NODES,), jnp.bool_),
        "active": jnp.zeros((MAX_NODES,), jnp.bool_),
        "respawn_in": jnp.zeros((MAX_NODES,), jnp.int32),
    }


def node_at(nodes):
    """Return int32 [MAP*MAP]: lowest placed node index per tile (x * MAP + y), MAX_NODES where none."""
    idx = jnp.arange(MAX_NODES, dtype=jnp.int32)
    tile = nodes["pos"][:, 0] * MAP + nodes["pos"][:, 1]
    tile = jnp.where(nodes["placed"], tile, _NT_TILES)  # out of range: dropped
    return jnp.full((_NT_TILES,), MAX_NODES, jnp.int32).at[tile].min(idx, mode="drop")


def nodes_near(tile_nodes, pos):
    """Return int32 [N, 5]: node index (or MAX_NODES) on each pos's own tile and its 4 neighbours (clipped)."""
    d = jnp.asarray(DIR_DELTAS, jnp.int32)  # [5, 2], row 0 is (0, 0)
    p = jnp.clip(pos[:, None, :] + d[None], 0, MAP - 1)
    return tile_nodes[p[..., 0] * MAP + p[..., 1]]


def station_tiles():
    """Return int32 [NT, 2, 2]: per town, the furnace tile and the cooking-fire tile."""
    c = town_centers()
    h = TEMPLE_TILES // 2 + STATION_GAP
    furnace = np.stack([c[:, 0] + h, c[:, 1]], 1)
    fire = np.stack([c[:, 0] - h, c[:, 1]], 1)
    return jnp.asarray(np.stack([furnace, fire], 1), jnp.int32)


def _shift_min(a):
    """Elementwise min over the four axis neighbours of int [..., MAP, MAP], borders padded with the array max."""
    pad = jnp.iinfo(a.dtype).max
    p = jnp.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], constant_values=pad)
    return jnp.minimum(jnp.minimum(p[..., 2:, 1:-1], p[..., :-2, 1:-1]),
                       jnp.minimum(p[..., 1:-1, 2:], p[..., 1:-1, :-2]))


def _near(mask):
    """bool [MAP, MAP]: some 4-neighbour of the tile is in mask."""
    return _shift_min((~mask).astype(jnp.int8)) == 0


def place_nodes(world, key):
    """Fill world["craft"]["nodes"] from terrain, walls and pz. Other craft keys are left untouched."""
    terrain, walk = world["terrain"], (world["walls"] == 0) & ~world["pz"]
    ore = walk & _near(terrain == T_ROCK)
    fish = walk & _near(terrain == T_WATER) & ~ore
    tree = walk & _near(terrain == T_TREE) & ~ore & ~fish
    herb = walk & (terrain == T_GRASS) & ~ore & ~fish & ~tree

    order = jnp.argsort(jax.random.uniform(key, (_NT_TILES,))).astype(jnp.int32)  # tiles in random order
    ore_r = jnp.cumsum(ore.reshape(-1)[order]) - 1
    # (kind, eligible flag in sorted order, rank within that kind's set)
    ore_start = {NODE_COPPER: 0, NODE_IRON: NODE_COUNT[NODE_COPPER], NODE_COAL: NODE_COUNT[NODE_COPPER] + NODE_COUNT[NODE_IRON]}
    specs = [(k, ore.reshape(-1)[order], ore_r - s) for k, s in ore_start.items()]
    for k, m in ((NODE_FISH, fish), (NODE_HERB, herb), (NODE_TREE, tree)):
        e = m.reshape(-1)[order]
        specs.append((k, e, jnp.cumsum(e) - 1))

    tile = jnp.full((MAX_NODES,), -1, jnp.int32)
    kind = jnp.zeros((MAX_NODES,), jnp.int32)
    for k, e, r in specs:
        ok = e & (r >= 0) & (r < NODE_COUNT[k])
        slot = jnp.where(ok, NODE_SLOT_START[k] + r, MAX_NODES)
        tile = tile.at[slot].set(order, mode="drop")
        kind = kind.at[slot].set(k, mode="drop")
    placed = tile >= 0
    pos = jnp.stack([jnp.maximum(tile, 0) // MAP, jnp.maximum(tile, 0) % MAP], 1)

    st = station_tiles()
    for k, col in ((NODE_FURNACE, 0), (NODE_FIRE, 1)):
        n = st.shape[0]
        sl = NODE_SLOT_START[k] + jnp.arange(n)
        pos = pos.at[sl].set(st[:, col])
        kind = kind.at[sl].set(k)
        placed = placed.at[sl].set(True)

    nodes = {**world["craft"]["nodes"], "pos": pos.astype(jnp.int32), "kind": kind, "placed": placed,
             "active": placed, "respawn_in": jnp.zeros((MAX_NODES,), jnp.int32)}
    return {**world, "craft": {**world["craft"], "nodes": nodes}}


def node_fields(world):
    """Return int16 [N_NODE_KINDS, MAP, MAP]: walking distance to the nearest placed node of each kind.

    Relaxation as game.map_gen.distance_fields: d = min(d, 1 + min(4-neighbours)) over walkable tiles until stable.
    Gather kinds (0..N_GATHER_KINDS-1) are DIST_UNREACHABLE on pz tiles; station kinds pass through towns.
    """
    nodes = world["craft"]["nodes"]
    # Gather-kind fields treat town (pz) tiles as blocked so they route around towns: a bot must never enter pz to
    # follow one (it would jam the gate). Station fields keep pz walkable, since stations stand inside towns.
    gather_kind = jnp.arange(N_NODE_KINDS) < N_GATHER_KINDS
    free = (world["walls"] == 0)[None] & ~(world["pz"][None] & gather_kind[:, None, None])
    tile = jnp.where(nodes["placed"], nodes["kind"] * _NT_TILES + nodes["pos"][:, 0] * MAP + nodes["pos"][:, 1],
                     N_NODE_KINDS * _NT_TILES)
    src = jnp.zeros((N_NODE_KINDS * _NT_TILES,), jnp.bool_).at[tile].set(True, mode="drop")
    d0 = jnp.where(src.reshape(N_NODE_KINDS, MAP, MAP), 0, DIST_UNREACHABLE).astype(jnp.int16)

    def relax(d):
        return jnp.where(free, jnp.minimum(d, jnp.minimum(_shift_min(d), DIST_UNREACHABLE - 1) + 1),
                         jnp.int16(DIST_UNREACHABLE))

    def body(carry):
        d, _ = carry
        nd = relax(d)
        return nd, jnp.any(nd != d)

    d, _ = jax.lax.while_loop(lambda c: c[1], body, (d0, jnp.bool_(True)))
    return d
