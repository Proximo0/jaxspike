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
from game import world_layout as wl
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
    """Return int32 [N, 9]: node index (or MAX_NODES) on each pos's own tile and its 8 neighbours (clipped)."""
    d = jnp.asarray(DIR_DELTAS, jnp.int32)  # [9, 2], row 0 is (0, 0): RO talk range is the 8 neighbours
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


RESERVED_LINE_BAND = 1  # tiles either side of a zone split line that never get a node


def reserved_tiles():
    """bool [MAP, MAP] (numpy): tiles no resource node may take.

    The zone split lines are T_ROCK, so without this every tile beside them counted as an ore deposit and ore nodes
    lined the whole grid (round-2 defect 9). Reserved: the split lines with RESERVED_LINE_BAND tiles either side
    (outside town and dungeon boxes, where no line is drawn) and every portal lane (the road across a line from a
    portal's warp to its station, ROAD_HALF_WIDTH wide), so the lanes stay clear for walkers. Static per layout.
    """
    xs, ys = np.meshgrid(np.arange(MAP), np.arange(MAP), indexing="ij")
    band = np.zeros((MAP, MAP), bool)
    for v in wl.zone_line_tiles():
        band |= (np.abs(xs - v) <= RESERVED_LINE_BAND) | (np.abs(ys - v) <= RESERVED_LINE_BAND)
    keep = np.zeros((MAP, MAP), bool)
    for x0, y0, x1, y1 in wl.zone_boxes()[[z[1] != wl.ZONE_FIELD for z in wl.ZONES]].tolist():
        keep |= (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    out = band & ~keep
    hw = wl.ROAD_HALF_WIDTH
    for legs in wl.zone_portals().tolist():        # [2 legs, (warp, station), (x, y)]
        pts = np.array(legs).reshape(-1, 2)
        x0, y0 = pts.min(0) - hw
        x1, y1 = pts.max(0) + hw
        out |= (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return out


def place_nodes(world, key):
    """Fill world["craft"]["nodes"] from terrain, walls and pz (minus reserved_tiles). Other craft keys are untouched."""
    terrain, walk = world["terrain"], (world["walls"] == 0) & ~world["pz"] & ~jnp.asarray(reserved_tiles())
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

    # One kind at a time (lax.map over the kind axis, as map_gen.distance_fields_ro does over towns): the loop's
    # temporaries are [MAP, MAP] instead of [N_NODE_KINDS, MAP, MAP] and the result is identical (kinds are
    # independent). At MAP 2048 this is the bigger half of a world build (docs/world_scale.md).
    def relax(d, free_k):
        return jnp.where(free_k, jnp.minimum(d, jnp.minimum(_shift_min(d), DIST_UNREACHABLE - 1) + 1),
                         jnp.int16(DIST_UNREACHABLE))

    def one_kind(args):
        d, free_k = args

        def body(carry):
            d, _ = carry
            nd = relax(d, free_k)
            return nd, jnp.any(nd != d)

        d, _ = jax.lax.while_loop(lambda c: c[1], body, (d, jnp.bool_(True)))
        return d

    return jax.lax.map(one_kind, (d0, free))
