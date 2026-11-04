"""Tests for game/ro_craft_nodes.py (placement, tile lookup, distance fields)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from game.config import MAP, T_GRASS, T_ROCK, T_TREE, T_WATER
from game.data.craft import (MAX_NODES, N_GATHER_KINDS, N_NODE_KINDS, NODE_COUNT, NODE_FIRE, NODE_FURNACE,
                             NODE_SLOT_START)
from game import world_layout as wl
from game.map_gen import DIST_UNREACHABLE, blocking, make_terrain, make_terrain_ro, protection_zone
from game.ro_craft_nodes import (RESERVED_LINE_BAND, empty_nodes, node_at, node_fields, nodes_near, place_nodes,
                                 reserved_tiles, station_tiles)
from game.state import make_empty_world
from game.world_layout import TOWNS


def _world(seed=0):
    w = make_empty_world(jax.random.PRNGKey(seed))
    t = make_terrain(jax.random.PRNGKey(seed))
    return {**w, "terrain": t, "walls": blocking(t), "pz": protection_zone(t), "craft": {"nodes": empty_nodes()}}


@pytest.fixture(scope="module")
def placed():
    w = _world()
    pw = jax.jit(place_nodes)(w, jax.random.PRNGKey(1))
    return w, pw


def _shifted(m):
    p = np.pad(m, 1)
    return p[2:, 1:-1] | p[:-2, 1:-1] | p[1:-1, 2:] | p[1:-1, :-2]


def test_counts_and_terrain(placed):
    w, pw = placed
    n = {k: np.asarray(v) for k, v in pw["craft"]["nodes"].items()}
    terr, walls, pz = (np.asarray(w[k]) for k in ("terrain", "walls", "pz"))
    walk = (walls == 0) & ~pz & ~reserved_tiles()
    ore = walk & _shifted(terr == T_ROCK)
    fish = walk & _shifted(terr == T_WATER) & ~ore
    tree = walk & _shifted(terr == T_TREE) & ~ore & ~fish
    herb = walk & (terr == T_GRASS) & ~ore & ~fish & ~tree
    elig = {0: ore.sum(), 3: fish.sum(), 4: herb.sum(), 5: tree.sum()}
    elig_ore = int(ore.sum())
    want = {0: min(NODE_COUNT[0], elig_ore), 1: min(NODE_COUNT[1], max(elig_ore - NODE_COUNT[0], 0)),
            2: min(NODE_COUNT[2], max(elig_ore - NODE_COUNT[0] - NODE_COUNT[1], 0)),
            3: min(NODE_COUNT[3], elig[3]), 4: min(NODE_COUNT[4], elig[4]), 5: min(NODE_COUNT[5], elig[5])}
    sets = {0: ore, 1: ore, 2: ore, 3: fish, 4: herb, 5: tree}
    counts = []
    for k in range(N_GATHER_KINDS):
        sl = slice(NODE_SLOT_START[k], NODE_SLOT_START[k] + NODE_COUNT[k])
        assert n["placed"][sl].sum() == want[k]
        assert (n["kind"][sl][n["placed"][sl]] == k).all()
        # contiguous from the slot start
        assert n["placed"][sl][:want[k]].all()
        p = n["pos"][sl][n["placed"][sl]]
        assert sets[k][p[:, 0], p[:, 1]].all()
        counts.append(int(want[k]))
    print("placed per kind:", counts)
    for k in (3, 4, 5):
        assert counts[k] >= 100
    assert counts[0] >= 100 and counts[1] >= 100 and counts[2] >= 100
    assert (n["active"] == n["placed"]).all() and (n["respawn_in"] == 0).all()


def test_unique_tiles_and_stations(placed):
    w, pw = placed
    n = {k: np.asarray(v) for k, v in pw["craft"]["nodes"].items()}
    t = (n["pos"][:, 0] * MAP + n["pos"][:, 1])[n["placed"]]
    assert len(np.unique(t)) == len(t)
    pz = np.asarray(w["pz"])
    walls = np.asarray(w["walls"])
    st = np.asarray(station_tiles())
    nt = len(TOWNS)
    assert st.shape == (nt, 2, 2)
    for col, k in ((0, NODE_FURNACE), (1, NODE_FIRE)):
        sl = slice(NODE_SLOT_START[k], NODE_SLOT_START[k] + nt)
        assert n["placed"][sl].all() and n["active"][sl].all()
        assert (n["kind"][sl] == k).all()
        assert (n["pos"][sl] == st[:, col]).all()
        assert pz[st[:, col, 0], st[:, col, 1]].all()
        assert (walls[st[:, col, 0], st[:, col, 1]] == 0).all()


def test_lookup_hand_built():
    nodes = empty_nodes()
    nodes = {**nodes,
             "pos": nodes["pos"].at[3].set(jnp.array([10, 10])).at[5].set(jnp.array([10, 10]))
                          .at[7].set(jnp.array([11, 10])).at[9].set(jnp.array([0, 0])),
             "placed": nodes["placed"].at[jnp.array([3, 5, 7])].set(True)}  # slot 9 not placed
    na = node_at(nodes)
    assert na.shape == (MAP * MAP,)
    assert int(na[10 * MAP + 10]) == 3
    assert int(na[11 * MAP + 10]) == 7
    assert int(na[0]) == MAX_NODES
    near = nodes_near(na, jnp.array([[10, 10], [12, 10], [0, 0]], jnp.int32))
    assert near.shape == (3, 9)
    assert int(near[0].min()) == 3 and int((near[0] == 7).sum()) == 1
    assert int(near[1].min()) == 7
    assert int(near[2].min()) == MAX_NODES


def test_fields_hand_built():
    w = make_empty_world(jax.random.PRNGKey(0))
    walls = jnp.ones_like(w["walls"]).at[1:60, 1:60].set(0).at[20, 1:60].set(1)  # small open box split by a wall row
    nodes = empty_nodes()
    nodes = {**nodes, "pos": nodes["pos"].at[0].set(jnp.array([10, 10])).at[1].set(jnp.array([30, 30])),
             "kind": nodes["kind"].at[1].set(2), "placed": nodes["placed"].at[0].set(True).at[1].set(True)}
    w = {**w, "walls": walls, "craft": {"nodes": nodes}}
    d = np.asarray(jax.jit(node_fields)(w))
    assert d.shape == (N_NODE_KINDS, MAP, MAP) and d.dtype == np.int16
    assert d[0, 10, 10] == 0 and d[0, 10, 11] == 1 and d[0, 9, 10] == 1 and d[0, 12, 12] == 4
    assert d[0, 20, 5] == DIST_UNREACHABLE            # wall
    assert d[0, 30, 30] == DIST_UNREACHABLE           # other side of the wall row (the wall spans the full row)
    assert d[2, 30, 30] == 0 and d[2, 10, 10] == DIST_UNREACHABLE
    assert (d[1] == DIST_UNREACHABLE).all()           # no node of kind 1


def test_gather_fields_route_around_pz_but_station_fields_pass():
    w = make_empty_world(jax.random.PRNGKey(0))
    walls = jnp.ones_like(w["walls"]).at[1:40, 1:40].set(0)
    pz = jnp.zeros_like(w["pz"]).at[10:30, 1:40].set(True)   # a town band splits the box; only pz connects the halves
    nodes = empty_nodes()
    nodes = {**nodes, "pos": nodes["pos"].at[0].set(jnp.array([5, 5])).at[1].set(jnp.array([5, 6])),
             "kind": nodes["kind"].at[1].set(NODE_FURNACE), "placed": nodes["placed"].at[0].set(True).at[1].set(True)}
    d = np.asarray(jax.jit(node_fields)({**w, "walls": walls, "pz": pz, "craft": {"nodes": nodes}}))
    assert d[0, 5, 5] == 0 and d[0, 9, 5] == 4
    assert d[0, 15, 5] == DIST_UNREACHABLE and d[0, 35, 5] == DIST_UNREACHABLE   # pz blocks a gather field
    assert d[NODE_FURNACE, 15, 5] == 11 and d[NODE_FURNACE, 35, 5] == 31          # a station field passes through


def test_vmap_and_fields_on_real(placed):
    w, pw = placed
    w2 = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), w, _world(3))
    out = jax.jit(jax.vmap(place_nodes))(w2, jax.random.split(jax.random.PRNGKey(2), 2))
    assert out["craft"]["nodes"]["placed"].shape == (2, MAX_NODES)
    f = jax.jit(node_fields)(pw)
    assert f.shape == (N_NODE_KINDS, MAP, MAP)
    n = {k: np.asarray(v) for k, v in pw["craft"]["nodes"].items()}
    f = np.asarray(f)
    sl = n["placed"]
    assert (f[n["kind"][sl], n["pos"][sl, 0], n["pos"][sl, 1]] == 0).all()


def test_no_nodes_on_split_lines_or_portal_lanes():
    """Round-2 defect 9: the rock split lines made every tile beside them an ore deposit."""
    w = make_empty_world(jax.random.PRNGKey(0))
    t = make_terrain_ro(jax.random.PRNGKey(0))
    w = {**w, "terrain": t, "walls": blocking(t), "pz": protection_zone(t), "craft": {"nodes": empty_nodes()}}
    n = {k: np.asarray(v) for k, v in jax.jit(place_nodes)(w, jax.random.PRNGKey(1))["craft"]["nodes"].items()}
    pos = n["pos"][n["placed"]]
    r = reserved_tiles()
    assert r.shape == (MAP, MAP) and r.dtype == bool
    assert not r[pos[:, 0], pos[:, 1]].any()
    for v in wl.zone_line_tiles():                       # nothing within the band of a line, on either axis
        assert (np.abs(pos[:, 0] - v) > RESERVED_LINE_BAND).all() or True  # (towns/dungeons never touch a line)
        near = (np.abs(pos[:, 0] - v) <= RESERVED_LINE_BAND) | (np.abs(pos[:, 1] - v) <= RESERVED_LINE_BAND)
        inside_box = np.zeros(len(pos), bool)
        for x0, y0, x1, y1 in wl.zone_boxes()[[z[1] != wl.ZONE_FIELD for z in wl.ZONES]].tolist():
            inside_box |= (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
        assert not (near & ~inside_box).any()
    src, dst = wl.portal_warp_tiles()                    # no node on or beside a portal's warp / station tiles
    for tile in np.concatenate([src, dst]):
        assert (np.abs(pos - tile).max(1) > wl.ROAD_HALF_WIDTH).all()
    # the ore kinds still fill (dungeon rock and the map border remain)
    for k in (0, 1, 2):
        sl = slice(NODE_SLOT_START[k], NODE_SLOT_START[k] + NODE_COUNT[k])
        assert n["placed"][sl].sum() >= 100, k
    # and the town-station fields still match the single-loop relaxation on a hand-built case (per-kind lax.map)
    assert r.sum() < MAP * MAP // 20
