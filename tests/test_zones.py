"""RO zones (ro-world2): zone table and zone_of, split lines and edge portals, town layouts, waybell data."""
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from game import map_gen, world_layout as wl
from game.config import MAP, T_ROAD, T_ROCK, T_TEMPLE, T_WALL, T_WATER
from game.data.terrain import T_WAYBELL, T_SHOP, T_STATION, T_WARP
from game.ro_warps import apply_warps
from game.state import make_empty_world

TER = np.asarray(map_gen.make_terrain_ro(jax.random.PRNGKey(0)))
FREE = np.asarray(map_gen.blocking(jnp.asarray(TER))) == 0
PZ = np.asarray(map_gen.protection_zone(jnp.asarray(TER)))
BOXES = wl.zone_boxes()
XS, YS = np.meshgrid(np.arange(MAP), np.arange(MAP), indexing="ij")
ALL = jnp.asarray(np.stack([XS.ravel(), YS.ravel()], 1), jnp.int32)
ZONE = np.asarray(jax.jit(wl.zone_of)(ALL)).reshape(MAP, MAP)
SRC, DST = wl.warp_tiles()
WARP_TO = {tuple(s): tuple(d) for s, d in zip(SRC.tolist(), DST.tolist())}


def walk(starts, warps=True):
    """Forward BFS over walkable tiles; stepping on a warp src lands on its dst (when warps)."""
    seen = np.zeros((MAP, MAP), bool)
    q = deque()
    for s in starts:
        s = tuple(int(v) for v in s)
        seen[s] = True
        q.append(s)
    while q:
        x, y = q.popleft()
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= n[0] < MAP and 0 <= n[1] < MAP) or not FREE[n] or seen[n]:
                continue
            seen[n] = True
            if warps and n in WARP_TO:
                n = WARP_TO[n]
                if seen[n]:
                    continue
                seen[n] = True
            q.append(n)
    return seen


def in_box(b):
    x0, y0, x1, y1 = b
    return (XS >= x0) & (XS <= x1) & (YS >= y0) & (YS <= y1)


def test_zone_table():
    assert len(set(wl.ZONE_NAMES)) == wl.N_ZONES == BOXES.shape[0]
    kinds = [z[1] for z in wl.ZONES]
    assert kinds.count(wl.ZONE_TOWN) == len(wl.TOWNS) and kinds.count(wl.ZONE_DUNGEON) == len(wl.DUNGEONS)
    assert kinds.count(wl.ZONE_FIELD) == 9
    for name, kind, ref, (lo, hi), music in wl.ZONES:
        assert lo <= hi and music > 0 and wl.ZONE_KINDS[kind]
    for name in ("solmera", "sol_fld01", "sol_fld02", "lun_fld00", "pin_fld01", "kha_fld01",
                 "tid_fld01", "sol_sew01", "pin_dun00"):
        assert name in wl.ZONE_NAMES
    fr = wl.zone_box_fractions()
    assert (fr >= 0).all() and (fr <= 1).all() and (fr[:, 2:] > fr[:, :2]).all()


def test_fields_tile_the_map_and_nested_zones_fit_one_cell():
    field = [i for i, z in enumerate(wl.ZONES) if z[1] == wl.ZONE_FIELD]
    cover = sum(in_box(BOXES[i]).astype(int) for i in field)
    assert (cover == 1).all()
    lines = wl.zone_line_tiles()
    for i, z in enumerate(wl.ZONES):
        if z[1] == wl.ZONE_FIELD:
            continue
        x0, y0, x1, y1 = BOXES[i].tolist()
        for v in lines:   # never crosses a split line
            assert not (x0 <= v <= x1) and not (y0 <= v <= y1)


def test_zone_of_matches_reference():
    ref = np.full((MAP, MAP), -1)
    for i, z in enumerate(wl.ZONES):
        if z[1] == wl.ZONE_FIELD:
            ref[in_box(BOXES[i])] = i
    for kind in (wl.ZONE_TOWN, wl.ZONE_DUNGEON):
        for i, z in enumerate(wl.ZONES):
            if z[1] == kind:
                ref[in_box(BOXES[i])] = i
    assert (ZONE == ref).all()
    assert ZONE.dtype == np.int32
    batch = jnp.stack([ALL[:100], ALL[-100:]])
    out = np.asarray(jax.vmap(wl.zone_of)(batch))
    assert (out == np.stack([ZONE.ravel()[:100], ZONE.ravel()[-100:]])).all()
    c = wl.town_centers()[0]
    assert wl.ZONE_NAMES[int(wl.zone_of(jnp.asarray([c]))[0])] == "solmera"


def test_split_lines_are_rock_outside_towns_and_dungeons():
    keep = np.zeros((MAP, MAP), bool)
    for i, z in enumerate(wl.ZONES):
        if z[1] != wl.ZONE_FIELD:
            keep |= in_box(BOXES[i])
    for v in wl.zone_line_tiles():
        assert (TER[v, :][~keep[v, :]] == T_ROCK).all()
        assert (TER[:, v][~keep[:, v]] == T_ROCK).all()


def test_portals_pair_up_across_lines():
    p = wl.zone_portals()
    assert p.shape[0] >= 8 and len(wl.WARPS) <= wl.N_WARPS
    for (w0, s0), (w1, s1) in p.tolist():
        z = lambda t: int(ZONE[t[0], t[1]])
        assert z(w0) == z(s1) != z(w1) == z(s0)   # each leg crosses into the neighbour zone
        assert wl.ZONES[z(w0)][1] == wl.ZONES[z(w1)][1] == wl.ZONE_FIELD
        for w, s in ((w0, s0), (w1, s1)):
            assert TER[w[0], w[1]] == T_WARP and TER[s[0], s[1]] == T_STATION
            assert sum(abs(a - b) for a, b in zip(w, s)) == wl.PORTAL_STEP_TILES + 1   # straight across
            assert not PZ[w[0], w[1]] and not PZ[s[0], s[1]]
        assert WARP_TO[tuple(w0)] == tuple(s0) and WARP_TO[tuple(w1)] == tuple(s1)


def test_every_zone_reachable_from_solmera_only_through_warps():
    temples = np.asarray(wl.temple_tiles())[:, 0]
    plain = walk([temples[0]], warps=False)
    assert set(np.unique(ZONE[plain]).tolist()) == {wl.ZONE_NAMES.index("solmera"), wl.ZONE_NAMES.index("sol_fld08")}
    seen = walk([temples[0]])
    assert set(np.unique(ZONE[seen]).tolist()) == set(range(wl.N_ZONES))
    assert seen[temples[:, 0], temples[:, 1]].all()
    for arr in (wl.waybell_tiles(), wl.town_station_tiles(), wl.shop_tiles()[0], SRC, DST):
        assert seen[arr[:, 0], arr[:, 1]].all()


def test_apply_warps_moves_across_a_portal():
    (w0, s0), _ = wl.zone_portals()[0].tolist()
    w = make_empty_world(jax.random.PRNGKey(0))
    p = w["players"]
    p = {**p, "pos": p["pos"].at[0].set(jnp.array(w0, jnp.int32)), "active": p["active"].at[0].set(True),
         "dead": p["dead"].at[0].set(False)}
    out = apply_warps({**w, "terrain": jnp.asarray(TER), "players": p})
    new = np.asarray(out["players"]["pos"][0])
    assert tuple(new) == tuple(s0)
    assert int(wl.zone_of(jnp.asarray([new]))[0]) != int(wl.zone_of(jnp.asarray([w0]))[0])


def test_town_layouts():
    tt = np.asarray(wl.temple_tiles()).reshape(-1, 2)
    assert (TER[tt[:, 0], tt[:, 1]] == T_TEMPLE).all()
    shops, town = wl.shop_tiles()
    assert (TER[shops[:, 0], shops[:, 1]] == T_SHOP).all() and PZ[shops[:, 0], shops[:, 1]].all()
    assert np.bincount(town).tolist() == list(wl.SHOP_ROW_TILES)
    kt = wl.waybell_tiles()
    assert (TER[kt[:, 0], kt[:, 1]] == T_WAYBELL).all()
    temples = np.asarray(wl.temple_tiles())[:, 0]
    patterns = []
    for t, (box, (cx, cy)) in enumerate(zip(wl.town_boxes().tolist(), wl.town_centers().tolist())):
        x0, y0, x1, y1 = box
        sh = wl.TOWN_STREET_HALF[t]
        inner = TER[x0 + 1:x1, y0 + 1:y1]
        # cross streets run wall to wall, never blocked
        assert np.isin(TER[cx - sh:cx + sh + 1, y0 + 1:y1], [T_ROAD, T_TEMPLE, T_WAYBELL, T_STATION]).all()
        assert np.isin(TER[x0 + 1:x1, cy - sh:cy + sh + 1], [T_ROAD, T_TEMPLE, T_WAYBELL, T_STATION]).all()
        # every walkable tile inside the walls is reachable from the temple without warps
        seen = walk([temples[t]], warps=False)
        box_free = FREE & in_box(box)
        assert seen[box_free].all()
        assert (~FREE[x0 + 1:x1, y0 + 1:y1]).sum() > 0   # each town has decoration
        patterns.append(inner)
    # Solmera: castle block, four fountain basins, a 5-wide cross
    assert wl.TOWN_STREET_HALF[0] == 2
    for _, x0, y0, x1, y1 in wl.town_fountains():
        assert (TER[x0:x1 + 1, y0:y1 + 1] == T_WATER).all()
    castle = [d for d in wl.town_decor() if d[0] == 0 and d[1] == "wall"][0]
    assert (TER[castle[2]:castle[4] + 1, castle[3]:castle[5] + 1] == T_WALL).all()
    # the four simpler towns are distinct layouts
    small = [p for p in patterns[1:]]
    for i in range(len(small)):
        for j in range(i + 1, len(small)):
            assert small[i].shape != small[j].shape or not np.array_equal(small[i], small[j])


def test_waybell_teleport_and_save_points():
    cost = wl.waybell_teleport_cost()
    nt = len(wl.TOWNS)
    assert cost.shape == (nt, nt) and cost.dtype == np.int32
    assert (np.diag(cost) == wl.WAYBELL_NO_ROUTE).all()
    assert ((cost == wl.WAYBELL_NO_ROUTE) | (cost > 0)).all()
    assert ((cost > 0).sum(1) >= 2).all()          # every waybell offers at least two towns
    assert (cost[0] > 0).sum() == nt - 1            # the capital reaches every town
    dest = wl.waybell_teleport_dest()
    assert (TER[dest[:, 0], dest[:, 1]] == T_STATION).all() and FREE[dest[:, 0], dest[:, 1]].all()
    sp = wl.save_point_tiles()
    assert (TER[sp[:, 0], sp[:, 1]] == T_TEMPLE).all()
