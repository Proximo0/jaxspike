from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from game import world_layout as wl
from game.config import MAP, T_ROCK, T_TEMPLE, T_WALL
from game.data.terrain import T_CAVE, T_KAFRA, T_STATION, T_WARP
from game.map_gen import (DIST_UNREACHABLE, blocking, distance_fields, distance_fields_ro,
                          make_terrain, make_terrain_ro, protection_zone)

SEEDS = (0, 1)
TERRAINS = {s: np.asarray(make_terrain_ro(jax.random.PRNGKey(s))) for s in SEEDS}
OLD = {s: np.asarray(make_terrain(jax.random.PRNGKey(s))) for s in SEEDS}
SRC, DST = wl.warp_tiles()
PADS = wl.dungeon_pads()
BOXES = wl.dungeon_boxes()


def bfs(free, starts):
    seen = np.zeros_like(free)
    q = deque()
    for s in starts:
        s = tuple(s)
        if free[s] and not seen[s]:
            seen[s] = True
            q.append(s)
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < MAP and 0 <= ny < MAP and free[nx, ny] and not seen[nx, ny]:
                seen[nx, ny] = True
                q.append((nx, ny))
    return seen


def bfs_warps(free, starts):
    """bfs where stepping on a warp src lands on its dst."""
    to = {tuple(a): tuple(b) for a, b in zip(SRC.tolist(), DST.tolist())}
    seen = np.zeros_like(free)
    q = deque()
    for s in starts:
        seen[tuple(s)] = True
        q.append(tuple(s))
    while q:
        x, y = q.popleft()
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= n[0] < MAP and 0 <= n[1] < MAP and free[n] and not seen[n]:
                seen[n] = True
                if n in to and not seen[to[n]]:
                    seen[to[n]] = True
                    q.append(to[n])
                elif n not in to:
                    q.append(n)
    return seen


def free_of(t):
    return np.asarray(blocking(jnp.asarray(t))) == 0


def test_warp_and_special_tiles():
    for t in TERRAINS.values():
        assert t.dtype == np.int8
        assert (t[SRC[:, 0], SRC[:, 1]] == T_WARP).all()
        assert (t[DST[:, 0], DST[:, 1]] == T_STATION).all()
        kt, st = wl.kafra_tiles(), wl.town_station_tiles()
        assert (t[kt[:, 0], kt[:, 1]] == T_KAFRA).all()
        assert (t[st[:, 0], st[:, 1]] == T_STATION).all()
        pz = np.asarray(protection_zone(jnp.asarray(t)))
        assert pz[kt[:, 0], kt[:, 1]].all() and pz[st[:, 0], st[:, 1]].all()
        # dungeon-side stations and warps stay outside the pz
        assert not pz[SRC[:, 0], SRC[:, 1]].any() and not pz[DST[:, 0], DST[:, 1]].any()


def test_dungeon_ring_and_interior():
    w = wl.DUNGEON_WALL_TILES
    for t in TERRAINS.values():
        for x0, y0, x1, y1 in BOXES:
            box = t[x0:x1 + 1, y0:y1 + 1]
            ring = np.ones(box.shape, bool)
            ring[w:-w, w:-w] = False
            assert (box[ring] == T_ROCK).all()
            assert np.isin(box[~ring], [T_CAVE, T_ROCK, T_STATION, T_WARP]).all()
            assert (box[~ring] == T_CAVE).any() and (box[~ring] == T_ROCK).any()


def test_interior_connected_and_sealed():
    for t in TERRAINS.values():
        free = free_of(t)
        temples = np.asarray(wl.temple_tiles())[:, 0]
        outside = bfs(free, temples)
        for k, (x0, y0, x1, y1) in enumerate(BOXES):
            _, ins, ext, _ = PADS[k]
            seen = bfs(free, [ins])
            interior = (t[x0:x1 + 1, y0:y1 + 1] == T_CAVE)
            assert seen[x0:x1 + 1, y0:y1 + 1][interior].all()
            assert seen[ext[0], ext[1]]
            # nothing inside the box is reachable from a temple without warps
            assert not outside[x0:x1 + 1, y0:y1 + 1].any()
            assert not (seen & outside).any()


def test_pads_walkable_and_connected_to_town():
    for t in TERRAINS.values():
        free = free_of(t)
        centre = [wl.town_centers()[0]]
        seen = bfs_warps(free, centre)   # the dungeon entrances sit in other zones, past portals
        for ent, ins, ext, out in PADS:
            for p in (ent, ins, ext, out):
                assert free[p[0], p[1]]
            assert seen[ent[0], ent[1]] and seen[out[0], out[1]]


def test_town_rings_unchanged_except_gates():
    # town interiors get their RO layouts (tests/test_zones.py); the wall ring only gains gates
    for s in SEEDS:
        a, b = OLD[s], TERRAINS[s]
        for i, (x0, y0, x1, y1) in enumerate(wl.town_boxes()):
            ra, rb = a[x0:x1 + 1, y0:y1 + 1], b[x0:x1 + 1, y0:y1 + 1]
            diff = ra != rb
            diff[1:-1, 1:-1] = False
            if diff.any():
                assert (ra[diff] == T_WALL).all()
        # temples identical
        tt = np.asarray(wl.temple_tiles()).reshape(-1, 2)
        assert (OLD[s][tt[:, 0], tt[:, 1]] == T_TEMPLE).all()
        assert (TERRAINS[s][tt[:, 0], tt[:, 1]] == T_TEMPLE).all()


def gate_count(t, box):
    x0, y0, x1, y1 = box
    ring = ([(x, y0) for x in range(x0, x1 + 1)] + [(x1, y) for y in range(y0 + 1, y1 + 1)]
            + [(x, y1) for x in range(x1 - 1, x0 - 1, -1)] + [(x0, y) for y in range(y1 - 1, y0, -1)])
    open_ = [t[p] != T_WALL for p in ring]
    return sum(1 for i in range(len(open_)) if open_[i] and not open_[i - 1])


def test_town0_gates():
    for t in TERRAINS.values():
        assert gate_count(t, wl.town_boxes()[0]) >= 4


def test_border_rock():
    for t in TERRAINS.values():
        for edge in (t[0], t[-1], t[:, 0], t[:, -1]):
            assert (edge == T_ROCK).all()


FIELDS = {}


def fields(s):
    if s not in FIELDS:
        FIELDS[s] = (np.asarray(distance_fields_ro(jnp.asarray(TERRAINS[s]))),
                     np.asarray(distance_fields(jnp.asarray(TERRAINS[s]))))
    return FIELDS[s]


def test_distance_fields_ro():
    t = TERRAINS[0]
    ro, base = fields(0)
    cave = t == T_CAVE
    assert (ro[:, cave] < DIST_UNREACHABLE).all()
    dung = np.zeros((MAP, MAP), bool)
    for x0, y0, x1, y1 in BOXES:
        dung[x0:x1 + 1, y0:y1 + 1] = True
    # warps only add paths: never longer than the plain field
    assert (ro <= base).all()
    assert (ro[:, ~dung] < DIST_UNREACHABLE).any()
    # exact: reverse BFS from the gate aprons over walk steps plus warp edges d[src] = d[dst] + 1
    from game.map_gen import gate_aprons
    aprons = np.asarray(gate_aprons(jnp.asarray(t)))
    free = free_of(t)
    back = {}
    for a, b in zip(SRC.tolist(), DST.tolist()):
        back.setdefault(tuple(b), []).append(tuple(a))
    for k in range(aprons.shape[0]):
        ref = np.full((MAP, MAP), DIST_UNREACHABLE, np.int32)
        q = deque()
        for x, y in zip(*np.nonzero(aprons[k])):
            ref[x, y] = 0
            q.append((x, y))
        while q:
            u = q.popleft()
            nb = [(u[0] + 1, u[1]), (u[0] - 1, u[1]), (u[0], u[1] + 1), (u[0], u[1] - 1)] + back.get(u, [])
            for n in nb:
                if 0 <= n[0] < MAP and 0 <= n[1] < MAP and free[n] and ref[n] == DIST_UNREACHABLE:
                    ref[n] = ref[u] + 1
                    q.append(n)
        assert (ro[k] == ref).all()
    for ent, ins, ext, out in PADS:
        assert (ro[:, ins[0], ins[1]] == ro[:, out[0], out[1]].astype(np.int32) + wl.WARP_GAP_TILES + 1).all()


def test_jit_vmap():
    keys = jnp.stack([jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
    out = np.asarray(jax.jit(jax.vmap(make_terrain_ro))(keys))
    assert out.shape == (2, MAP, MAP)
    for i, s in enumerate(SEEDS):
        assert (out[i] == TERRAINS[s]).all()
