from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from game.config import (MAP, MAX_PLAYERS, T_FLOOR, T_GRASS, T_ROAD, T_ROCK, T_TEMPLE,
                         T_TREE, T_WALL, T_WATER)
from game.map_gen import blocking, make_terrain, make_walls, protection_zone, walkable
from game.world_layout import (ROAD_HALF_WIDTH, TEMPLE_TILES, TOWNS, dist_to_nearest_town,
                               temple_tiles, town_boxes)


def terrain_of(seed):
    return np.asarray(make_terrain(jax.random.PRNGKey(seed)))


def outside_boxes_mask():
    m = np.ones((MAP, MAP), bool)
    for x0, y0, x1, y1 in town_boxes():
        m[x0:x1 + 1, y0:y1 + 1] = False
    return m


def test_border_is_rock():
    t = terrain_of(0)
    assert t.shape == (MAP, MAP) and t.dtype == np.int8
    for edge in (t[0], t[-1], t[:, 0], t[:, -1]):
        assert (edge == T_ROCK).all()


def test_towns_are_floor_with_wall_ring_and_temple():
    for s in range(3):
        t = terrain_of(s)
        for x0, y0, x1, y1 in town_boxes():
            inner = t[x0 + 1:x1, y0 + 1:y1]
            assert np.isin(inner, [T_FLOOR, T_TEMPLE, T_ROAD]).all()
            for side in (t[x0, y0:y1 + 1], t[x1, y0:y1 + 1], t[x0:x1 + 1, y0], t[x0:x1 + 1, y1]):
                assert np.isin(side, [T_WALL, T_ROAD]).all()
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            h = TEMPLE_TILES // 2
            assert (t[cx - h:cx + h, cy - h:cy + h] == T_TEMPLE).all()


def test_no_water_or_trees_in_towns():
    for s in range(3):
        t = terrain_of(s)
        for x0, y0, x1, y1 in town_boxes():
            box = t[x0:x1 + 1, y0:y1 + 1]
            assert not np.isin(box, [T_WATER, T_TREE]).any()


def test_each_town_has_two_gates():
    t = terrain_of(0)
    for x0, y0, x1, y1 in town_boxes():
        ring = np.concatenate([t[x0, y0:y1 + 1], t[x1, y0:y1 + 1],
                               t[x0 + 1:x1, y0], t[x0 + 1:x1, y1]])
        assert (ring == T_ROAD).sum() >= 2 * (2 * ROAD_HALF_WIDTH + 1)


def test_wall_rings_mostly_intact():
    t = terrain_of(0)
    for x0, y0, x1, y1 in town_boxes():
        for side in (t[x0, y0:y1 + 1], t[x1, y0:y1 + 1], t[x0:x1 + 1, y0], t[x0:x1 + 1, y1]):
            assert (side == T_WALL).mean() >= 0.5


def bfs_reached(free, start):
    seen = np.zeros_like(free)
    seen[start] = True
    q = deque([start])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < MAP and 0 <= ny < MAP and free[nx, ny] and not seen[nx, ny]:
                seen[nx, ny] = True
                q.append((nx, ny))
    return seen


def test_temples_connected():
    for s in range(2):
        t = jax.random.PRNGKey(s)
        free = np.asarray(blocking(make_terrain(t))) == 0
        tt = np.asarray(temple_tiles())
        seen = bfs_reached(free, tuple(tt[0][0]))
        for k in range(len(tt)):
            assert seen[tuple(tt[k][0])]


def test_pz_covers_gate_not_outside_road():
    for s in range(2):
        t = terrain_of(s)
        pz = np.asarray(protection_zone(jnp.asarray(t)))
        d = np.asarray(dist_to_nearest_town(
            jnp.asarray(np.stack(np.nonzero(t == T_ROAD), 1), jnp.int32)))
        rx, ry = np.nonzero(t == T_ROAD)
        assert pz[rx[d == 0], ry[d == 0]].all()
        assert not pz[rx[d > 0], ry[d > 0]].any()
        assert (d == 0).any() and (d > 0).any()
        assert pz[(t == T_FLOOR) | (t == T_TEMPLE)].all()
        assert not pz[t == T_GRASS].any()


def test_wilderness_has_water_and_trees():
    out = outside_boxes_mask()
    for s in range(3):
        t = terrain_of(s)[out]
        assert 0.02 < (t == T_WATER).mean() < 0.25
        assert 0.03 < (t == T_TREE).mean() < 0.30


def test_towns_fixed_across_keys():
    a, b = terrain_of(7), terrain_of(8)
    out = outside_boxes_mask()
    assert np.array_equal(a[~out], b[~out])
    assert not np.array_equal(a[out], b[out])


def test_deterministic():
    assert np.array_equal(terrain_of(7), terrain_of(7))


def test_jit_and_vmap():
    k = jax.random.PRNGKey(7)
    assert np.array_equal(np.asarray(jax.jit(make_terrain)(k)), np.asarray(make_terrain(k)))
    keys = jax.random.split(jax.random.PRNGKey(2), 3)
    assert jax.vmap(make_terrain)(keys).shape == (3, MAP, MAP)


def test_dist_to_nearest_town():
    boxes = town_boxes()
    assert boxes.shape == (len(TOWNS), 4)
    tt = np.asarray(temple_tiles())
    x1, y1 = boxes[0][2], boxes[0][3]
    far = np.array([MAP - 1, 0])
    pos = jnp.asarray(np.stack([tt[0][0], [x1 + 1, y1], far]), jnp.int32)
    d = np.asarray(dist_to_nearest_town(pos))
    assert d[0] == 0 and d[1] == 1

    def cheb(p, b):
        return max(b[0] - p[0], 0, p[0] - b[2], b[1] - p[1], 0, p[1] - b[3])

    assert d[2] == min(cheb(far, b) for b in boxes)


def test_temple_capacity_covers_max_players():
    assert len(TOWNS) * TEMPLE_TILES ** 2 >= MAX_PLAYERS


def test_walkable_walls_and_oob():
    w = make_walls(jax.random.PRNGKey(0))
    pos = jnp.array([[0, 0], [20, 20], [-1, 5], [5, -1], [MAP, 5], [5, MAP]], jnp.int32)
    out = np.asarray(walkable(w, pos))
    assert not out[0] and not out[2:].any()
    wn = np.asarray(w)
    xs, ys = np.nonzero(wn == 1)
    p = jnp.array(np.stack([xs[:50], ys[:50]], 1), jnp.int32)
    assert not np.asarray(walkable(w, p)).any()
    tt = temple_tiles()[0]
    assert np.asarray(walkable(w, tt)).all()
