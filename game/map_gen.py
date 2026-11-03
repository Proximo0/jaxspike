"""Terrain generation: wilderness noise, walled towns with temples, and connecting roads."""
import jax
import jax.numpy as jnp
import numpy as np

from game.config import (MAP, T_FLOOR, T_GRASS, T_ROAD, T_ROCK, T_TEMPLE, T_TREE, T_WALL,
                         T_WATER)
from game.data.terrain import T_CAVE, T_KAFRA, T_SHOP, T_STATION, T_WARP
from game.world_layout import (FOREST_THRESHOLD, NOISE_CELLS, ROAD_HALF_WIDTH, ROADS,
                               TEMPLE_TILES, TREE_DENSITY, WATER_THRESHOLD, town_boxes,
                               town_centers)
from game import world_layout as wl


def _noise(key):
    """Return float [MAP, MAP] smooth uniform-ish noise from a coarse random grid."""
    n = MAP // NOISE_CELLS
    return jax.image.resize(jax.random.uniform(key, (n, n)), (MAP, MAP), "bilinear")


def _in_rect(xs, ys, x0, y0, x1, y1):
    return (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)


def make_terrain(key):
    """Return int8 [MAP, MAP] terrain ids for a fresh world."""
    xs, ys = jnp.meshgrid(jnp.arange(MAP), jnp.arange(MAP), indexing="ij")
    k_water, k_forest, k_coin = jax.random.split(key, 3)
    t = jnp.full((MAP, MAP), T_GRASS, jnp.int8)

    t = jnp.where(_noise(k_water) > WATER_THRESHOLD, T_WATER, t)
    tree = (_noise(k_forest) > FOREST_THRESHOLD) & jax.random.bernoulli(k_coin, TREE_DENSITY, (MAP, MAP))
    t = jnp.where(tree & (t == T_GRASS), T_TREE, t)

    h = TEMPLE_TILES // 2
    for (x0, y0, x1, y1), (cx, cy) in zip(town_boxes().tolist(), town_centers().tolist()):
        ring = _in_rect(xs, ys, x0, y0, x1, y1) & ~_in_rect(xs, ys, x0 + 1, y0 + 1, x1 - 1, y1 - 1)
        t = jnp.where(_in_rect(xs, ys, x0, y0, x1, y1), T_FLOOR, t)
        t = jnp.where(ring, T_WALL, t)
        t = jnp.where(_in_rect(xs, ys, cx - h, cy - h, cx + h - 1, cy + h - 1), T_TEMPLE, t)

    hw = ROAD_HALF_WIDTH
    centers = town_centers().tolist()
    replaceable = (t == T_GRASS) | (t == T_WATER) | (t == T_TREE) | (t == T_WALL)
    road = jnp.zeros((MAP, MAP), bool)
    for a, b in ROADS:
        (ax, ay), (bx, by) = centers[a], centers[b]
        road |= _in_rect(xs, ys, min(ax, bx) - hw, ay - hw, max(ax, bx) + hw, ay + hw)
        road |= _in_rect(xs, ys, bx - hw, min(ay, by) - hw, bx + hw, max(ay, by) + hw)
    t = jnp.where(road & replaceable, T_ROAD, t)

    border = (xs == 0) | (xs == MAP - 1) | (ys == 0) | (ys == MAP - 1)
    return jnp.where(border, T_ROCK, t).astype(jnp.int8)


def blocking(terrain):
    """Return int8 [MAP, MAP]: 1 where terrain is water, tree, wall or rock."""
    b = (terrain == T_WATER) | (terrain == T_TREE) | (terrain == T_WALL) | (terrain == T_ROCK)
    return b.astype(jnp.int8)


def protection_zone(terrain):
    """Return bool [MAP, MAP]: floor, temple, and road tiles inside a town box."""
    xs, ys = jnp.meshgrid(jnp.arange(MAP), jnp.arange(MAP), indexing="ij")
    in_town = jnp.zeros((MAP, MAP), bool)
    for x0, y0, x1, y1 in town_boxes().tolist():
        in_town |= _in_rect(xs, ys, x0, y0, x1, y1)
    special = (terrain == T_KAFRA) | (terrain == T_STATION) | (terrain == T_SHOP)
    return (terrain == T_FLOOR) | (terrain == T_TEMPLE) | (((terrain == T_ROAD) | special) & in_town)


DIST_UNREACHABLE = 32767  # int16 max: tiles with no walkable path to a town's gates


def _town_masks():
    """Return bool [NT, MAP, MAP]: tiles inside each town's wall box."""
    xs, ys = jnp.meshgrid(jnp.arange(MAP), jnp.arange(MAP), indexing="ij")
    return jnp.stack([_in_rect(xs, ys, *box) for box in town_boxes().tolist()])


def _shift_min(a):
    """Elementwise min over the four axis neighbours of int [..., MAP, MAP], borders padded with the array max."""
    pad = jnp.iinfo(a.dtype).max
    p = jnp.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], constant_values=pad)
    return jnp.minimum(jnp.minimum(p[..., 2:, 1:-1], p[..., :-2, 1:-1]),
                       jnp.minimum(p[..., 1:-1, 2:], p[..., 1:-1, :-2]))


def gate_aprons(terrain):
    """Return bool [NT, MAP, MAP]: walkable non-town tiles one step outside each town's gates."""
    pz = protection_zone(terrain)
    town_pz = _town_masks() & pz[None]
    # a tile is an apron tile when a 4-neighbour is in this town's pz: min over neighbours of ~town_pz is 0
    near = _shift_min((~town_pz).astype(jnp.int8)) == 0
    return near & ~pz[None] & (blocking(terrain) == 0)[None]


def distance_fields(terrain):
    """Return int16 [NT, MAP, MAP]: walking distance from every tile to each town's gate aprons.

    Relaxation `d = min(d, 1 + min(neighbours))` over walkable tiles until it stops changing;
    unreachable tiles hold DIST_UNREACHABLE. Runs once per world at creation.
    """
    free = blocking(terrain) == 0
    d0 = jnp.where(gate_aprons(terrain), 0, DIST_UNREACHABLE).astype(jnp.int32)

    def relax(d):
        return jnp.where(free[None], jnp.minimum(d, jnp.minimum(_shift_min(d) + 1, DIST_UNREACHABLE)), DIST_UNREACHABLE)

    def body(carry):
        d, _ = carry
        nd = relax(d)
        return nd, jnp.any(nd != d)

    d, _ = jax.lax.while_loop(lambda c: c[1], body, (d0, jnp.bool_(True)))
    return d.astype(jnp.int16)


def make_walls(key):
    """Return int8 [MAP, MAP] blocking mask of a fresh world (kept for existing call sites)."""
    return blocking(make_terrain(key))


def walkable(walls, pos):
    """Return bool [N]: pos (int [N,2]) is in bounds and not a wall."""
    x, y = pos[:, 0], pos[:, 1]
    inb = (x >= 0) & (x < MAP) & (y >= 0) & (y < MAP)
    cx = jnp.clip(x, 0, MAP - 1)
    cy = jnp.clip(y, 0, MAP - 1)
    return inb & (walls[cx, cy] == 0)


def warp_table(terrain=None):
    """Return (src int32 [N_WARPS, 2], dst int32 [N_WARPS, 2]) fixed warp arrays, padded with -1 rows.

    Warps are part of the layout (world_layout.WARPS), not read from the terrain; the argument is
    accepted so callers can pass world["terrain"] uniformly. ro_warps.apply_warps looks a player's
    tile up in src (jnp.where over N_WARPS) and moves it to the matching dst.
    """
    from game.world_layout import N_WARPS, warp_tiles
    s, d = warp_tiles()
    pad = N_WARPS - s.shape[0]
    src = jnp.asarray(np.concatenate([s, np.full((pad, 2), -1, np.int32)]), jnp.int32)
    dst = jnp.asarray(np.concatenate([d, np.full((pad, 2), -1, np.int32)]), jnp.int32)
    return src, dst


def make_terrain_ro(key):
    """Return int8 [MAP, MAP] terrain of the RO world: make_terrain plus dungeons, zone split lines with
    portals, town layouts (plaza, streets, decoration, shops), warps, kafras, stations."""
    t = make_terrain(key)
    xs, ys = jnp.meshgrid(jnp.arange(MAP), jnp.arange(MAP), indexing="ij")
    hw = ROAD_HALF_WIDTH
    centers = town_centers().tolist()
    pads = wl.dungeon_pads().tolist()
    boxes = wl.dungeon_boxes().tolist()
    dcent = wl.dungeon_centers().tolist()

    # a. dungeon roads: entrance warp tile -> town centre, same L-shape and rule as the town roads
    replaceable = (t == T_GRASS) | (t == T_WATER) | (t == T_TREE) | (t == T_WALL)
    road = jnp.zeros((MAP, MAP), bool)
    for k, town in wl.DUNGEON_ROADS:
        (ax, ay), (bx, by) = pads[k][0], centers[town]
        road |= _in_rect(xs, ys, min(ax, bx) - hw, ay - hw, max(ax, bx) + hw, ay + hw)
        road |= _in_rect(xs, ys, bx - hw, min(ay, by) - hw, bx + hw, max(ay, by) + hw)
    t = jnp.where(road & replaceable, T_ROAD, t)

    # b. entrance plaza over the outside pad tiles plus one tile of margin
    for ent, _, ext, out in pads:
        px0, px1 = min(ent[0], out[0], ext[0]) - 1, max(ent[0], out[0], ext[0]) + 1
        py0, py1 = min(ent[1], out[1]) - 1, max(ent[1], out[1]) + 1
        t = jnp.where(_in_rect(xs, ys, px0, py0, px1, py1), T_ROAD, t)

    # c. dungeon boxes: cave floor, rock ring, rock pillars clear of the inside pad tiles
    w = wl.DUNGEON_WALL_TILES
    for (x0, y0, x1, y1), (cx, cy), (_, ins, ext, _) in zip(boxes, dcent, pads):
        inner = _in_rect(xs, ys, x0 + w, y0 + w, x1 - w, y1 - w)
        box = _in_rect(xs, ys, x0, y0, x1, y1)
        px = (xs - cx - wl.DUNGEON_PILLAR_PHASE) % wl.DUNGEON_PILLAR_PERIOD < wl.DUNGEON_PILLAR_SIZE
        py = (ys - cy - wl.DUNGEON_PILLAR_PHASE) % wl.DUNGEON_PILLAR_PERIOD < wl.DUNGEON_PILLAR_SIZE
        pillar = px & py & inner
        for (qx, qy) in (ins, ext):
            for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                pillar &= ~((xs == qx + ox) & (ys == qy + oy))
        t = jnp.where(box, T_ROCK, t)
        t = jnp.where(inner, T_CAVE, t)
        t = jnp.where(pillar, T_ROCK, t)

    # d. zone split lines: one tile of rock across the map outside town and dungeon boxes; the portal
    #    warps of step f sit on the road tiles either side of each crossing
    keep = jnp.zeros((MAP, MAP), bool)
    for box in wl.zone_boxes()[[z[1] != wl.ZONE_FIELD for z in wl.ZONES]].tolist():
        keep |= _in_rect(xs, ys, *box)
    line = jnp.zeros((MAP, MAP), bool)
    for v in wl.zone_line_tiles():
        line |= (xs == v) | (ys == v)
    t = jnp.where(line & ~keep, T_ROCK, t)

    # e. town layouts: decoration over floor only, then plaza and streets, then Prontera's fountains
    decor_tid = {"wall": T_WALL, "water": T_WATER, "tree": T_TREE}
    for _, kind, x0, y0, x1, y1, period, size in wl.town_decor():
        m = _in_rect(xs, ys, x0, y0, x1, y1) & ((xs - x0) % period < size) & ((ys - y0) % period < size)
        t = jnp.where(m & (t == T_FLOOR), decor_tid[kind], t)
    for _, x0, y0, x1, y1 in wl.town_paving():
        t = jnp.where(_in_rect(xs, ys, x0, y0, x1, y1) & (t != T_TEMPLE), T_ROAD, t)
    for _, x0, y0, x1, y1 in wl.town_fountains():
        t = jnp.where(_in_rect(xs, ys, x0, y0, x1, y1), T_WATER, t)

    # f. special tiles last (warps include the zone portals)
    src, dst = wl.warp_tiles()
    for arr, tid in ((src, T_WARP), (dst, T_STATION), (wl.kafra_tiles(), T_KAFRA),
                     (wl.town_station_tiles(), T_STATION), (wl.shop_tiles()[0], T_SHOP)):
        t = t.at[arr[:, 0], arr[:, 1]].set(tid)
    return t.astype(jnp.int8)


def distance_fields_ro(terrain):
    """Return int16 [NT, MAP, MAP] like distance_fields, but relaxing across warps: d[src] <= d[dst] + 1.

    Towns are independent, so the relaxation runs one town at a time (`lax.map` over the town axis):
    the loop's temporaries are [MAP, MAP] instead of [NT, MAP, MAP] (docs/world_scale.md plan item 1),
    and the result is identical to relaxing all towns together.
    """
    free = blocking(terrain) == 0
    d0 = jnp.where(gate_aprons(terrain), 0, DIST_UNREACHABLE).astype(jnp.int32)
    src, dst = warp_table(terrain)
    real = src[:, 0] >= 0
    sx, sy = jnp.clip(src[:, 0], 0, MAP - 1), jnp.clip(src[:, 1], 0, MAP - 1)
    dx, dy = jnp.clip(dst[:, 0], 0, MAP - 1), jnp.clip(dst[:, 1], 0, MAP - 1)

    def relax(d):
        d = jnp.where(free, jnp.minimum(d, jnp.minimum(_shift_min(d) + 1, DIST_UNREACHABLE)), DIST_UNREACHABLE)
        via = jnp.where(real, jnp.minimum(d[dx, dy] + 1, DIST_UNREACHABLE), DIST_UNREACHABLE)
        return d.at[sx, sy].min(via)

    def body(carry):
        d, _ = carry
        nd = relax(d)
        return nd, jnp.any(nd != d)

    def one_town(d):
        d, _ = jax.lax.while_loop(lambda c: c[1], body, (d, jnp.bool_(True)))
        return d.astype(jnp.int16)

    return jax.lax.map(one_town, d0)
