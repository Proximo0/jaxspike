"""Terrain generation: wilderness noise, walled towns with temples, and connecting roads."""
import jax
import jax.numpy as jnp

from game.config import (MAP, T_FLOOR, T_GRASS, T_ROAD, T_ROCK, T_TEMPLE, T_TREE, T_WALL,
                         T_WATER)
from game.world_layout import (FOREST_THRESHOLD, NOISE_CELLS, ROAD_HALF_WIDTH, ROADS,
                               TEMPLE_TILES, TREE_DENSITY, WATER_THRESHOLD, town_boxes,
                               town_centers)


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
    return (terrain == T_FLOOR) | (terrain == T_TEMPLE) | ((terrain == T_ROAD) & in_town)


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
