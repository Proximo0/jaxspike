"""World layout spec: towns, roads and temples as map fractions, plus layout queries."""
import jax.numpy as jnp
import numpy as np

from game.config import MAP, MAX_PLAYERS

TOWNS = (  # (cx, cy, radius) as fractions of MAP; radius is half the wall-ring side
    (0.50, 0.50, 0.060),   # Center
    (0.20, 0.22, 0.050),
    (0.80, 0.20, 0.050),
    (0.78, 0.80, 0.050),
    (0.22, 0.78, 0.050),
)
ROADS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))  # town index pairs
TEMPLE_TILES = 12          # temple square side in tiles, scale-independent
RAT_RING = 0.08            # fraction of MAP: monsters closer than this to a town wall are rats

WATER_THRESHOLD = 0.78     # noise above this becomes water
FOREST_THRESHOLD = 0.62    # noise above this (and a coin flip) becomes trees
TREE_DENSITY = 0.6
NOISE_CELLS = 32           # noise grid is MAP // NOISE_CELLS cells per side
ROAD_HALF_WIDTH = 1        # roads are 2 * ROAD_HALF_WIDTH + 1 tiles wide

assert len(TOWNS) * TEMPLE_TILES ** 2 >= MAX_PLAYERS


def town_centers():
    """Return int32 [NT, 2] town centre tiles."""
    return np.array([(round(cf * MAP), round(cf_y * MAP)) for cf, cf_y, _ in TOWNS], np.int32)


def town_boxes():
    """Return int32 [NT, 4] inclusive (x0, y0, x1, y1) bounds of each town's wall ring."""
    out = []
    for cf, cf_y, rf in TOWNS:
        r = max(round(rf * MAP), TEMPLE_TILES // 2 + 4)
        cx, cy = round(cf * MAP), round(cf_y * MAP)
        out.append((cx - r, cy - r, cx + r, cy + r))
    return np.array(out, np.int32)


def temple_tiles():
    """Return int32 [NT, TEMPLE_TILES**2, 2] row-major tiles of each town's temple square."""
    off = np.arange(TEMPLE_TILES) - TEMPLE_TILES // 2
    dx, dy = np.meshgrid(off, off, indexing="ij")
    rel = np.stack([dx.ravel(), dy.ravel()], 1)
    return jnp.asarray(town_centers()[:, None, :] + rel[None], jnp.int32)


def dist_to_nearest_town(pos):
    """Return int32 [N]: Chebyshev distance from each pos [N, 2] to the nearest town box."""
    d = None
    for x0, y0, x1, y1 in town_boxes():
        dx = jnp.maximum(jnp.maximum(x0 - pos[:, 0], pos[:, 0] - x1), 0)
        dy = jnp.maximum(jnp.maximum(y0 - pos[:, 1], pos[:, 1] - y1), 0)
        cur = jnp.maximum(dx, dy)
        d = cur if d is None else jnp.minimum(d, cur)
    return d.astype(jnp.int32)
