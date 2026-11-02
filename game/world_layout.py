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


# ----------------------------------------------------------------------------------------------
# RO world (ro/world): names, field bands, dungeons, warps, kafras. Still fractions of MAP (and
# tile-count constants that do not scale, like TEMPLE_TILES). Terrain ids 8..11: game/data/terrain.py.
# ----------------------------------------------------------------------------------------------
TOWN_NAMES = ("Prontera", "Geffen", "Payon", "Alberta", "Morroc")   # index = TOWNS index
TOWN_ROLES = ("capital", "mage city", "archer village", "merchant port", "desert thief town")
assert len(TOWN_NAMES) == len(TOWNS) == len(TOWN_ROLES)

# Field rings by walking distance to the nearest town gate (fractions of MAP), with the monster kind
# names (game/data/monsters.py) that spawn in each band. The last band is open-ended.
FIELD_BANDS = (
    (0.00, 0.08, ("Poring", "Fabre", "Lunatic", "Drops")),
    (0.08, 0.16, ("Poporing", "Willow", "Rocker", "Spore", "Thief Bug")),
    (0.16, 9.99, ("Wolf", "Peco Peco", "Hode", "Bigfoot", "Orc Warrior")),
)

# Dungeons: (cx, cy, radius) as fractions of MAP (radius = half the rock-ring side), name, the kinds
# that spawn inside, the MVP kind (exactly one per dungeon), and the side its entrance faces
# (+1 = the max-y side, -1 = the min-y side; both face the capital).
DUNGEONS = (
    (0.50, 0.09, 0.055, "Prontera Culvert", ("Thief Bug", "Spore", "Zombie", "Skeleton"), "Golden Thief Bug", +1),
    (0.50, 0.91, 0.055, "Payon Cave", ("Zombie", "Skeleton", "Soldier Skeleton", "Mummy"), "Eddga", -1),
)
DUNGEON_ROADS = ((0, 0), (1, 0))   # (dungeon index, town index): a road from the entrance pad to the town
DUNGEON_WALL_TILES = 2             # rock ring thickness
DUNGEON_PILLAR_PERIOD = 8          # rock pillars inside the dungeon every this many tiles ...
DUNGEON_PILLAR_SIZE = 2            # ... each this many tiles square ...
DUNGEON_PILLAR_PHASE = 5           # ... starting at this offset from the dungeon centre (mod PERIOD)
WARP_PAD_TILES = 3                 # warp/station tiles sit this far outside/inside the rock ring
WARP_GAP_TILES = 3                 # exit warp and outside station sit this far right of the entrance
KAFRA_OFFSET_TILES = 2             # kafra tile this far right of the temple square; station this far left
N_WARPS = 16                       # fixed warp-table size (padded with -1 rows); must stay <= 64


def dungeon_boxes():
    """Return int32 [ND, 4] inclusive (x0, y0, x1, y1) bounds of each dungeon's rock ring."""
    out = []
    for cf, cf_y, rf, *_ in DUNGEONS:
        r = max(round(rf * MAP), 2 * WARP_PAD_TILES + DUNGEON_WALL_TILES + 2)
        cx, cy = round(cf * MAP), round(cf_y * MAP)
        out.append((cx - r, cy - r, cx + r, cy + r))
    return np.array(out, np.int32)


def dungeon_centers():
    """Return int32 [ND, 2] dungeon centre tiles."""
    return np.array([(round(cf * MAP), round(cf_y * MAP)) for cf, cf_y, *_ in DUNGEONS], np.int32)


def dungeon_pads():
    """Return int32 [ND, 4, 2]: per dungeon (entrance warp, inside station, exit warp, outside station) tiles.

    The entrance warp (T_WARP) stands WARP_PAD_TILES outside the ring on the entrance side and sends
    to the inside station (T_STATION) WARP_PAD_TILES inside the ring. The exit warp sits inside,
    WARP_GAP_TILES to the right of the inside station, and sends to the outside station, the same
    distance to the right of the entrance warp.
    """
    out = []
    for (x0, y0, x1, y1), (cx, _), d in zip(dungeon_boxes().tolist(), dungeon_centers().tolist(), DUNGEONS):
        side = d[6]
        edge = y1 if side > 0 else y0
        y_out = edge + side * WARP_PAD_TILES
        y_in = edge - side * WARP_PAD_TILES
        out.append(((cx, y_out), (cx, y_in), (cx + WARP_GAP_TILES, y_in), (cx + WARP_GAP_TILES, y_out)))
    return np.array(out, np.int32)


def _frac(tile):
    return (tile[0] / MAP, tile[1] / MAP)


# (from fraction, to fraction) pairs: stepping on `from` moves the player to `to`. Derived from the
# dungeon pads so the two stay consistent; MAP is a power of two, so fraction * MAP is exact.
WARPS = tuple(
    pair for pads in dungeon_pads().tolist()
    for pair in ((_frac(pads[0]), _frac(pads[1])), (_frac(pads[2]), _frac(pads[3])))
)
assert len(WARPS) <= N_WARPS <= 64


def warp_tiles():
    """Return (src int32 [N_REAL, 2], dst int32 [N_REAL, 2]) warp tiles from WARPS (unpadded)."""
    src = np.array([(round(fx * MAP), round(fy * MAP)) for (fx, fy), _ in WARPS], np.int32).reshape(-1, 2)
    dst = np.array([(round(fx * MAP), round(fy * MAP)) for _, (fx, fy) in WARPS], np.int32).reshape(-1, 2)
    return src, dst


def kafra_tiles():
    """Return int32 [NT, 2]: the kafra tile of each town, just right of its temple square."""
    c = town_centers()
    return np.stack([c[:, 0] + TEMPLE_TILES // 2 + KAFRA_OFFSET_TILES, c[:, 1]], 1).astype(np.int32)


def town_station_tiles():
    """Return int32 [NT, 2]: the arrival station of each town, just left of its temple square."""
    c = town_centers()
    return np.stack([c[:, 0] - TEMPLE_TILES // 2 - KAFRA_OFFSET_TILES, c[:, 1]], 1).astype(np.int32)


def dungeon_index(pos):
    """Return int32 [N]: index of the dungeon whose box holds each pos [N, 2], or -1."""
    out = jnp.full((pos.shape[0],), -1, jnp.int32)
    for k, (x0, y0, x1, y1) in enumerate(dungeon_boxes().tolist()):
        inside = (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
        out = jnp.where(inside, k, out)
    return out
