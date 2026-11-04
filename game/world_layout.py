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
# RO world (ro/world): names, field bands, dungeons, warps, waybells. Still fractions of MAP (and
# tile-count constants that do not scale, like TEMPLE_TILES). Terrain ids 8..11: game/data/terrain.py.
# ----------------------------------------------------------------------------------------------
TOWN_NAMES = ("Solmera", "Lunavell", "Pinehollow", "Tidewick", "Kharsand")   # index = TOWNS index
TOWN_ROLES = ("capital", "mage city", "archer village", "merchant port", "desert thief town")
assert len(TOWN_NAMES) == len(TOWNS) == len(TOWN_ROLES)

# Field rings by walking distance to the nearest town gate (fractions of MAP), with the monster kind
# names (game/data/monsters.py) that spawn in each band. The last band is open-ended.
FIELD_BANDS = (
    (0.00, 0.08, ("Blobbin", "Wrigglet", "Moonbun", "Emberblob")),
    (0.08, 0.16, ("Vexblob", "Bramblet", "Fiddlehop", "Capling", "Pilferbug")),
    (0.16, 9.99, ("Snarlfang", "Gallopeck", "Sandgulper", "Grumblebear", "Grukk Brute")),
)

# Dungeons: (cx, cy, radius) as fractions of MAP (radius = half the rock-ring side), name, the kinds
# that spawn inside, the MVP kind (exactly one per dungeon), and the side its entrance faces
# (+1 = the max-y side, -1 = the min-y side; both face the capital).
DUNGEONS = (
    (0.50, 0.09, 0.055, "Solmera Culvert", ("Pilferbug", "Capling", "Rotling", "Clackbone"), "Gildscarab", +1),
    (0.50, 0.91, 0.055, "Pinehollow Cave", ("Rotling", "Clackbone", "Clackbone Guard", "Tombwrap"), "Scorchstripe", -1),
)
DUNGEON_ROADS = ((0, 0), (1, 0))   # (dungeon index, town index): a road from the entrance pad to the town
DUNGEON_WALL_TILES = 2             # rock ring thickness
DUNGEON_PILLAR_PERIOD = 8          # rock pillars inside the dungeon every this many tiles ...
DUNGEON_PILLAR_SIZE = 2            # ... each this many tiles square ...
DUNGEON_PILLAR_PHASE = 5           # ... starting at this offset from the dungeon centre (mod PERIOD)
WARP_PAD_TILES = 3                 # warp/station tiles sit this far outside/inside the rock ring
WARP_GAP_TILES = 3                 # exit warp and outside station sit this far right of the entrance
WAYBELL_OFFSET_TILES = 2             # waybell tile this far right of the temple square; station this far left
N_WARPS = 48                       # fixed warp-table size (padded with -1 rows); must stay <= 64 (4 dungeon + 28 portal legs)


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
# WARPS (all warps: these, then the zone portals) is completed after the zone section below.
DUNGEON_WARPS = tuple(
    pair for pads in dungeon_pads().tolist()
    for pair in ((_frac(pads[0]), _frac(pads[1])), (_frac(pads[2]), _frac(pads[3])))
)


def warp_tiles():
    """Return (src int32 [N_REAL, 2], dst int32 [N_REAL, 2]) warp tiles from WARPS (unpadded)."""
    src = np.array([(round(fx * MAP), round(fy * MAP)) for (fx, fy), _ in WARPS], np.int32).reshape(-1, 2)
    dst = np.array([(round(fx * MAP), round(fy * MAP)) for _, (fx, fy) in WARPS], np.int32).reshape(-1, 2)
    return src, dst


def waybell_tiles():
    """Return int32 [NT, 2]: the waybell tile of each town, just right of its temple square."""
    c = town_centers()
    return np.stack([c[:, 0] + TEMPLE_TILES // 2 + WAYBELL_OFFSET_TILES, c[:, 1]], 1).astype(np.int32)


def town_station_tiles():
    """Return int32 [NT, 2]: the arrival station of each town, just left of its temple square."""
    c = town_centers()
    return np.stack([c[:, 0] - TEMPLE_TILES // 2 - WAYBELL_OFFSET_TILES, c[:, 1]], 1).astype(np.int32)


def dungeon_index(pos):
    """Return int32 [N]: index of the dungeon whose box holds each pos [N, 2], or -1."""
    out = jnp.full((pos.shape[0],), -1, jnp.int32)
    for k, (x0, y0, x1, y1) in enumerate(dungeon_boxes().tolist()):
        inside = (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
        out = jnp.where(inside, k, out)
    return out


# ----------------------------------------------------------------------------------------------
# RO zones (ro-world2): each RO map is a named rectangle of the one big map. Field zones tile the map
# in a 3 x 3 grid split by one-tile rock lines; towns (their wall box) and dungeons (their rock box)
# sit inside field cells and take precedence. Walking off a field's edge where a road meets the rock
# line steps on a T_WARP that lands on the matching road tile across the line (a portal pair).
# ----------------------------------------------------------------------------------------------
ZONE_LINES = (0.375, 0.625)   # field grid split lines, same fractions on x and y (exact at any power-of-two MAP)
ZONE_TOWN, ZONE_FIELD, ZONE_DUNGEON = 0, 1, 2
ZONE_KINDS = ("town", "field", "dungeon")
PORTAL_STEP_TILES = 2         # a portal's station is this many tiles past the rock line (its warp is 1 tile before it)
PORTAL_MIN_GAP_TILES = 6      # two road crossings of one line closer than this share one portal pair

# (name, kind, ref, level band (lo, hi), music id). ref: TOWNS index for towns, DUNGEONS index for
# dungeons, (column, row) of the field grid for fields (columns split at ZONE_LINES on x, rows on y).
# Index = zone id (`zone_of`). Names follow RO map names; level bands are the RO monster levels found
# there (irowiki map pages, approximate), music ids are the RO BGM track numbers of those maps.
ZONES = (
    ("solmera", ZONE_TOWN, 0, (0, 0), 8),
    ("lunavell", ZONE_TOWN, 1, (0, 0), 13),
    ("pinehollow", ZONE_TOWN, 2, (0, 0), 3),
    ("tidewick", ZONE_TOWN, 3, (0, 0), 11),
    ("kharsand", ZONE_TOWN, 4, (0, 0), 18),
    ("sol_sew01", ZONE_DUNGEON, 0, (5, 25), 19),
    ("pin_dun00", ZONE_DUNGEON, 1, (15, 40), 20),
    ("lun_fld00", ZONE_FIELD, (0, 0), (8, 25), 12),
    ("sol_fld01", ZONE_FIELD, (1, 0), (3, 15), 9),
    ("pin_fld01", ZONE_FIELD, (2, 0), (5, 20), 4),
    ("sol_fld05", ZONE_FIELD, (0, 1), (3, 15), 9),
    ("sol_fld08", ZONE_FIELD, (1, 1), (1, 6), 9),
    ("sol_fld06", ZONE_FIELD, (2, 1), (5, 18), 9),
    ("kha_fld01", ZONE_FIELD, (0, 2), (10, 30), 17),
    ("sol_fld02", ZONE_FIELD, (1, 2), (3, 15), 9),
    ("tid_fld01", ZONE_FIELD, (2, 2), (5, 20), 10),
)
ZONE_NAMES = tuple(z[0] for z in ZONES)
N_ZONES = len(ZONES)


def _grid_edges():
    """Return the inclusive tile ranges [(lo, hi)] of the three grid columns (same for rows)."""
    cuts = [round(f * MAP) for f in ZONE_LINES]
    return [(0, cuts[0] - 1), (cuts[0], cuts[1] - 1), (cuts[1], MAP - 1)]


def zone_line_tiles():
    """Return the int x (= y) coordinates of the rock split lines: the first tile of each later grid cell."""
    return [round(f * MAP) for f in ZONE_LINES]


def zone_boxes():
    """Return int32 [N_ZONES, 4] inclusive (x0, y0, x1, y1) tile boxes (towns: wall box, dungeons: rock box)."""
    edges, tb, db = _grid_edges(), town_boxes(), dungeon_boxes()
    out = []
    for _, kind, ref, _, _ in ZONES:
        if kind == ZONE_TOWN:
            out.append(tb[ref].tolist())
        elif kind == ZONE_DUNGEON:
            out.append(db[ref].tolist())
        else:
            (x0, x1), (y0, y1) = edges[ref[0]], edges[ref[1]]
            out.append([x0, y0, x1, y1])
    return np.array(out, np.int32)


def zone_box_fractions():
    """Return float [N_ZONES, 4]: zone_boxes as fractions of MAP (x1, y1 exclusive), for docs and the client."""
    b = zone_boxes().astype(np.float64)
    return np.concatenate([b[:, :2], b[:, 2:] + 1], 1) / MAP


def _field_zone_ids():
    """Return int [3, 3] zone id of the field cell at (column, row)."""
    ids = np.zeros((3, 3), np.int32)
    for i, (_, kind, ref, _, _) in enumerate(ZONES):
        if kind == ZONE_FIELD:
            ids[ref[0], ref[1]] = i
    return ids


def zone_of(pos):
    """Return int32 [N]: zone id of each tile pos int [N, 2] (dungeon box > town box > field cell)."""
    lines = zone_line_tiles()
    col = (pos[:, 0] >= lines[0]).astype(jnp.int32) + (pos[:, 0] >= lines[1]).astype(jnp.int32)
    row = (pos[:, 1] >= lines[0]).astype(jnp.int32) + (pos[:, 1] >= lines[1]).astype(jnp.int32)
    out = jnp.asarray(_field_zone_ids())[col, row]
    boxes = zone_boxes()
    for kind in (ZONE_TOWN, ZONE_DUNGEON):   # later kinds win
        for i, z in enumerate(ZONES):
            if z[1] != kind:
                continue
            x0, y0, x1, y1 = boxes[i].tolist()
            inside = (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
            out = jnp.where(inside, i, out)
    return out.astype(jnp.int32)


def _road_segments():
    """Return [(axis, fixed, lo, hi)] centre lines of every road: axis 0 = runs along x at y = fixed."""
    centers = town_centers().tolist()
    pads = dungeon_pads().tolist()
    pairs = [(centers[a], centers[b]) for a, b in ROADS] + [(pads[k][0], centers[t]) for k, t in DUNGEON_ROADS]
    segs = []
    for (ax, ay), (bx, by) in pairs:   # same L as map_gen: along x at ay to bx, then along y at bx
        segs.append((0, ay, min(ax, bx), max(ax, bx)))
        segs.append((1, bx, min(ay, by), max(ay, by)))
    return segs


def zone_portals():
    """Return int32 [NP, 2, 2, 2]: per portal pair, two (warp tile, station tile) one-way legs across a rock line.

    Every road centre line that crosses a split line gets one pair (crossings of one line closer than
    PORTAL_MIN_GAP_TILES are merged). Leg 0 goes from the low side to the high side, leg 1 back: the
    warp stands 1 tile before the line, its station PORTAL_STEP_TILES past it.
    """
    s = PORTAL_STEP_TILES
    found = {}   # (line axis, line coord) -> sorted crossing coords
    for axis, fixed, lo, hi in _road_segments():
        for line in zone_line_tiles():
            if lo < line < hi:   # a segment along x crosses the vertical lines x = line (line axis 0)
                found.setdefault((axis, line), []).append(fixed)
    out = []
    for (axis, line), coords in sorted(found.items()):
        kept = []
        for c in sorted(set(coords)):
            if not kept or c - kept[-1] >= PORTAL_MIN_GAP_TILES:
                kept.append(c)
        for c in kept:
            def tile(v):   # v runs across the line; c along it
                return (v, c) if axis == 0 else (c, v)
            out.append(((tile(line - 1), tile(line + s)), (tile(line + 1), tile(line - s))))
    return np.array(out, np.int32).reshape(-1, 2, 2, 2)


def portal_warp_tiles():
    """Return (src int32 [2 NP, 2], dst int32 [2 NP, 2]) one-way zone-portal warps (leg order of zone_portals)."""
    p = zone_portals().reshape(-1, 2, 2)
    return p[:, 0], p[:, 1]


# Waybell teleport service (RO: a town's waybell sends you to another town for ryn). (from town, to town,
# ryn), TOWNS indices; arrival is the destination's town station tile. Prices are classic RO waybell
# fares (irowiki "Kafra Teleport Service"), rounded where the pair is not in the RO list.
WAYBELL_TELEPORT = (
    (0, 1, 1200), (0, 2, 1200), (0, 3, 1800), (0, 4, 1200),
    (1, 0, 1200), (1, 2, 1200), (1, 4, 1600),
    (2, 0, 1200), (2, 1, 1200), (2, 3, 1200),
    (3, 0, 1800), (3, 2, 1200),
    (4, 0, 1200), (4, 1, 1600),
)
WAYBELL_NO_ROUTE = -1


def waybell_teleport_cost():
    """Return int32 [NT, NT]: ryn to teleport from town i's waybell to town j, WAYBELL_NO_ROUTE where not offered."""
    c = np.full((len(TOWNS), len(TOWNS)), WAYBELL_NO_ROUTE, np.int32)
    for a, b, z in WAYBELL_TELEPORT:
        c[a, b] = z
    return c


def waybell_teleport_dest():
    """Return int32 [NT, 2]: arrival tile of a waybell teleport to each town (its town station)."""
    return town_station_tiles()


PORTAL_WARPS = tuple((_frac(a), _frac(b)) for a, b in zip(*(t.tolist() for t in portal_warp_tiles())))
WARPS = DUNGEON_WARPS + PORTAL_WARPS
assert len(WARPS) <= N_WARPS <= 64


def save_point_tiles():
    """Return int32 [NT, 2]: respawn tile after saving at each town's waybell (the temple centre, as ro_waybell.phase)."""
    return town_centers()


# ----------------------------------------------------------------------------------------------
# Town layouts (ro-world2). Every town keeps its centred temple square (spawn), waybell and station
# tiles; on top of that: a paved plaza around the temple, cross streets to the wall, a row of T_SHOP
# tiles beside the east street, and decoration that only ever replaces T_FLOOR (never a road, so
# gates and through-roads stay open). map_gen order: decor, paving, fountains, special tiles. Solmera: 5-wide streets, four fountain basins at the plaza
# corners, the castle block north-west, a long shop row. Lunavell: a moat around the plaza crossed by
# the streets. Pinehollow: trees dotted between the streets. Tidewick: a harbour basin south-east.
# Kharsand: house blocks on a grid and an oasis north-east.
# ----------------------------------------------------------------------------------------------
PLAZA_MARGIN_TILES = 6                    # plaza half-side = TEMPLE_TILES // 2 + this
TOWN_STREET_HALF = (2, 1, 1, 1, 1)        # cross-street half-width per town (Solmera 5 wide, others 3)
SHOP_ROW_TILES = (12, 4, 4, 6, 4)         # shop tiles per town, in one row beside the east street
SHOP_GAP_TILES = 4                        # the shop row starts this far east of the plaza edge (past Lunavell's moat)
assert len(TOWN_STREET_HALF) == len(SHOP_ROW_TILES) == len(TOWNS)


def plaza_half():
    return TEMPLE_TILES // 2 + PLAZA_MARGIN_TILES


def town_paving():
    """Return [(town, x0, y0, x1, y1)] inclusive road rects: each town's plaza and its two cross streets.

    Streets stop one tile inside the wall ring, so they open no new gates.
    """
    out, ph = [], plaza_half()
    for t, ((x0, y0, x1, y1), (cx, cy)) in enumerate(zip(town_boxes().tolist(), town_centers().tolist())):
        sh = TOWN_STREET_HALF[t]
        out += [(t, cx - ph, cy - ph, cx + ph, cy + ph),
                (t, cx - sh, y0 + 1, cx + sh, y1 - 1),
                (t, x0 + 1, cy - sh, x1 - 1, cy + sh)]
    return out


def town_decor():
    """Return [(town, kind, x0, y0, x1, y1, period, size)] decoration rects; kind is "wall", "water" or "tree".

    A tile of the rect is decorated when (x - x0) % period < size and (y - y0) % period < size (period 1,
    size 1 = solid). map_gen applies them in order, only over T_FLOOR, before the paving (which wins).
    """
    ph, out = plaza_half(), []
    boxes, centers = town_boxes().tolist(), town_centers().tolist()
    for t, ((x0, y0, x1, y1), (cx, cy)) in enumerate(zip(boxes, centers)):
        sh = TOWN_STREET_HALF[t]
        if t == 0:      # Solmera: castle block NW (the fountains are town_fountains), then the PvP arena ring SW
            out.append((t, "wall", x0 + 4, y0 + 4, cx - sh - 3, cy - ph - 4, 1, 1))
            out += [(t, "wall", ax0, ay0, ax1, ay1, 1, 1) for ax0, ay0, ax1, ay1 in arena_walls()]  # ro4-pvp
        elif t == 1:    # Lunavell: a 2-wide moat ring just outside the plaza (the streets bridge it)
            a, b = ph + 1, ph + 2
            out += [(t, "water", cx - b, cy - b, cx + b, cy - a, 1, 1), (t, "water", cx - b, cy + a, cx + b, cy + b, 1, 1),
                    (t, "water", cx - b, cy - b, cx - a, cy + b, 1, 1), (t, "water", cx + a, cy - b, cx + b, cy + b, 1, 1)]
        elif t == 2:    # Pinehollow: single trees every 4 tiles
            out.append((t, "tree", x0 + 2, y0 + 2, x1 - 2, y1 - 2, 4, 1))
        elif t == 3:    # Tidewick: harbour basin in the south-east quadrant
            out.append((t, "water", cx + ph + 3, cy + sh + 3, x1 - 3, y1 - 3, 1, 1))
        else:           # Kharsand: oasis north-east, 3 x 3 houses every 7 tiles around it
            out.append((t, "water", cx + ph + 3, y0 + 4, x1 - 4, cy - sh - 4, 1, 1))
            out.append((t, "wall", x0 + 3, y0 + 3, x1 - 3, y1 - 3, 7, 3))
    return out


def town_fountains():
    """Return [(town, x0, y0, x1, y1)] 2 x 2 water basins at Solmera's plaza corners, applied over the paving."""
    (cx, cy), ph, out = town_centers()[0].tolist(), plaza_half(), []
    for sx in (-1, 1):
        for sy in (-1, 1):
            bx, by = cx + sx * (ph - 3), cy + sy * (ph - 3)
            out.append((0, min(bx, bx + sx), min(by, by + sy), max(bx, bx + sx), max(by, by + sy)))
    return out


def shop_tiles():
    """Return (int32 [NS, 2] shop tiles, int32 [NS] their town): one row north of each town's east street."""
    tiles, town = [], []
    for t, (cx, cy) in enumerate(town_centers().tolist()):
        x = cx + plaza_half() + SHOP_GAP_TILES
        y = cy - TOWN_STREET_HALF[t] - 1
        for i in range(SHOP_ROW_TILES[t]):
            tiles.append((x + i, y))
            town.append(t)
    return np.array(tiles, np.int32), np.array(town, np.int32)


def _check_shops_inside():
    tiles, town = shop_tiles()
    b = town_boxes()[town]
    assert ((tiles[:, 0] > b[:, 0] + 1) & (tiles[:, 0] < b[:, 2] - 1)).all(), "shop row runs into the wall"


_check_shops_inside()


# ----------------------------------------------------------------------------------------------
# PvP (ro4-pvp). Every zone carries a PvP flag (`pvp_zone`): towns never, dungeons never, fields only when
# PVP_FIELDS is on (off in v1, where a field fight would also need both players PK-enabled). The arena is a
# walled ARENA_SIDE square in the south-west quadrant of Solmera (between the plaza and the south wall, west of
# the north-south street) with an ARENA_GATE_TILES gap in its north wall: PvP is always on inside it, PK flag or
# not. It is drawn as town decor (four wall rects, appended after the castle) and stays part of the Solmera zone.
# ----------------------------------------------------------------------------------------------
PVP_FIELDS = False         # v1: no open-field PvP
ARENA_TOWN = 0             # TOWNS index (Solmera)
ARENA_SIDE = 16            # wall ring side in tiles (interior ARENA_SIDE - 2 square)
ARENA_WALL_GAP = 3         # tiles between the town wall ring and the arena's west wall
ARENA_PLAZA_GAP = 1        # floor rows between the plaza edge and the arena's north wall
ARENA_GATE_TILES = 2       # gate width, at the east end of the north wall (the side facing the temple)


def pvp_zone():
    """Return bool [N_ZONES]: PvP allowed in this zone (outside the arena; both players must be PK-enabled)."""
    return np.array([kind == ZONE_FIELD and PVP_FIELDS for _, kind, *_ in ZONES], bool)


def arena_ring():
    """Return (x0, y0, x1, y1) inclusive bounds of the arena's wall ring."""
    bx0 = int(town_boxes()[ARENA_TOWN][0])
    cy = int(town_centers()[ARENA_TOWN][1])
    x0 = bx0 + 1 + ARENA_WALL_GAP
    y0 = cy + plaza_half() + 1 + ARENA_PLAZA_GAP
    return x0, y0, x0 + ARENA_SIDE - 1, y0 + ARENA_SIDE - 1


def arena_box():
    """Return int32 [4]: inclusive (x0, y0, x1, y1) of the arena floor (inside the ring): PvP always on here."""
    x0, y0, x1, y1 = arena_ring()
    return np.array([x0 + 1, y0 + 1, x1 - 1, y1 - 1], np.int32)


def arena_gate():
    """Return int32 [ARENA_GATE_TILES, 2]: the gate tiles in the north wall (walkable floor)."""
    x0, y0, x1, _ = arena_ring()
    return np.array([(x1 - ARENA_GATE_TILES + k, y0) for k in range(ARENA_GATE_TILES)], np.int32)


def arena_walls():
    """Return [(x0, y0, x1, y1)] inclusive wall rects of the ring, the gate left open."""
    x0, y0, x1, y1 = arena_ring()
    gx0 = x1 - ARENA_GATE_TILES
    return [(x0, y0, gx0 - 1, y0), (x1, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)]


def in_arena(pos):
    """Return bool [N]: each tile pos int [N, 2] is on the arena floor."""
    x0, y0, x1, y1 = arena_box().tolist()
    return (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)


def _check_arena_inside():
    x0, y0, x1, y1 = arena_ring()
    bx0, by0, bx1, by1 = town_boxes()[ARENA_TOWN].tolist()
    cx = int(town_centers()[ARENA_TOWN][0])
    assert x0 > bx0 + 1 and y1 < by1 - 1, "arena runs into the town wall"
    assert x1 < cx - TOWN_STREET_HALF[ARENA_TOWN], "arena runs into the north-south street"
    assert y0 > int(town_centers()[ARENA_TOWN][1]) + plaza_half(), "arena runs into the plaza"


_check_arena_inside()
