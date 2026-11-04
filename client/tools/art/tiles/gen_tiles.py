"""Procedural terrain tileset and map props in the chibi cel look of the character sheets (Pillow, fixed seed).

    python client/tools/art/tiles/gen_tiles.py [--out client/assets/tiles]

Writes `terrain.png` (13 columns = terrain ids x 48 rows, 64 px cells), `props.png` (64 px grid, multi-cell props)
and `tiles.json` (the index `client/game/world/map_renderer.gd` reads). Every pixel is drawn here: no source
images, nothing from any game. The look follows `client/tools/art` (render_kind / blender_render): light from the
upper left, three flat tones per material (shadow / mid / light, like the 0.55 / 0.8 / 1.0 ramp), a hard white
highlight spot, near-black ink outlines on props, bright lifted colours. Drawn at 4x and downsampled (LANCZOS).

Terrain rows per column (see tiles.json "terrains"):
  blob47   row = index in BLOB47_MASKS of the 8-neighbour mask (N=1 E=2 S=4 W=8 NE=16 SE=32 SW=64 NW=128, a
           diagonal kept only when both its sides connect): road, water, wall. Rims follow a distance field, so
           inner corners get a rounded rim and outer corners a rounded water / road edge (no staircase).
  blob4    row = 4-neighbour mask of connecting terrains (N=1, E=2, S=4, W=8): temple
  variants row = a weighted pick from a per-tile hash: grass, floor, rock, cave
  single   row 0: tree (base under the canopy prop), warp, waybell, station, shop
Props sit on a second layer, may overhang the tile above (trees, bell post, stalls) and one 2x2 fountain marks the
centre of every temple square. `furnace` and `fire` are drawn for the crafting-station node entities (406, 407).
"""
import argparse
import json
import math
import os
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "assets", "tiles"))

T = 64            # output cell px (2x the default tile_px 32, like the 2x character sheets)
SS = 4            # supersample factor
W = T * SS        # drawing size of one cell
ROWS = 48         # 47 blob rows (+1 spare)
N_TERRAIN = 13
NAMES = ("grass", "road", "floor", "temple", "water", "tree", "wall", "rock",
         "warp", "waybell", "station", "cave", "shop")
INK = (46, 32, 52, 255)
INK_SOFT = (70, 54, 70, 255)
WHITE = (255, 255, 255, 255)

# Material palettes: (light, mid, shadow); bright and lifted like the character `lift`.
GRASS = ((156, 230, 104), (118, 206, 88), (84, 168, 74))
DIRT = ((240, 214, 160), (222, 190, 136), (184, 146, 100))
FLOOR = ((244, 234, 206), (228, 214, 182), (196, 178, 146))
MARBLE = ((252, 252, 255), (236, 238, 248), (200, 206, 226))
GOLD = ((255, 232, 140), (240, 196, 84), (196, 142, 52))
WATER = ((150, 220, 255), (72, 164, 238), (44, 118, 214))
SAND = ((250, 240, 204), (236, 220, 172), (204, 184, 136))
STONE = ((222, 214, 228), (196, 186, 204), (150, 138, 164))
BRICK = ((176, 162, 186), (150, 136, 160), (114, 100, 128))
BOULDER = ((222, 218, 236), (178, 172, 196), (124, 118, 146))
CAVE = ((158, 128, 108), (130, 102, 86), (100, 76, 66))
LEAF = (((150, 226, 96), (98, 192, 74), (56, 140, 66)),
        ((196, 228, 92), (148, 200, 70), (98, 152, 60)),
        ((118, 210, 120), (70, 170, 96), (40, 118, 82)))
BARK = ((178, 122, 78), (142, 92, 58), (100, 62, 44))
WOOD = ((226, 172, 110), (196, 138, 84), (148, 96, 60))
AWNING = ((236, 72, 84), (64, 132, 236), (72, 190, 96), (246, 186, 50))


def rgba(c, a=255):
    return (int(c[0]), int(c[1]), int(c[2]), a)


def new_cell(w=1, h=1, fill=None):
    return Image.new("RGBA", (W * w, W * h), rgba(fill) if fill else (0, 0, 0, 0))


def new_mask(img):
    return Image.new("L", img.size, 0)


def shifted(mask, dx, dy):
    out = Image.new("L", mask.size, 0)
    out.paste(mask, (int(dx), int(dy)))
    return out


def dilate(mask, r):
    size = max(3, int(r) * 2 + 1)
    return mask.filter(ImageFilter.MaxFilter(size if size % 2 else size + 1))


def cel(img, mask, pal, k=None, ink=3.0, highlight=None, light_k=None):
    """Paint `mask` with three flat tones lit from the upper left plus an ink outline (in output px)."""
    box = mask.getbbox()
    if box is None:
        return
    span = min(box[2] - box[0], box[3] - box[1])
    k = k if k is not None else max(SS, span * 0.16)
    lk = light_k if light_k is not None else k * 0.55
    if ink > 0:
        img.paste(INK, (0, 0), dilate(mask, ink * SS))
    img.paste(rgba(pal[1]), (0, 0), mask)
    shadow = ImageChops.subtract(mask, shifted(mask, -k, -k))
    img.paste(rgba(pal[2]), (0, 0), shadow)
    light = ImageChops.subtract(ImageChops.subtract(mask, shifted(mask, lk, lk)), shadow)
    img.paste(rgba(pal[0]), (0, 0), light)
    if highlight is not None:
        d = ImageDraw.Draw(img)
        x, y, r = highlight
        d.ellipse((x - r, y - r * 0.7, x + r, y + r * 0.7), fill=WHITE)


def ellipse_mask(img, box):
    m = new_mask(img)
    ImageDraw.Draw(m).ellipse(box, fill=255)
    return m


def poly_mask(img, pts):
    m = new_mask(img)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return m


def down(img):
    return img.resize((img.width // SS, img.height // SS), Image.LANCZOS)


def harden(img):
    r, g, b, a = img.split()
    a = a.point(lambda v: 255 if v >= 120 else 0)
    return Image.merge("RGBA", (r, g, b, a))


# ---------------------------------------------------------------- ground details

def tuft(d, x, y, s, pal):
    """Three grass blades with a dark root, s in supersampled px."""
    for dx, h, lean in ((-s * 0.55, s * 0.9, -0.35), (0, s * 1.25, 0.05), (s * 0.55, s * 0.85, 0.4)):
        bx = x + dx
        d.polygon([(bx - s * 0.22, y), (bx + s * 0.22, y), (bx + lean * s, y - h)], fill=rgba(pal[2]))
        d.polygon([(bx - s * 0.08, y - s * 0.1), (bx + s * 0.12, y - s * 0.1), (bx + lean * s, y - h)],
                  fill=rgba(pal[0]))


def flower(d, x, y, s, col):
    for k in range(5):
        a = k * 2 * math.pi / 5 - math.pi / 2
        px, py = x + math.cos(a) * s, y + math.sin(a) * s * 0.85
        d.ellipse((px - s * 0.62, py - s * 0.62, px + s * 0.62, py + s * 0.62), fill=rgba(col))
    d.ellipse((x - s * 0.45, y - s * 0.45, x + s * 0.45, y + s * 0.45), fill=(255, 214, 70, 255))


def pebble(img, x, y, rx, ry, pal):
    m = ellipse_mask(img, (x - rx, y - ry, x + rx, y + ry))
    cel(img, m, pal, ink=1.2, k=max(2, ry * 0.5))


def grass_tile(rng, kind):
    img = new_cell(fill=GRASS[1])
    d = ImageDraw.Draw(img)
    m = 20  # keep details off the edges so the tiles join without seams
    if kind in (4, 5):  # soft lighter patch of clover
        cx, cy = rng.uniform(80, 176), rng.uniform(80, 176)
        for _ in range(5):
            ox, oy, r = cx + rng.uniform(-34, 34), cy + rng.uniform(-24, 24), rng.uniform(16, 26)
            d.ellipse((ox - r, oy - r * 0.7, ox + r, oy + r * 0.7), fill=rgba((134, 216, 94)))
    n = (2, 3, 4, 3, 2, 2, 2, 2)[kind]
    for _ in range(n):
        tuft(d, rng.uniform(m + 14, W - m - 14), rng.uniform(m + 40, W - m), rng.uniform(20, 28), GRASS)
    for _ in range(4):  # short darker strokes give the field a painted texture
        x, y = rng.uniform(m, W - m), rng.uniform(m, W - m)
        d.line([(x, y), (x + rng.uniform(-6, 6), y - rng.uniform(10, 16))], fill=rgba((100, 186, 80)), width=int(1.4 * SS))
    if kind == 6:
        cols = ((255, 255, 255), (255, 150, 190), (255, 236, 110))
        for _ in range(3):
            flower(d, rng.uniform(m + 14, W - m - 14), rng.uniform(m + 14, W - m - 14), rng.uniform(13, 16),
                   rng.choice(cols))
    if kind == 7:
        pebble(img, rng.uniform(90, 166), rng.uniform(100, 170), 22, 15, STONE)
    return img


def grass_base():
    return grass_tile(random.Random(7), 0)


def cobbles(img, pal, size, seed):
    """A seamless cobble field: rows of rounded stones, period dividing the cell."""
    rng = random.Random(seed)
    d = ImageDraw.Draw(img)
    rows = W // size
    for r in range(rows + 1):
        off = (size // 2) if r % 2 else 0
        for c in range(-1, rows + 1):
            x0, y0 = c * size + off, r * size
            g = size * 0.1
            box = (x0 + g, y0 + g, x0 + size - g, y0 + size - g)
            tint = rng.uniform(-8, 8)
            mid = tuple(min(255, max(0, v + tint)) for v in pal[1])
            d.rounded_rectangle(box, radius=size * 0.32, fill=rgba(pal[2]))
            d.rounded_rectangle((box[0], box[1], box[2] - size * 0.08, box[3] - size * 0.1), radius=size * 0.3,
                                fill=rgba(mid))
            d.rounded_rectangle((box[0] + size * 0.12, box[1] + size * 0.1, box[0] + size * 0.45,
                                 box[1] + size * 0.24), radius=size * 0.07, fill=rgba(pal[0]))


def edge_band(img, mask_bits, width, pal, ink=True, inner_line=None):
    """Draw a band along the sides whose bit is NOT set (N=1, E=2, S=4, W=8)."""
    d = ImageDraw.Draw(img)
    sides = []
    if not mask_bits & 1:
        sides.append((0, 0, W, width))
    if not mask_bits & 4:
        sides.append((0, W - width, W, W))
    if not mask_bits & 8:
        sides.append((0, 0, width, W))
    if not mask_bits & 2:
        sides.append((W - width, 0, W, W))
    for box in sides:
        d.rectangle(box, fill=rgba(pal[1]))
    # light on the inner top-left facing parts, shadow on the bottom-right outer parts
    lw = max(SS, width // 4)
    if not mask_bits & 1:
        d.rectangle((0, width - lw, W, width), fill=rgba(pal[2]))
    if not mask_bits & 8:
        d.rectangle((width - lw, 0, width, W), fill=rgba(pal[2]))
    if not mask_bits & 4:
        d.rectangle((0, W - width, W, W - width + lw), fill=rgba(pal[0]))
    if not mask_bits & 2:
        d.rectangle((W - width, 0, W - width + lw, W), fill=rgba(pal[0]))
    if inner_line is not None:
        col, off, lwid = inner_line
        if not mask_bits & 1:
            d.rectangle((0, width + off, W, width + off + lwid), fill=rgba(col))
        if not mask_bits & 4:
            d.rectangle((0, W - width - off - lwid, W, W - width - off), fill=rgba(col))
        if not mask_bits & 8:
            d.rectangle((width + off, 0, width + off + lwid, W), fill=rgba(col))
        if not mask_bits & 2:
            d.rectangle((W - width - off - lwid, 0, W - width - off, W), fill=rgba(col))
    if ink:
        iw = int(2.2 * SS)
        if not mask_bits & 1:
            d.rectangle((0, 0, W, iw), fill=INK)
        if not mask_bits & 4:
            d.rectangle((0, W - iw, W, W), fill=INK)
        if not mask_bits & 8:
            d.rectangle((0, 0, iw, W), fill=INK)
        if not mask_bits & 2:
            d.rectangle((W - iw, 0, W, W), fill=INK)



# ---------------------------------------------------------------- 8-neighbour blob (47 tiles)

B_N, B_E, B_S, B_WE, B_NE, B_SE, B_SW, B_NW = 1, 2, 4, 8, 16, 32, 64, 128


def blob_reduce(m):
    """Drop a diagonal bit unless both of its sides connect (the 256 raw masks fold onto 47)."""
    for diag, a, b in ((B_NE, B_N, B_E), (B_SE, B_S, B_E), (B_SW, B_S, B_WE), (B_NW, B_N, B_WE)):
        if m & diag and not (m & a and m & b):
            m &= ~diag
    return m


BLOB47_MASKS = sorted({blob_reduce(m) for m in range(256)})
assert len(BLOB47_MASKS) == 47

_GRID = (np.arange(-1, W + 1, dtype=np.float64) + 0.5)
_GX, _GY = np.meshgrid(_GRID, _GRID)  # padded by one px for the gradient


def blob_field(m, ro):
    """Distance (supersampled px) from each pixel to the open boundary of mask `m`, on a (W+2)^2 grid.

    Open sides are lines on the cell edge; an inner corner (both sides connect, the diagonal does not) is the corner
    point, so the rim bends round it in a quarter circle; an outer corner (both sides open) is rounded with radius
    `ro` (values below 0 there are outside the rounded edge). Both tiles of a shared edge see the same boundary
    near it, so neighbours join without seams."""
    x, y = _GX, _GY
    d = np.full(x.shape, 1e9)
    if not m & B_N:
        d = np.minimum(d, y)
    if not m & B_S:
        d = np.minimum(d, W - y)
    if not m & B_WE:
        d = np.minimum(d, x)
    if not m & B_E:
        d = np.minimum(d, W - x)
    corners = ((W, 0, B_N, B_E, B_NE, -1, 1), (W, W, B_S, B_E, B_SE, -1, -1),
               (0, W, B_S, B_WE, B_SW, 1, -1), (0, 0, B_N, B_WE, B_NW, 1, 1))
    for vx, vy, a, b, diag, sx, sy in corners:
        if m & a and m & b and not m & diag:
            d = np.minimum(d, np.hypot(x - vx, y - vy))
        elif not m & a and not m & b and ro > 0:
            qx, qy = vx + sx * ro, vy + sy * ro
            region = ((x - qx) * sx < 0) & ((y - qy) * sy < 0)
            d = np.where(region, np.minimum(d, ro - np.hypot(x - qx, y - qy)), d)
    return d


def _mask_img(sel):
    return Image.fromarray(np.where(sel[1:-1, 1:-1], 255, 0).astype(np.uint8), "L")


def rim_tile(img, m, rim, ro, pal, ink=True, bands=(), outside=None):
    """Paint the rim of a blob47 tile: `pal` (light, mid, shadow) band of width `rim` on the open boundary, a
    shade line on its inner side (shadow where the rim is on a top or left side, light on a bottom or right side,
    as the 4-neighbour bands did), extra `bands` [(from, to, colour)] inside it, and an ink line on the cell edge.
    With `outside` (a colour) the rounded outer corners are cut out: the wedge beyond the curve takes that colour
    (a light beach sand for water, which sits well on grass, flagstones and rock) and the ink follows the curve,
    so a shore reads round, not stepped."""
    d = blob_field(m, ro)
    for lo, hi, col in bands:
        img.paste(rgba(col), (0, 0), _mask_img((d >= lo) & (d < hi)))
    img.paste(rgba(pal[1]), (0, 0), _mask_img(d < rim))
    gy, gx = np.gradient(d)
    top_left = (gx + gy) > 0  # the inward normal points down/right
    lw = max(SS, rim // 4)
    line = (d >= rim - lw) & (d < rim)
    img.paste(rgba(pal[2]), (0, 0), _mask_img(line & top_left))
    img.paste(rgba(pal[0]), (0, 0), _mask_img(line & ~top_left))
    if ink:
        e = d if outside is not None else blob_field(m, 0)  # else square: ink on the cell edge
        img.paste(INK, (0, 0), _mask_img(e < int(2.2 * SS)))
    if outside is not None:
        img.paste(rgba(outside), (0, 0), _mask_img(d < 0))
    return d

KERB = ((226, 218, 230), (200, 190, 208), (150, 138, 164))  # a pale stone kerb that reads against the cobbles
ROAD_RIM, ROAD_RO = 7 * SS, 26 * SS


def road_tile(m8):
    """Blob47 road: cobbles with a kerb on the open sides, rounded on outer corners and bends."""
    img = new_cell(fill=DIRT[1])
    cobbles(img, DIRT, W // 4, 11)
    rim_tile(img, m8, ROAD_RIM, ROAD_RO, KERB, outside=KERB[0])
    return img


def floor_tile(rng, kind):
    img = new_cell(fill=FLOOR[1])
    d = ImageDraw.Draw(img)
    g = int(1.2 * SS)
    # one big flagstone per tile, rows offset by half a tile (the grout of the row is on the top edge)
    tint = (0, 6, -5, 3)[kind]
    mid = tuple(min(255, max(0, v + tint)) for v in FLOOR[1])
    d.rectangle((0, 0, W, W), fill=rgba(mid))
    d.rectangle((g, g, W, g * 3), fill=rgba(FLOOR[0]))
    d.rectangle((g, g, g * 3, W), fill=rgba(FLOOR[0]))
    d.rectangle((W - g * 2, g, W, W), fill=rgba(FLOOR[2]))
    d.rectangle((g, W - g * 2, W, W), fill=rgba(FLOOR[2]))
    d.rectangle((0, 0, W, g), fill=rgba((180, 160, 128)))
    d.rectangle((0, 0, g, W), fill=rgba((180, 160, 128)))
    if kind == 1:  # a crack
        x, y = rng.uniform(60, 120), rng.uniform(150, 200)
        pts = [(x, y)]
        for _ in range(4):
            x, y = x + rng.uniform(8, 18), y + rng.uniform(-10, 10)
            pts.append((x, y))
        d.line(pts, fill=rgba(FLOOR[2]), width=int(1.4 * SS))
    if kind == 2:  # a little moss
        tuft(d, rng.uniform(150, 200), rng.uniform(200, 230), 10, GRASS)
    if kind == 3:  # a small four-petal inlay
        c = W / 2
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            d.ellipse((c + dx * 14 - 9, c + dy * 14 - 9, c + dx * 14 + 9, c + dy * 14 + 9), fill=rgba(FLOOR[2]))
        d.ellipse((c - 7, c - 7, c + 7, c + 7), fill=rgba(GOLD[1]))
    return img


def temple_tile(mask_bits):
    img = new_cell(fill=MARBLE[1])
    d = ImageDraw.Draw(img)
    half = W // 2
    g = int(1.0 * SS)
    for sy in (0, 1):
        for sx in (0, 1):
            x0, y0 = sx * half, sy * half
            d.rectangle((x0, y0, x0 + half, y0 + half), fill=rgba(MARBLE[1] if (sx + sy) % 2 == 0 else (226, 230, 246)))
            d.rectangle((x0 + g, y0 + g, x0 + half, y0 + g * 3), fill=rgba(MARBLE[0]))
            d.rectangle((x0, y0, x0 + half, y0 + g), fill=rgba(MARBLE[2]))
            d.rectangle((x0, y0, x0 + g, y0 + half), fill=rgba(MARBLE[2]))
    # a gold diamond inlay on the shared corner of the four slabs
    c = half
    r = 6 * SS  # a small pale inlay where four slabs meet
    d.polygon([(c, c - r), (c + r, c), (c, c + r), (c - r, c)], fill=rgba((214, 220, 240)))
    d.polygon([(c, c - r + 6), (c + r - 6, c), (c, c + r - 6), (c - r + 6, c)], fill=rgba((248, 230, 170)))
    edge_band(img, mask_bits, 6 * SS, GOLD)
    return img


WATER_RIM, WATER_RO = 8 * SS, 30 * SS


def water_tile(mask_bits, rng):
    img = new_cell(fill=WATER[1])
    d = ImageDraw.Draw(img)
    for _ in range(3):  # short bright wave arcs
        x, y = rng.uniform(50, 200), rng.uniform(60, 210)
        r = rng.uniform(18, 26)
        d.arc((x - r, y - r * 0.5, x + r, y + r * 0.5), 200, 340, fill=rgba(WATER[0]), width=int(1.6 * SS))
    for _ in range(2):
        x, y = rng.uniform(60, 196), rng.uniform(60, 196)
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=WHITE)
    # deeper water under the rim, a white foam line, then the sand rim with ink; rounded shores
    foam = int(1.6 * SS)
    rim_tile(img, mask_bits, WATER_RIM, WATER_RO, SAND, outside=SAND[0],
             bands=((WATER_RIM, WATER_RIM + 6 * SS, WATER[2]), (WATER_RIM, WATER_RIM + foam, (240, 252, 255))))
    return img


def wall_tile(m8):
    """Blob47 wall: the 4-neighbour wall of the low bits, plus inner corners (both sides wall, the diagonal open)
    that close the ink outline: a corner square on top, and on the bottom an ink edge beside the neighbour's brick
    front face."""
    img = wall_tile4(m8 & 15)
    d = ImageDraw.Draw(img)
    iw = int(2.2 * SS)
    fy = int(W * 0.56)
    if m8 & B_N and m8 & B_E and not m8 & B_NE:
        d.rectangle((W - iw, 0, W, iw), fill=INK)
    if m8 & B_N and m8 & B_WE and not m8 & B_NW:
        d.rectangle((0, 0, iw, iw), fill=INK)
    if m8 & B_S and m8 & B_E and not m8 & B_SE:
        d.rectangle((W - iw, fy, W, W), fill=INK)
        d.rectangle((W - iw - SS * 1.5, fy, W - iw, W), fill=rgba(STONE[2]))
    if m8 & B_S and m8 & B_WE and not m8 & B_SW:
        d.rectangle((0, fy, iw, W), fill=INK)
        d.rectangle((iw, fy, iw + SS * 1.5, W), fill=rgba(STONE[0]))
    return img


def wall_tile4(mask_bits):
    """Top face everywhere; a brick front face on the lower part when there is no wall to the south."""
    img = new_cell(fill=STONE[1])
    d = ImageDraw.Draw(img)
    front = not mask_bits & 4
    top_h = int(W * 0.56) if front else W
    # top face: big slabs with soft joints
    for y in range(0, top_h, W // 2):
        d.line([(0, y), (W, y)], fill=rgba(STONE[2]), width=SS)
        d.line([(0, y + SS * 1.5), (W, y + SS * 1.5)], fill=rgba(STONE[0]), width=SS)
    d.line([(W // 2, 0), (W // 2, min(top_h, W // 2))], fill=rgba(STONE[2]), width=SS)
    if top_h > W // 2:
        d.line([(W // 4, W // 2), (W // 4, top_h)], fill=rgba(STONE[2]), width=SS)
    if front:
        fy = top_h
        d.rectangle((0, fy, W, W), fill=rgba(BRICK[1]))
        bh = (W - fy) // 3
        for r in range(3):
            y0 = fy + r * bh
            off = (W // 8) if r % 2 else 0
            d.line([(0, y0), (W, y0)], fill=rgba(BRICK[2]), width=int(1.2 * SS))
            for x in range(-W // 4 + off, W, W // 4):
                d.line([(x, y0), (x, y0 + bh)], fill=rgba(BRICK[2]), width=int(1.2 * SS))
                d.line([(x + SS * 1.5, y0 + SS * 1.5), (x + W // 4 - SS * 2, y0 + SS * 1.5)], fill=rgba(BRICK[0]),
                       width=SS)
        d.rectangle((0, fy, W, fy + int(2.2 * SS)), fill=INK)
        d.rectangle((0, fy + int(2.2 * SS), W, fy + int(3.6 * SS)), fill=rgba(BRICK[0]))
    iw = int(2.2 * SS)
    if not mask_bits & 1:
        d.rectangle((0, 0, W, iw), fill=INK)
        d.rectangle((0, iw, W, iw + SS * 1.5), fill=rgba(STONE[0]))
    if not mask_bits & 8:
        d.rectangle((0, 0, iw, W), fill=INK)
        d.rectangle((iw, 0, iw + SS * 1.5, top_h), fill=rgba(STONE[0]))
    if not mask_bits & 2:
        d.rectangle((W - iw, 0, W, W), fill=INK)
        d.rectangle((W - iw - SS * 1.5, 0, W - iw, top_h), fill=rgba(STONE[2]))
    if front:
        d.rectangle((0, W - iw, W, W), fill=INK)
    return img


def tree_base():
    img = grass_tile(random.Random(21), 0)
    d = ImageDraw.Draw(img)
    d.ellipse((30, 150, W - 30, 236), fill=rgba((84, 168, 74)))
    return img


def tree_planter():
    """A town tree (next to flagstones, the `near` rule): a flagstone with a round stone-edged bed of soil and moss
    under the trunk, instead of a square of grass."""
    img = floor_tile(random.Random(700), 0)
    d = ImageDraw.Draw(img)
    c = W / 2
    d.ellipse((c - 104, 132, c + 104, 250), fill=INK)
    d.ellipse((c - 96, 138, c + 96, 242), fill=rgba(STONE[1]))
    d.ellipse((c - 96, 138, c + 90, 234), fill=rgba(STONE[0]))
    d.ellipse((c - 80, 150, c + 80, 232), fill=rgba((120, 86, 64)))
    d.ellipse((c - 80, 150, c + 70, 220), fill=rgba((142, 104, 76)))
    tuft(d, c - 50, 214, 12, GRASS)
    tuft(d, c + 46, 206, 11, GRASS)
    return img


def boulder(img, cx, cy, rx, ry, rng):
    pts = []
    n = 9
    for k in range(n):
        a = k * 2 * math.pi / n + rng.uniform(-0.15, 0.15)
        rr = rng.uniform(0.82, 1.0)
        pts.append((cx + math.cos(a) * rx * rr, cy + math.sin(a) * ry * rr))
    m = poly_mask(img, pts)
    m = m.filter(ImageFilter.GaussianBlur(5)).point(lambda v: 255 if v > 128 else 0)
    cel(img, m, BOULDER, ink=2.6, k=rx * 0.34, light_k=rx * 0.22,
        highlight=(cx - rx * 0.42, cy - ry * 0.45, rx * 0.12))
    d = ImageDraw.Draw(img)
    x, y = cx + rx * 0.1, cy - ry * 0.3
    d.line([(x, y), (x + rx * 0.15, y + ry * 0.3), (x + rx * 0.05, y + ry * 0.55)], fill=INK_SOFT, width=int(1.4 * SS))


def rock_tile(rng, kind, ground=None):
    """Boulders on the ground they stand in: grass in the fields, the cave floor next to a cave (the `near` rule),
    so a rock line or a dungeon pillar shows no grey square."""
    if ground == "cave":
        img = cave_tile(random.Random(600 + kind), 0)
        shade = CAVE[2]
    else:
        img = grass_tile(random.Random(520 + kind), 0)
        shade = GRASS[2]
    d = ImageDraw.Draw(img)
    d.ellipse((24, 150, W - 24, 246), fill=rgba(shade))  # contact shadow
    if kind == 0:
        boulder(img, 128, 136, 104, 92, rng)
    elif kind == 1:
        boulder(img, 100, 150, 76, 70, rng)
        boulder(img, 170, 110, 66, 60, rng)
    else:
        boulder(img, 132, 128, 110, 100, rng)
    return img


def cave_tile(rng, kind):
    img = new_cell(fill=CAVE[1])
    d = ImageDraw.Draw(img)
    for _ in range(3 + kind):
        x, y = rng.uniform(30, 226), rng.uniform(30, 226)
        d.ellipse((x - 12, y - 7, x + 12, y + 7), fill=rgba(CAVE[2]))
        d.ellipse((x - 10, y - 7, x + 6, y + 2), fill=rgba(CAVE[0]))
    if kind == 2:
        pebble(img, rng.uniform(80, 170), rng.uniform(80, 170), 16, 11, BOULDER)
    if kind == 3:  # a glowing crystal shard, bright accent in the dark
        x, y = rng.uniform(90, 160), rng.uniform(120, 180)
        for dx, h, wd in ((-26, 48, 14), (24, 40, 12), (0, 76, 22)):  # a cluster of three shards
            m = poly_mask(img, [(x + dx, y - h), (x + dx + wd, y), (x + dx, y + 10), (x + dx - wd, y)])
            cel(img, m, ((210, 250, 255), (120, 210, 250), (70, 140, 220)), ink=1.8, k=10)
    return img


def paved_base():
    return road_tile(255)


def warp_tile():
    img = paved_base()
    d = ImageDraw.Draw(img)
    c = W / 2
    glow = new_cell()
    gd = ImageDraw.Draw(glow)
    gd.ellipse((c - 116, c - 84, c + 116, c + 84), fill=(150, 120, 255, 150))
    glow = glow.filter(ImageFilter.GaussianBlur(14))
    img.alpha_composite(glow)
    d.ellipse((c - 104, c - 74, c + 104, c + 74), fill=INK)
    d.ellipse((c - 96, c - 67, c + 96, c + 67), fill=rgba((150, 104, 255)))
    d.ellipse((c - 80, c - 55, c + 80, c + 55), fill=rgba((96, 196, 255)))
    d.ellipse((c - 56, c - 38, c + 56, c + 38), fill=rgba((176, 240, 255)))
    d.ellipse((c - 26, c - 18, c + 26, c + 18), fill=WHITE)
    for k in range(4):  # swirl arcs
        a0 = k * 90
        d.arc((c - 76, c - 52, c + 76, c + 52), a0, a0 + 50, fill=WHITE, width=int(2 * SS))
    for k in range(6):
        a = k * math.pi / 3 + 0.3
        x, y = c + math.cos(a) * 88, c + math.sin(a) * 60
        d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=WHITE)
    return img


def waybell_tile():
    img = paved_base()
    d = ImageDraw.Draw(img)
    c = W / 2
    d.ellipse((c - 100, c - 70, c + 100, c + 70), fill=rgba(GOLD[2]))
    d.ellipse((c - 92, c - 63, c + 92, c + 63), fill=rgba(MARBLE[1]))
    d.ellipse((c - 92, c - 63, c + 80, c + 50), fill=rgba(MARBLE[0]))
    for k in range(8):
        a = k * math.pi / 4
        x, y = c + math.cos(a) * 78, c + math.sin(a) * 53
        d.ellipse((x - 7, y - 5, x + 7, y + 5), fill=rgba(GOLD[1]))
    return img


def station_tile():
    img = paved_base()
    d = ImageDraw.Draw(img)
    c = W / 2
    teal = ((190, 255, 240), (86, 214, 200), (40, 150, 150))
    d.ellipse((c - 102, c - 72, c + 102, c + 72), fill=rgba(teal[2]))
    d.ellipse((c - 94, c - 65, c + 94, c + 65), fill=rgba(teal[1]))
    d.ellipse((c - 74, c - 50, c + 74, c + 50), fill=rgba(FLOOR[0]))
    d.ellipse((c - 66, c - 45, c + 66, c + 45), outline=rgba(teal[1]), width=int(1.5 * SS))
    for k in range(8):
        a = k * math.pi / 4 + math.pi / 8
        x, y = c + math.cos(a) * 84, c + math.sin(a) * 58
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=rgba(teal[0]))
    s = 22
    d.polygon([(c, c - s), (c + s * 0.6, c), (c, c + s), (c - s * 0.6, c)], fill=rgba(teal[1]))
    d.polygon([(c, c - s), (c - s * 0.6, c), (c - 3, c - 3)], fill=rgba(teal[0]))
    return img


def shop_tile():
    img = paved_base()
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((26, 120, W - 26, 236), radius=16, fill=rgba((200, 90, 80)))
    d.rounded_rectangle((36, 130, W - 36, 226), radius=12, fill=rgba((232, 130, 100)))
    return img


# ---------------------------------------------------------------- props (64 px grid cells, 4x drawn)

def tree_prop(variant):
    """1x2 cells: trunk on the lower cell, a puffy canopy overhanging the cell above."""
    rng = random.Random(100 + variant)
    img = new_cell(1, 2)
    pal = LEAF[variant]
    d = ImageDraw.Draw(img)
    base_y = W + W * 0.9
    # trunk
    tm = poly_mask(img, [(W / 2 - 26, base_y), (W / 2 + 26, base_y), (W / 2 + 18, W + 60), (W / 2 - 18, W + 60)])
    cel(img, tm, BARK, ink=2.4, k=14)
    # canopy: overlapping puffs filling the tile width (a forest reads as one canopy)
    m = new_mask(img)
    md = ImageDraw.Draw(m)
    cy = W * 1.02
    puffs = [(W / 2, cy - 84, 84), (W / 2 - 68, cy + 10, 58), (W / 2 + 68, cy + 6, 58), (W / 2, cy + 40, 84),
             (W / 2 - 46, cy - 46, 68), (W / 2 + 48, cy - 42, 68)]
    for x, y, r in puffs:
        r *= rng.uniform(0.92, 1.05)
        md.ellipse((x - r, y - r * 0.92, x + r, y + r * 0.92), fill=255)
    cel(img, m, pal, ink=2.6, k=38, light_k=22, highlight=(W / 2 - 50, cy - 80, 11))
    # leaf-clump shadow arcs inside the canopy
    for x, y, r in puffs[1:4]:
        d.arc((x - r * 0.7, y - r * 0.5, x + r * 0.7, y + r * 0.6), 20, 160, fill=rgba(pal[2]), width=int(1.6 * SS))
    if variant == 0:  # a few red fruits
        for x, y in ((W / 2 + 30, cy - 10), (W / 2 - 40, cy + 30), (W / 2 + 10, cy + 50)):
            d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=INK)
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(240, 70, 80, 255))
            d.ellipse((x - 5, y - 5, x - 1, y - 1), fill=WHITE)
    return img


def bell_prop():
    """1x2 cells: a wooden post with a little roof and a golden bell (the Waybell service)."""
    img = new_cell(1, 2)
    d = ImageDraw.Draw(img)
    c = W / 2
    foot = W + W * 0.72
    pm = poly_mask(img, [(c - 12, foot), (c + 12, foot), (c + 12, 70), (c - 12, 70)])
    cel(img, pm, WOOD, ink=2.4, k=10)
    am = poly_mask(img, [(c - 70, 120), (c + 70, 120), (c + 70, 136), (c - 70, 136)])
    cel(img, am, WOOD, ink=2.2, k=7)
    rm = poly_mask(img, [(c, 22), (c + 92, 88), (c + 80, 100), (c - 80, 100), (c - 92, 88)])
    cel(img, rm, ((120, 170, 255), (72, 120, 232), (44, 78, 180)), ink=2.6, k=16, light_k=10)
    d.line([(c + 46, 136), (c + 46, 156)], fill=INK, width=int(1.6 * SS))
    bm = new_mask(img)
    bd = ImageDraw.Draw(bm)
    bx, by = c + 46, 150
    bd.pieslice((bx - 34, by, bx + 34, by + 76), 180, 360, fill=255)
    bd.rectangle((bx - 34, by + 37, bx + 34, by + 62), fill=255)
    bd.ellipse((bx - 42, by + 52, bx + 42, by + 70), fill=255)
    cel(img, bm, GOLD, ink=2.6, k=16, light_k=10, highlight=(bx - 14, by + 24, 7))
    d.ellipse((bx - 8, by + 64, bx + 8, by + 80), fill=INK)
    # base stone
    sm = ellipse_mask(img, (c - 40, foot - 18, c + 40, foot + 18))
    cel(img, sm, STONE, ink=2.0, k=10)
    return img


def stall_prop(colour):
    """1x2 cells: a market stall whose striped awning reaches both edges so a row of shops reads as one market."""
    img = new_cell(1, 2)
    d = ImageDraw.Draw(img)
    c = W / 2
    for x in (22, W - 22):  # poles
        pm = poly_mask(img, [(x - 7, W + W * 0.8), (x + 7, W + W * 0.8), (x + 7, 120), (x - 7, 120)])
        cel(img, pm, WOOD, ink=2.0, k=6)
    cm = poly_mask(img, [(10, W + 100), (W - 10, W + 100), (W - 10, W + 196), (10, W + 196)])
    cel(img, cm, WOOD, ink=2.4, k=16)
    for k, col in enumerate(((240, 90, 80), (255, 210, 70), (120, 210, 90))):  # goods on the counter
        x = 52 + k * 76
        gm = ellipse_mask(img, (x - 26, W + 66, x + 26, W + 112))
        cel(img, gm, (tuple(min(255, v + 40) for v in col), col, tuple(int(v * 0.7) for v in col)), ink=1.8, k=10,
            highlight=(x - 9, W + 80, 5))
    # awning: slanted striped cloth, scalloped hem, edge to edge
    top, hem = 96, 176
    am = poly_mask(img, [(0, top), (W, top), (W, hem), (0, hem)])
    stripes = new_cell(1, 2)
    sd = ImageDraw.Draw(stripes)
    col = AWNING[colour]
    sw = W // 4
    for k in range(4):
        x0 = k * sw
        fill = col if k % 2 == 0 else (255, 250, 240)
        sd.rectangle((x0, top, x0 + sw, hem + 24), fill=rgba(fill))
    for k in range(4):  # scallops
        x0 = k * sw
        fill = col if k % 2 == 0 else (255, 250, 240)
        sd.ellipse((x0, hem - 14, x0 + sw, hem + 22), fill=rgba(fill))
    hm = new_mask(img)
    hd = ImageDraw.Draw(hm)
    hd.rectangle((0, top, W, hem), fill=255)
    for k in range(4):
        hd.ellipse((k * sw, hem - 14, k * sw + sw, hem + 22), fill=255)
    img.paste(INK, (0, 0), dilate(hm, 2.4 * SS))
    img.paste(stripes, (0, 0), hm)
    shade = ImageChops.subtract(hm, shifted(hm, 0, -26))
    dark = Image.new("RGBA", img.size, (40, 20, 60, 70))
    img.paste(Image.alpha_composite(img, dark), (0, 0), shade)
    d.rectangle((0, top + 4, W, top + 12), fill=(255, 255, 255, 150))
    del am
    return img


def fountain_prop():
    """2x2 cells: a round stone basin with water and a spouting centre pillar, for the temple square."""
    img = new_cell(2, 2)
    d = ImageDraw.Draw(img)
    c = W  # centre of the 2x2
    cy = W + 40
    bm = ellipse_mask(img, (c - 230, cy - 150, c + 230, cy + 150))
    cel(img, bm, MARBLE, ink=3.0, k=40, light_k=24)
    wall = ellipse_mask(img, (c - 230, cy - 120, c + 230, cy + 180))
    front = ImageChops.subtract(wall, bm)
    img.paste(rgba(STONE[1]), (0, 0), front)
    img.paste(INK, (0, 0), ImageChops.subtract(dilate(wall, 3 * SS), dilate(bm, 3 * SS)).point(
        lambda v: 255 if v else 0))
    wm = ellipse_mask(img, (c - 196, cy - 124, c + 196, cy + 124))
    cel(img, wm, WATER, ink=0, k=-30, light_k=20)
    img.paste(rgba(WATER[2]), (0, 0), ImageChops.subtract(wm, shifted(wm, 0, 26)))
    for k in range(5):
        a = k * 2 * math.pi / 5 + 0.4
        x, y = c + math.cos(a) * 140, cy + math.sin(a) * 86
        d.arc((x - 26, y - 10, x + 26, y + 10), 200, 340, fill=rgba(WATER[0]), width=int(1.6 * SS))
    # centre pillar and bowl
    pm = poly_mask(img, [(c - 26, cy + 20), (c + 26, cy + 20), (c + 20, cy - 110), (c - 20, cy - 110)])
    cel(img, pm, MARBLE, ink=2.6, k=14)
    bowl = ellipse_mask(img, (c - 78, cy - 150, c + 78, cy - 100))
    cel(img, bowl, MARBLE, ink=2.6, k=16)
    # water jet and falling drops
    jm = ellipse_mask(img, (c - 22, cy - 230, c + 22, cy - 120))
    cel(img, jm, ((240, 252, 255), (170, 228, 255), (100, 190, 245)), ink=2.0, k=8)
    for side in (-1, 1):
        d.arc((c - 120 if side < 0 else c, cy - 220, c if side < 0 else c + 120, cy - 40),
              180 if side < 0 else 270, 270 if side < 0 else 360, fill=rgba((190, 236, 255)), width=int(2.2 * SS))
    for x, y in ((c - 70, cy - 70), (c + 80, cy - 60), (c - 20, cy - 250), (c + 30, cy - 240)):
        d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=WHITE)
    return img.resize((W * 4, W * 4), Image.LANCZOS)  # drawn on 2x2, shown on 4x4 cells of the 12x12 square


def furnace_prop():
    """1x2 cells: a squat brick furnace with a glowing mouth and a chimney (crafting node 406)."""
    img = new_cell(1, 2)
    d = ImageDraw.Draw(img)
    c = W / 2
    ch = poly_mask(img, [(c + 30, 150), (c + 74, 150), (c + 74, 40), (c + 30, 40)])
    cel(img, ch, BRICK, ink=2.4, k=12)
    bm = new_mask(img)
    bd = ImageDraw.Draw(bm)
    bd.rounded_rectangle((28, 130, W - 28, W + W * 0.86), radius=60, fill=255)
    cel(img, bm, ((236, 160, 120), (204, 118, 90), (150, 78, 70)), ink=2.8, k=34, light_k=20,
        highlight=(70, 170, 8))
    for y in range(190, int(W + W * 0.8), 34):
        d.line([(44, y), (W - 44, y)], fill=rgba((150, 78, 70)), width=int(1.2 * SS))
    mm = new_mask(img)
    md = ImageDraw.Draw(mm)
    md.pieslice((c - 52, W + 40, c + 52, W + 160), 180, 360, fill=255)
    md.rectangle((c - 52, W + 100, c + 52, W + 160), fill=255)
    img.paste(INK, (0, 0), dilate(mm, 2.4 * SS))
    img.paste((255, 170, 60, 255), (0, 0), mm)
    fm = ellipse_mask(img, (c - 34, W + 80, c + 34, W + 160))
    img.paste((255, 236, 140, 255), (0, 0), fm)
    for x, y, r in ((c + 52, 30, 14), (c + 62, 6, 10)):  # smoke puffs
        d.ellipse((x - r, y - r, x + r, y + r), fill=(240, 240, 250, 220))
    return img


def furnace_small_prop():
    """1x1 cell: the furnace squat enough for a one-tile EntitySprite (crafting node 406): a domed brick body with
    a glowing mouth and a short chimney with a smoke puff, inside the cell so nothing is cut off."""
    img = new_cell(1, 1)
    d = ImageDraw.Draw(img)
    c = W / 2
    ch = poly_mask(img, [(c + 40, 96), (c + 80, 96), (c + 80, 40), (c + 40, 40)])
    cel(img, ch, BRICK, ink=2.4, k=10)
    bm = new_mask(img)
    ImageDraw.Draw(bm).rounded_rectangle((24, 74, W - 24, W - 18), radius=56, fill=255)
    cel(img, bm, ((236, 160, 120), (204, 118, 90), (150, 78, 70)), ink=2.8, k=30, light_k=18,
        highlight=(70, 106, 8))
    for y in range(124, W - 30, 30):
        d.line([(40, y), (W - 40, y)], fill=rgba((150, 78, 70)), width=int(1.2 * SS))
    mm = new_mask(img)
    md = ImageDraw.Draw(mm)
    md.pieslice((c - 48, 128, c + 48, 220), 180, 360, fill=255)
    md.rectangle((c - 48, 174, c + 48, W - 30), fill=255)
    img.paste(INK, (0, 0), dilate(mm, 2.4 * SS))
    img.paste((255, 170, 60, 255), (0, 0), mm)
    img.paste((255, 236, 140, 255), (0, 0), ellipse_mask(img, (c - 30, 160, c + 30, W - 30)))
    for x, y, r in ((c + 62, 26, 14), (c + 40, 12, 9)):  # smoke puffs
        d.ellipse((x - r, y - r, x + r, y + r), fill=(240, 240, 250, 220))
    return img


def fire_prop():
    """1x1 cell: a cooking fire on crossed logs (crafting node 407)."""
    img = new_cell(1, 1)
    c = W / 2
    for a in (-0.35, 0.35):
        dx, dy = math.cos(a) * 86, math.sin(a) * 30
        lm = new_mask(img)
        ImageDraw.Draw(lm).line([(c - dx, 200 - dy), (c + dx, 200 + dy)], fill=255, width=30)
        cel(img, lm, BARK, ink=2.2, k=8)
    fm = poly_mask(img, [(c - 54, 196), (c - 40, 120), (c - 14, 150), (c, 40), (c + 22, 130), (c + 44, 100),
                         (c + 54, 196)])
    fm = fm.filter(ImageFilter.GaussianBlur(6)).point(lambda v: 255 if v > 128 else 0)
    cel(img, fm, ((255, 246, 160), (255, 176, 60), (236, 92, 50)), ink=2.4, k=-22, light_k=14)
    inner = poly_mask(img, [(c - 24, 190), (c - 10, 130), (c + 6, 150), (c + 22, 120), (c + 26, 190)])
    img.paste((255, 248, 200, 255), (0, 0), inner)
    return img


# ---------------------------------------------------------------- assembly

# variant weights per terrain column (rows 0..n-1); blob4 columns use all 16 rows as masks
VARIANT_WEIGHTS = {
    0: [30, 30, 24, 16, 8, 6, 4, 3],   # grass: plain-ish first, clover, flowers and a stone rarer
    2: [40, 10, 6, 6],                 # floor
    7: [5, 3, 3],                      # rock
    11: [16, 10, 4, 2],                # cave
}
BLOB47 = {1: [1, 8, 9, 10, 12], 4: [4], 6: [6]}   # terrain -> terrains it connects to
BLOB4 = {3: [3]}
# terrain -> (ids, row offset): when any of the 8 neighbours has one of the ids, the row moves down by the offset
NEAR = {5: ([2], 1),      # a tree next to town flagstones stands in a planter
        7: ([11], 3)}     # boulders next to the cave floor stand on cave ground


def build_terrain():
    atlas = Image.new("RGBA", (T * N_TERRAIN, T * ROWS), (0, 0, 0, 0))
    filled = {}
    rng = random.Random(1234)

    def put(col, row, cell):
        atlas.paste(down(cell), (col * T, row * T))
        filled.setdefault(col, []).append(row)

    for kind in range(len(VARIANT_WEIGHTS[0])):
        put(0, kind, grass_tile(random.Random(500 + kind), kind))
    for m in range(16):
        put(3, m, temple_tile(m))
    for r, m8 in enumerate(BLOB47_MASKS):
        put(1, r, road_tile(m8))
        put(4, r, water_tile(m8, random.Random(900 + r % 4)))
        put(6, r, wall_tile(m8))
    for kind in range(len(VARIANT_WEIGHTS[2])):
        put(2, kind, floor_tile(random.Random(700 + kind), kind))
    put(5, 0, tree_base())
    put(5, 1, tree_planter())
    for kind in range(len(VARIANT_WEIGHTS[7])):
        put(7, kind, rock_tile(random.Random(800 + kind), kind))
        put(7, kind + NEAR[7][1], rock_tile(random.Random(800 + kind), kind, "cave"))
    put(8, 0, warp_tile())
    put(9, 0, waybell_tile())
    put(10, 0, station_tile())
    for kind in range(len(VARIANT_WEIGHTS[11])):
        put(11, kind, cave_tile(random.Random(600 + kind), kind))
    put(12, 0, shop_tile())
    del rng
    return atlas, filled


# props atlas: name -> (col, row, w, h) on the 64 px grid
PROP_CELLS = {
    "tree0": (0, 0, 1, 2), "tree1": (1, 0, 1, 2), "tree2": (2, 0, 1, 2), "bell": (3, 0, 1, 2),
    "stall0": (4, 0, 1, 2), "stall1": (5, 0, 1, 2), "stall2": (6, 0, 1, 2), "stall3": (7, 0, 1, 2),
    "furnace": (8, 0, 1, 2), "fountain": (0, 2, 4, 4), "fire": (4, 2, 1, 1), "furnace_small": (5, 2, 1, 1),
}
PROP_COLS, PROP_ROWS = 9, 6
PROP_BASE_PX = {}  # px below the base cell's centre where a prop touches the ground (y-sort); default T // 2


def build_props():
    atlas = Image.new("RGBA", (T * PROP_COLS, T * PROP_ROWS), (0, 0, 0, 0))
    makers = {"tree0": lambda: tree_prop(0), "tree1": lambda: tree_prop(1), "tree2": lambda: tree_prop(2),
              "bell": bell_prop, "stall0": lambda: stall_prop(0), "stall1": lambda: stall_prop(1),
              "stall2": lambda: stall_prop(2), "stall3": lambda: stall_prop(3), "furnace": furnace_prop,
              "fountain": fountain_prop, "fire": fire_prop, "furnace_small": furnace_small_prop}
    for name, (c, r, _, _) in PROP_CELLS.items():
        atlas.paste(harden(down(makers[name]())), (c * T, r * T))
    return atlas


def index_json(filled):
    terrains = []
    for tid in range(N_TERRAIN):
        e = {"id": tid, "name": NAMES[tid], "source": "procedural (gen_tiles.py)"}
        if tid in BLOB47:
            e.update({"mode": "blob47", "connects": BLOB47[tid]})
        elif tid in BLOB4:
            e.update({"mode": "blob4", "connects": BLOB4[tid]})
        elif tid in VARIANT_WEIGHTS:
            e.update({"mode": "variants", "weights": VARIANT_WEIGHTS[tid]})
        else:
            e.update({"mode": "single"})
        if tid in NEAR:
            e["near"] = {"ids": NEAR[tid][0], "offset": NEAR[tid][1]}
        terrains.append(e)

    def cell(name, origin):
        c, r, w, h = PROP_CELLS[name]
        return {"atlas": [c, r], "size": [w, h], "origin": origin, "base_px": PROP_BASE_PX.get(name, T // 2)}

    tall = [0, T // 2]  # a 1x2 prop drawn with its bottom on the cell's bottom edge
    props = {
        "cells": {n: cell(n, tall if PROP_CELLS[n][3] == 2 and PROP_CELLS[n][2] == 1 else
                          ([-T // 2, -T // 2] if n == "fountain" else [0, 0])) for n in PROP_CELLS},
        "on_terrain": {"5": ["tree0", "tree1", "tree2"], "9": ["bell"],
                       "12": ["stall0", "stall1", "stall2", "stall3"]},
        "shop_pick": "x",           # stalls cycle by x so a row of shops alternates colours
        "block_center": {"3": "fountain"},
        "nodes": {"406": "furnace_small", "407": "fire"},  # 1x1 cells: an EntitySprite draws one tile
    }
    return {
        "version": 1, "cell_px": T, "columns": N_TERRAIN, "rows": ROWS,
        "terrain_atlas": "res://assets/tiles/terrain.png",
        "props_atlas": "res://assets/tiles/props.png", "props_columns": PROP_COLS, "props_rows": PROP_ROWS,
        "mask_bits": {"N": 1, "E": 2, "S": 4, "W": 8, "NE": 16, "SE": 32, "SW": 64, "NW": 128},
        "blob47_masks": BLOB47_MASKS,
        "terrains": terrains, "props": props,
        "_doc_base_px": "props.cells.<name>.base_px: px below the base cell's centre where the prop touches the "
                        "ground; MapRenderer uses it as the tile's y_sort_origin (minus the entity feet offset) so "
                        "characters sort behind/in front of props",
        "licence": "CC0 1.0, project-original, generated procedurally by client/tools/art/tiles/gen_tiles.py",
        "filled_rows": {str(k): sorted(v) for k, v in filled.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    terrain, filled = build_terrain()
    props = build_props()
    tp, pp = os.path.join(args.out, "terrain.png"), os.path.join(args.out, "props.png")
    terrain.save(tp, optimize=True)
    props.save(pp, optimize=True)
    with open(os.path.join(args.out, "tiles.json"), "w", encoding="utf-8") as f:
        json.dump(index_json(filled), f, indent=1)
    kb = (os.path.getsize(tp) + os.path.getsize(pp)) // 1024
    print("GEN_TILES done terrain=%dx%d props=%dx%d total_kb=%d" % (terrain.width, terrain.height, props.width,
                                                                    props.height, kb))


if __name__ == "__main__":
    main()
