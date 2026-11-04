"""Procedural chibi anime-style placeholder sheets for round 2 (docs/round2_contract.md 1.4).

Writes `client/assets/anim/<kind>/<action>.png` plus `client/assets/anim/<kind>/act.json` for the 7 jobs, the
20 monster kinds and the 2 NPC kinds, and `client/assets/anim/_review/contact_sheet.png` for review. Stdlib only (zlib PNG writer, no Pillow).

Sheet format: rows = directions 0..4 (S, SW, W, NW, N); rows 5..7 (NE, E, SE) are mirrored by the client from
rows 3..1 ("mirror": true). Columns = frames. Every frame is frame_w x frame_h with the feet at `anchor`.

Style: chibi proportions (big head, big eyes, small body), 1 px dark outline around every part (so parts get
inner lines too), two-tone cel shading lit from the upper left plus a highlight, saturated palettes.

Usage (from the repo root or client/): python client/tools/gen_placeholder_anim.py [--out DIR] [--only KIND]
"""
import argparse
import json
import math
import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "assets", "anim"))

N_ROWS = 5  # directions 0..4 drawn; 5..7 mirrored from 3..1
# screen facing vector per drawn direction: 0 S, 1 SW, 2 W, 3 NW, 4 N
FACE = {0: (0.0, 1.0), 1: (-0.71, 0.71), 2: (-1.0, 0.0), 3: (-0.71, -0.71), 4: (0.0, -1.0)}

PLAYER_ACTIONS = ["idle", "walk", "sit", "pickup", "standby", "attack", "hurt", "dead", "cast"]
MONSTER_ACTIONS = ["idle", "walk", "attack", "hurt", "dead"]
NPC_ACTIONS = ["idle", "talk"]
# action -> (frames, delay_ms, loop)
CLIP = {
    "idle": (4, 200, True), "walk": (8, 75, True), "sit": (4, 300, True), "pickup": (4, 100, False),
    "standby": (4, 150, True), "attack": (6, 100, False), "hurt": (4, 75, False), "dead": (5, 120, False),
    "cast": (6, 100, True), "talk": (4, 160, True),
}
ATK_FRAME = 3

NPCS = ["npc_shop", "npc_kafra"]
JOBS = ["novice", "swordsman", "mage", "archer", "acolyte", "thief", "merchant"]
# engine order (game/data/monsters.py)
MONSTERS = [
    "poring", "orc_warrior", "fabre", "lunatic", "drops", "poporing", "willow", "rocker", "spore", "thief_bug",
    "zombie", "wolf", "skeleton", "peco_peco", "hode", "soldier_skeleton", "bigfoot", "mummy",
    "golden_thief_bug", "eddga",
]

OUTLINE_DARK = 0.32
EYE_DARK = (40, 24, 48)
WHITE = (255, 255, 255)
HURT_FLASH = (255, 70, 50)  # saturated red-orange flinch flash (never white: that washes the figure out)
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ---------------------------------------------------------------- colour helpers
def hx(s):
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def clamp8(v):
    return max(0, min(255, int(round(v))))


def shade_of(c):
    # cel shadow: darker, nudged toward violet like anime shading
    return (clamp8(c[0] * 0.68 + 10), clamp8(c[1] * 0.60), clamp8(c[2] * 0.78 + 22))


def hi_of(c):
    return tuple(clamp8(v + (255 - v) * 0.55) for v in c)


def line_of(c):
    return (clamp8(c[0] * OUTLINE_DARK), clamp8(c[1] * OUTLINE_DARK * 0.85), clamp8(c[2] * OUTLINE_DARK + 12))


def mix(a, b, t):
    return tuple(clamp8(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ---------------------------------------------------------------- drawing
class Part:
    """A set of pixels stamped together: one outline around the union, cel shading per shape."""

    def __init__(self, base, line=None, shiny=False, soft=False):
        self.base = base
        self.shadow = mix(base, (205, 110, 120), 0.28) if soft else shade_of(base)
        self.shade_t = 0.5 if soft else 0.32
        self.hi = hi_of(base)
        self.line = line or line_of(base)
        self.shiny = shiny
        self.px = {}

    def _tone(self, nx, ny, shading, shiny):
        if not shading:
            return self.base
        if shiny and (nx + 0.38) ** 2 + (ny + 0.42) ** 2 < 0.05:
            return WHITE
        if (nx + 0.3) ** 2 + (ny + 0.45) ** 2 < 0.08 and shiny is not None:
            return self.hi if self.shiny or shiny else self.base
        if nx * 0.5 + ny * 0.8 > self.shade_t:
            return self.shadow
        return self.base

    def ellipse(self, cx, cy, rx, ry, shading=True, shiny=None, color=None):
        rx = max(rx, 0.6)
        ry = max(ry, 0.6)
        for y in range(int(math.floor(cy - ry)), int(math.ceil(cy + ry)) + 1):
            for x in range(int(math.floor(cx - rx)), int(math.ceil(cx + rx)) + 1):
                nx = (x + 0.5 - cx) / rx
                ny = (y + 0.5 - cy) / ry
                if nx * nx + ny * ny <= 1.0:
                    self.px[(x, y)] = color or self._tone(nx, ny, shading, self.shiny if shiny is None else shiny)

    def poly(self, pts, shading=True, color=None):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = max((x1 - x0) / 2, 0.5), max((y1 - y0) / 2, 0.5)
        n = len(pts)
        for y in range(int(math.floor(y0)), int(math.ceil(y1)) + 1):
            yc = y + 0.5
            hits = []
            for i in range(n):
                ax, ay = pts[i]
                bx, by = pts[(i + 1) % n]
                if (ay <= yc < by) or (by <= yc < ay):
                    hits.append(ax + (yc - ay) * (bx - ax) / (by - ay))
            hits.sort()
            for k in range(0, len(hits) - 1, 2):
                for x in range(int(math.ceil(hits[k] - 0.5)), int(math.floor(hits[k + 1] - 0.5)) + 1):
                    nx = (x + 0.5 - cx) / rx
                    ny = (yc - cy) / ry
                    self.px[(x, y)] = color or self._tone(nx, ny, shading, False)

    def thick_line(self, x0, y0, x1, y1, w, shading=True, color=None):
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy) or 1.0
        ox, oy = -dy / ln * w / 2, dx / ln * w / 2
        self.poly([(x0 + ox, y0 + oy), (x1 + ox, y1 + oy), (x1 - ox, y1 - oy), (x0 - ox, y0 - oy)], shading, color)


class Canvas:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.p = [None] * (w * h)

    def put(self, x, y, c):
        x = int(x)
        y = int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.p[y * self.w + x] = c

    def get(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            return self.p[y * self.w + x]
        return None

    def stamp(self, part):
        px = part.px
        for (x, y) in px:
            for dx, dy in N4:
                if (x + dx, y + dy) not in px:
                    self.put(x + dx, y + dy, part.line)
        for (x, y), c in px.items():
            self.put(x, y, c)

    def dot(self, x, y, c, w=1, h=1):
        for yy in range(int(round(y)), int(round(y)) + h):
            for xx in range(int(round(x)), int(round(x)) + w):
                self.put(xx, yy, c)

    def tint(self, c, t):
        """Flash colour; outline-dark pixels (luminance < 70) are left alone so edges stay crisp."""
        def lum(v):
            return 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2]
        self.p = [None if v is None else (v if lum(v) < 70 else mix(v, c, t)) for v in self.p]

    def bbox(self):
        xs, ys = [], []
        for i, v in enumerate(self.p):
            if v is not None:
                xs.append(i % self.w)
                ys.append(i // self.w)
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def rotated(self, deg, px, py):
        out = Canvas(self.w, self.h)
        a = math.radians(-deg)
        ca, sa = math.cos(a), math.sin(a)
        for y in range(self.h):
            for x in range(self.w):
                dx, dy = x + 0.5 - px, y + 0.5 - py
                sx = px + dx * ca - dy * sa
                sy = py + dx * sa + dy * ca
                out.p[y * self.w + x] = self.get(int(math.floor(sx)), int(math.floor(sy)))
        return out

    def shifted(self, ox, oy):
        out = Canvas(self.w, self.h)
        for y in range(self.h):
            for x in range(self.w):
                out.p[y * self.w + x] = self.get(x - ox, y - oy)
        return out

    def flipped(self):
        out = Canvas(self.w, self.h)
        for y in range(self.h):
            row = self.p[y * self.w:(y + 1) * self.w]
            out.p[y * self.w:(y + 1) * self.w] = row[::-1]
        return out


def write_png(path, w, h, pixels):
    """pixels: list of (r,g,b) or None (transparent), row-major."""
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            c = pixels[y * w + x]
            if c is None:
                raw += b"\x00\x00\x00\x00"
            else:
                raw += bytes((c[0], c[1], c[2], c[3] if len(c) > 3 else 255))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


# ---------------------------------------------------------------- poses
def pose_for(action, i, n):
    """Generic per-frame motion shared by all kinds. Distances in px, angles in degrees."""
    t = i / n
    p = {"bob": 0.0, "lunge": 0.0, "leg": 0.0, "arm": 0.0, "eyes": "open", "crouch": 0.0, "hop": 0.0,
         "sx": 1.0, "sy": 1.0, "fall": 0.0, "weapon": 20.0, "arms_up": 0.0, "mouth": 0.0, "flash": None,
         "sit": False, "stance": 0.0, "cast": -1.0, "reach": 0.0, "wave": None}
    if action == "idle":
        p["bob"] = [0, 1, 1, 0][i % 4]
        p["sy"] = [1.0, 0.97, 0.97, 1.0][i % 4]
        p["sx"] = [1.0, 1.03, 1.03, 1.0][i % 4]
    elif action == "walk":
        s = math.sin(2 * math.pi * t)
        p["leg"] = s
        p["arm"] = -s
        p["bob"] = abs(math.sin(2 * math.pi * t)) * 2.2
        p["hop"] = abs(math.sin(2 * math.pi * t)) * 7
        contact = 1 - abs(math.sin(2 * math.pi * t))
        p["sx"] = 1 + 0.22 * contact ** 3 - 0.06
        p["sy"] = 1 - 0.2 * contact ** 3 + 0.06
    elif action == "sit":
        p["sit"] = True
        p["crouch"] = 1.0
        p["bob"] = [0, 0.6, 0.6, 0][i % 4]
    elif action == "pickup":
        c = [0.4, 1.0, 1.0, 0.4][i % 4]
        p["crouch"] = c
        p["reach"] = c
        p["arm"] = 1.0 * c
    elif action == "standby":
        p["stance"] = 1.0
        p["bob"] = [0, 1, 1, 0][i % 4]
        p["weapon"] = -35
        p["arm"] = 0.5
    elif action == "attack":
        p["lunge"] = [-2, -3, 2, 5, 3, 0][i]
        p["weapon"] = [-70, -100, -10, 70, 95, 30][i]
        p["arm"] = [-0.8, -1.0, 0.6, 1.0, 1.0, 0.3][i]
        p["mouth"] = [0, 0, 1, 1, 1, 0][i]
        p["sx"] = [0.95, 0.92, 1.05, 1.12, 1.06, 1.0][i]
        p["sy"] = [1.04, 1.07, 0.97, 0.92, 0.96, 1.0][i]
        p["bob"] = [0, 1, 1, 0, 0, 0][i]
    elif action == "hurt":
        p["lunge"] = [-3, -4, -2, -1][i]
        p["eyes"] = "hurt" if i < 3 else "open"
        p["sx"] = [1.15, 1.08, 1.03, 1.0][i]
        p["sy"] = [0.85, 0.92, 0.97, 1.0][i]
        p["flash"] = [(HURT_FLASH, 0.45), (HURT_FLASH, 0.2), None, None][i]
        p["arm"] = [-1, -1, -0.5, 0][i]
        p["weapon"] = -40
    elif action == "dead":
        p["fall"] = [0.0, 0.25, 0.6, 0.92, 1.0][i]
        p["eyes"] = "hurt" if i == 0 else "dead"
        p["flash"] = [(HURT_FLASH, 0.45), None, None, None, None][i]
        p["lunge"] = [-2, -3, -3, -3, -3][i]
        p["sx"] = [1.0, 1.05, 1.12, 1.25, 1.35][i]
        p["sy"] = [1.0, 0.9, 0.7, 0.5, 0.42][i]
    elif action == "talk":  # npc: nod and wave the near hand, mouth chatters
        p["bob"] = [0, 1.5, 0, 1.5][i % 4]
        p["wave"] = [-1.0, 0.0, 1.0, 0.0][i % 4]
        p["mouth"] = [1, 0, 1, 0][i % 4]
        p["sy"] = [1.0, 0.97, 1.0, 0.97][i % 4]
    elif action == "cast":
        p["arms_up"] = 1.0
        p["cast"] = t
        p["bob"] = [0, 1, 1, 0, 0, 1][i % 6]
        p["weapon"] = -150
        p["mouth"] = 1
    return p


def lunge_xy(p, fx, fy):
    return fx * p["lunge"], fy * p["lunge"] * 0.5


# ---------------------------------------------------------------- shared face bits
def draw_eyes(cv, cx, cy, fx, back, style, iris, sep=4.0, big=True):
    """Anime eyes: tall dark ovals, coloured lower half, white glint top-left."""
    if back:
        return
    side = abs(fx) > 0.9
    centers = [cx + fx * 5] if side else [cx - sep + fx * 3, cx + sep + fx * 3]
    for ex in centers:
        ex = round(ex)
        ey = round(cy)
        if style == "hurt":  # >< squeezed eyes
            cv.put(ex - 1, ey - 1, EYE_DARK); cv.put(ex, ey, EYE_DARK); cv.put(ex - 1, ey + 1, EYE_DARK)
            cv.put(ex + 1, ey, EYE_DARK)
        elif style == "dead":  # x eyes
            for k in (-1, 0, 1):
                cv.put(ex + k, ey + k, EYE_DARK); cv.put(ex + k, ey - k, EYE_DARK)
        elif style == "closed":
            cv.dot(ex - 1, ey, EYE_DARK, 3, 1)
        elif style == "hollow":
            cv.dot(ex - 1, ey - 1, (20, 10, 20), 3, 3)
            cv.put(ex, ey, (255, 80, 60))
        elif style == "glow":
            cv.dot(ex - 1, ey, (255, 60, 40), 2, 1)
        elif style == "dots":
            cv.dot(ex, ey, EYE_DARK, 1, 2)
        else:
            h = 5 if big else 3
            cv.dot(ex - 1, ey - 2, EYE_DARK, 3, h)
            cv.dot(ex - 1, ey - 2 + h - 2, iris, 2 if side else 3, 1)
            cv.dot(ex, ey - 2 + h - 2, mix(iris, WHITE, 0.4), 1, 1)
            cv.put(ex - 1, ey - 2, WHITE)


def draw_mouth(cv, x, y, open_):
    if open_:
        cv.dot(x, y, (150, 30, 50), 2, 2)
    else:
        cv.dot(x, y, (120, 40, 50), 2, 1)


# ---------------------------------------------------------------- humanoids (jobs, orc, undead)
HUMANOID = {
    "novice": dict(skin="#ffd9b8", hair="#8a5a2b", hair_style="short", outfit="#d8b07a", outfit2="#5f9a3a",
                   shoes="#7a4a28", eye="#3a8adf", gear="ribbon", gear_c="#e8384f", weapon="knife", robe=False),
    "swordsman": dict(skin="#ffd9b8", hair="#e8551f", hair_style="spiky", outfit="#2f6fe0", outfit2="#e8c13a",
                      shoes="#5a3a2a", eye="#2a6ad0", gear="helm", gear_c="#9aaccc", weapon="sword", robe=False,
                      plate="#c8d4e8"),
    "mage": dict(skin="#ffe0c8", hair="#9a4ae0", hair_style="long", outfit="#8a2fd0", outfit2="#f0c040",
                 shoes="#4a2a6a", eye="#d03aa0", gear="witch", gear_c="#6a24b0", weapon="rod", robe=True),
    "archer": dict(skin="#ffd9b8", hair="#f6d04a", hair_style="ponytail", outfit="#2fae4a", outfit2="#a8683a",
                   shoes="#6a4020", eye="#2aa060", gear="feather", gear_c="#2a8a3a", weapon="bow", robe=False),
    "acolyte": dict(skin="#ffe0c8", hair="#5ab0f0", hair_style="short", outfit="#2a3a9a", outfit2="#ffffff",
                    shoes="#2a2a4a", eye="#3a5ad0", gear="biretta", gear_c="#202040", weapon="mace", robe=True),
    "thief": dict(skin="#f8d0a8", hair="#3a3a4a", hair_style="spiky", outfit="#5a2a7a", outfit2="#e0303a",
                  shoes="#2a1a2a", eye="#c03a3a", gear="bandana", gear_c="#e0303a", weapon="dagger", robe=False,
                  scarf=True),
    "merchant": dict(skin="#ffd9b8", hair="#a0602a", hair_style="short", outfit="#e8902a", outfit2="#7a4a20",
                     shoes="#5a3018", eye="#a06a2a", gear="cap", gear_c="#c0402a", weapon="axe", robe=False,
                     pack=True),
    "npc_shop": dict(skin="#ffd9b8", hair="#6a3a1a", hair_style="short", outfit="#e8d8b0", outfit2="#3aa0d8",
                     shoes="#5a3018", eye="#3a8a4a", gear="cap", gear_c="#e8902a", weapon="sign", robe=False,
                     apron=True, pouch=True),
    "npc_kafra": dict(skin="#ffe0c8", hair="#8a5a3a", hair_style="short", outfit="#1f3fa0", outfit2="#ffffff",
                      shoes="#f4f4fa", eye="#2a6ad0", gear="kafra", gear_c="#ffffff", weapon="clipboard",
                      robe=True, kafra=True),
    "orc_warrior": dict(skin="#5ab04a", hair="#3a2a1a", hair_style="none", outfit="#8a5a30", outfit2="#c0c0c8",
                        shoes="#3a2a1a", eye="#ffd020", gear="horns", gear_c="#9aa0b0", weapon="axe", robe=False,
                        tusks=True, scale=1.15, bulky=True),
    "zombie": dict(skin="#9ac07a", hair="#3a4a2a", hair_style="messy", outfit="#6a5a9a", outfit2="#4a3a5a",
                   shoes="#3a3030", eye="#ffffff", gear="none", weapon="none", robe=False, eye_style="dots",
                   arms_fwd=True, torn=True),
    "skeleton": dict(skin="#f2ecd8", hair=None, hair_style="none", outfit="#f2ecd8", outfit2="#a89a80",
                     shoes="#d8d0b8", eye="#000000", gear="none", weapon="none", robe=False, eye_style="hollow",
                     ribs=True, thin=True),
    "soldier_skeleton": dict(skin="#f2ecd8", hair=None, hair_style="none", outfit="#f2ecd8", outfit2="#a89a80",
                             shoes="#d8d0b8", eye="#000000", gear="helm", gear_c="#8a8f9a", weapon="sword",
                             robe=False, eye_style="hollow", ribs=True, thin=True, shield=True),
    "mummy": dict(skin="#ead9a8", hair=None, hair_style="none", outfit="#ead9a8", outfit2="#b8a070",
                  shoes="#ead9a8", eye="#000000", gear="none", weapon="none", robe=False, eye_style="glow",
                  arms_fwd=True, wraps=True),
}


def draw_weapon(cv, kind, hx_, hy, ang, s, spec):
    if kind == "none":
        return
    a = math.radians(ang - 90)  # 0 deg = pointing up
    def tip(L):
        return hx_ + math.cos(a) * L * s, hy + math.sin(a) * L * s
    metal = hx("#dfe6f2")
    if kind in ("sword", "knife", "dagger"):
        L = {"sword": 15, "knife": 8, "dagger": 9}[kind]
        blade = Part(metal, shiny=True)
        tx, ty = tip(L)
        blade.thick_line(hx_, hy, tx, ty, 2.6 * s)
        cv.stamp(blade)
        guard = Part(hx("#e8b830"))
        gx, gy = tip(1.5)
        perp = a + math.pi / 2
        guard.thick_line(gx - math.cos(perp) * 3 * s, gy - math.sin(perp) * 3 * s,
                         gx + math.cos(perp) * 3 * s, gy + math.sin(perp) * 3 * s, 1.6 * s)
        cv.stamp(guard)
    elif kind == "rod":
        stick = Part(hx("#9a6030"))
        tx, ty = tip(16)
        bx, by = tip(-4)
        stick.thick_line(bx, by, tx, ty, 2.0 * s)
        cv.stamp(stick)
        gem = Part(hx("#ff3a5a"), shiny=True)
        gem.ellipse(tx, ty, 2.8 * s, 2.8 * s)
        cv.stamp(gem)
    elif kind == "mace":
        stick = Part(hx("#8a5a30"))
        tx, ty = tip(11)
        stick.thick_line(hx_, hy, tx, ty, 2.0 * s)
        cv.stamp(stick)
        head = Part(hx("#e8c040"), shiny=True)
        head.ellipse(tx, ty, 3.2 * s, 3.2 * s)
        cv.stamp(head)
    elif kind == "axe":
        stick = Part(hx("#8a5a30"))
        tx, ty = tip(13)
        stick.thick_line(hx_, hy, tx, ty, 2.0 * s)
        cv.stamp(stick)
        blade = Part(metal, shiny=True)
        perp = a + math.pi / 2
        px_, py_ = math.cos(perp) * s, math.sin(perp) * s
        mx, my = tip(10)
        blade.poly([(mx, my), (tx, ty), (tx + px_ * 6, ty + py_ * 6 + 1), (mx + px_ * 6, my + py_ * 6 - 1)])
        cv.stamp(blade)
    elif kind == "clipboard":
        tx, ty = tip(9)
        board = Part(hx("#8a5a30"))
        board.poly([(tx - 5 * s, ty - 7 * s), (tx + 5 * s, ty - 7 * s), (tx + 5 * s, ty + 6 * s), (tx - 5 * s, ty + 6 * s)])
        cv.stamp(board)
        paper = Part(hx("#fffaf0"))
        paper.poly([(tx - 3.5 * s, ty - 5 * s), (tx + 3.5 * s, ty - 5 * s), (tx + 3.5 * s, ty + 4.5 * s), (tx - 3.5 * s, ty + 4.5 * s)], shading=False)
        cv.stamp(paper)
        for k in range(3):
            cv.dot(tx - 2 * s, ty - 3 * s + k * 3 * s, hx("#6a7a9a"), int(4 * s), 1)
        cv.dot(tx - 1, ty - 8 * s, hx("#d8d8e0"), 3, 2)
    elif kind == "sign":
        tx, ty = tip(11)
        post = Part(hx("#8a5a30"))
        post.thick_line(hx_, hy, tx, ty, 1.8 * s)
        cv.stamp(post)
        board = Part(hx("#f0c040"), shiny=True)
        board.poly([(tx - 6 * s, ty - 4 * s), (tx + 6 * s, ty - 4 * s), (tx + 6 * s, ty + 4 * s), (tx - 6 * s, ty + 4 * s)])
        cv.stamp(board)
        cv.dot(tx - 1, ty - 2 * s, hx("#8a4a10"), 3, int(3 * s))  # a coin mark
    elif kind == "bow":
        bow = Part(hx("#a0602a"))
        perp = a + math.pi / 2
        pts = []
        for k in range(-5, 6):
            u = k / 5
            off = (1 - u * u) * 4 * s
            pts.append((hx_ + math.cos(perp) * u * 9 * s + math.cos(a) * off,
                        hy + math.sin(perp) * u * 9 * s + math.sin(a) * off))
        for k in range(len(pts) - 1):
            bow.thick_line(pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1], 1.8 * s, shading=False)
        cv.stamp(bow)
        cv.dot(pts[0][0], pts[0][1], (250, 250, 240))
        for k in range(1, 10):
            u = k / 10
            cv.put(pts[0][0] + (pts[-1][0] - pts[0][0]) * u, pts[0][1] + (pts[-1][1] - pts[0][1]) * u,
                   (250, 245, 230))


def draw_humanoid(cv, name, d, p, ax, ay):
    sp = HUMANOID[name]
    s = sp.get("scale", 1.0)
    fx, fy = FACE[d]
    back = d >= 3
    side = d == 2
    lx, ly = lunge_xy(p, fx, fy)
    crouch = p["crouch"]
    stance = p["stance"]
    bulky = 1.25 if sp.get("bulky") else (0.82 if sp.get("thin") else 1.0)
    bx = ax + lx
    by = ay + ly
    body_drop = crouch * 6 * s
    lift = p["bob"] * s
    sx, sy = 1 + (p["sx"] - 1) * 0.45, 1 + (p["sy"] - 1) * 0.45  # humanoids squash less than blobs

    def X(v):  # horizontal offset scaled
        return bx + v * s * sx

    def Y(v):  # vertical offset above feet (positive v = up)
        return by - (v * s * sy) + body_drop - lift

    skin, outfit, outfit2 = hx(sp["skin"]), hx(sp["outfit"]), hx(sp["outfit2"])
    shoes = hx(sp["shoes"])
    hair = hx(sp["hair"]) if sp["hair"] else None

    head_y = 31
    hr = 10.5
    # weapon hand: character's right hand = screen left when facing south
    near = -1 if not back else 1
    if side:
        near = 1 if fx > 0 else -1

    def arm_pos(sign, swing):
        ox = sign * (7.5 * bulky)
        oy = 14 - swing * 0.5
        if side:
            ox = sign * 1.5 + fx * swing * 3.5
        if p["arms_up"] > 0:
            oy = 21
            ox = sign * 10.5 * bulky
        if p["reach"] > 0:
            oy = 8 - p["reach"] * 3
            ox += fx * 4
        if sp.get("arms_fwd") and p["arms_up"] == 0:
            oy = 18
            ox = sign * 5 + fx * 7
        if p["wave"] is not None and sign == near and not sp.get("prop_hand"):
            oy = 20
            ox = sign * 9.5 * bulky + p["wave"] * 2.5
        return ox, oy

    def draw_arm(sign, swing):
        ox, oy = arm_pos(sign, swing)
        a = Part(outfit if not sp.get("ribs") else skin)
        a.ellipse(X(ox), Y(oy), 2.4 * s * bulky, 4.2 * s)
        cv.stamp(a)
        hand = Part(skin, soft=True)
        hand.ellipse(X(ox), Y(oy - 4), 2.0 * s, 2.0 * s)
        cv.stamp(hand)
        return X(ox), Y(oy - 4)

    def draw_weapon_hand():
        swing = p["arm"]
        hxp, hyp = draw_arm(near, swing * (1 if not back else -1))
        ang = p["weapon"]
        if side and fx < 0:
            ang = -ang
        elif not side:
            ang = ang * (-1 if near < 0 else 1) * 0.7
        draw_weapon(cv, sp["weapon"], hxp, hyp, ang, s, sp)

    def draw_shield():
        if not sp.get("shield"):
            return
        ox, oy = arm_pos(-near, -p["arm"])
        sh = Part(hx("#7a5a3a"))
        sh.ellipse(X(ox) + fx * 2, Y(oy - 1), 4.5 * s, 5.5 * s)
        cv.stamp(sh)
        cv.dot(X(ox) + fx * 2 - 1, Y(oy - 1) - 1, hx("#e8c040"), 2, 2)

    def draw_long_hair():
        if sp["hair_style"] in ("long", "ponytail") and hair:
            h = Part(hair)
            if sp["hair_style"] == "long":
                h.poly([(X(-10.5), Y(head_y)), (X(10.5), Y(head_y)), (X(9), Y(19)), (X(-9), Y(19))])
            else:
                tx = -fx * 8 if side else 0
                h.ellipse(X(tx), Y(20 if back else 22), 3.5 * s, 7 * s)
            cv.stamp(h)

    def draw_legs():
        leg = p["leg"]
        legs = Part(shoes)
        spread = 3.2 * bulky + stance * 1.5
        if p["sit"]:
            for sgn in (-1, 1):
                legs.ellipse(X(sgn * spread + fx * 4), Y(1.5), 3.0 * s, 2.2 * s)
        elif side:
            for sgn in (-1, 1):
                legs.ellipse(X(fx * sgn * leg * 3.5), Y(3.5 + max(0, sgn * leg) * 1.5), 2.6 * s, 4.0 * s)
        else:
            for sgn in (-1, 1):
                up = max(0, sgn * leg) * 2.2
                legs.ellipse(X(sgn * spread), Y(3.5 + up), 2.6 * s, 4.0 * s)
        cv.stamp(legs)

    def draw_body():
        body = Part(outfit, shiny=bool(sp.get("plate")))
        body.ellipse(X(0), Y(13), (5.5 if side else 6.8) * s * bulky, 7.2 * s)
        if sp["robe"]:
            body.poly([(X(-6 * bulky), Y(15)), (X(6 * bulky), Y(15)), (X(8.5 * bulky), Y(2)),
                       (X(-8.5 * bulky), Y(2))])
        elif not sp.get("ribs"):
            body.poly([(X(-6 * bulky), Y(13)), (X(6 * bulky), Y(13)), (X(7.5 * bulky), Y(6)),
                       (X(-7.5 * bulky), Y(6))])
        cv.stamp(body)
        if sp.get("plate") and not back:
            pl = Part(hx(sp["plate"]), shiny=True)
            pl.ellipse(X(fx * 1.5), Y(15), 4.5 * s, 3.8 * s)
            cv.stamp(pl)
        if sp["robe"] and not back:  # collar / trim
            cv.dot(X(-2) + fx * 2, Y(19), outfit2, 5, 2)
            for yy in range(int(Y(15)), int(Y(3))):
                cv.put(X(0) + fx * 2, yy, outfit2)
        elif not sp.get("ribs") and not sp.get("wraps"):
            cv.dot(X(-6 * bulky), Y(10), outfit2, int(12 * bulky * s) + 1, 2)  # belt
        if sp.get("ribs"):
            for k in range(3):
                cv.dot(X(-3), Y(17 - k * 3), hx("#6a5a48"), 7, 1)
        if sp.get("wraps"):
            for k in range(5):
                yy = Y(19 - k * 3)
                for xx in range(int(X(-6)), int(X(6))):
                    if (xx + k) % 3 != 0 and cv.get(xx, int(yy)) is not None:
                        cv.put(xx, yy, hx(sp["outfit2"]))
        if sp.get("torn"):
            cv.dot(X(-4), Y(8), shade_of(outfit), 2, 2)
            cv.dot(X(3), Y(11), skin, 2, 2)
        if sp.get("apron") and not back:
            ap = Part(hx("#fffaf0"))
            ap.poly([(X(-4.5 + fx * 1.5), Y(13)), (X(4.5 + fx * 1.5), Y(13)), (X(5.5 + fx * 1.5), Y(3)),
                     (X(-5.5 + fx * 1.5), Y(3))])
            cv.stamp(ap)
            cv.dot(X(-1 + fx * 1.5), Y(8), outfit2, 2, 2)
        if sp.get("pouch"):
            pc = Part(hx("#b8802a"))
            pc.ellipse(X(6.5 * (1 if not side else 0) - fx * 4), Y(6), 3.0 * s, 3.2 * s)
            cv.stamp(pc)
            cv.dot(X(6.5 * (1 if not side else 0) - fx * 4) - 1, Y(6) - 1, hx("#f0d040"), 2, 2)
        if sp.get("kafra") and not back:
            tr = Part(WHITE)
            tr.poly([(X(-3), Y(18)), (X(3), Y(18)), (X(1.5), Y(8)), (X(-1.5), Y(8))], shading=False)
            cv.stamp(tr)
            cv.dot(X(-1 + fx * 1), Y(18), hx("#e0303a"), 3, 2)
            cv.dot(X(-8 * bulky), Y(3), outfit2, int(16 * bulky) + 1, 1)  # skirt hem
        if sp.get("scarf"):
            sc = Part(outfit2)
            sc.ellipse(X(0), Y(20), 6.5 * s, 2.6 * s)
            if not back:
                sc.poly([(X(-fx * 3 - 4), Y(20)), (X(-fx * 3 - 1), Y(20)), (X(-fx * 6 - 2), Y(11)),
                         (X(-fx * 6 - 5), Y(12))])
            cv.stamp(sc)

    def draw_pack():
        if sp.get("pack"):
            pk = Part(hx("#9a6a3a"))
            pk.ellipse(X(-fx * 6), Y(16), 5.5 * s, 6.5 * s)
            cv.stamp(pk)

    def draw_head():
        h = Part(skin, soft=True)
        h.ellipse(X(0), Y(head_y), hr * s, (hr - 0.6) * s)
        cv.stamp(h)
        if sp.get("tusks") and not back:
            for sgn in ((-1, 1) if not side else (1,)):
                tx_ = X(sgn * 3 + fx * 5) if not side else X(fx * 7)
                cv.dot(tx_, Y(head_y - 6), WHITE, 1, 3)
        if sp.get("wraps"):
            for k in range(4):
                yy = Y(head_y + 6 - k * 4)
                for xx in range(int(X(-10)), int(X(10))):
                    if (xx + k * 2) % 4 != 0 and cv.get(xx, int(yy)) is not None:
                        cv.put(xx, yy, hx(sp["outfit2"]))

    def draw_face():
        if not back:
            style = p["eyes"] if p["eyes"] != "open" else sp.get("eye_style", "open")
            draw_eyes(cv, X(0), Y(head_y - 1), fx, back, style, hx(sp["eye"]), sep=4.2 * s)
            if not sp.get("ribs") and not sp.get("wraps"):
                draw_mouth(cv, X(fx * 4) - (0 if side else 0.5), Y(head_y - 5.5), p["mouth"] > 0)
                if not side:  # blush
                    cv.dot(X(-7) + fx * 2, Y(head_y - 3.5), mix(skin, (255, 90, 120), 0.45), 2, 1)
                    cv.dot(X(6) + fx * 2, Y(head_y - 3.5), mix(skin, (255, 90, 120), 0.45), 2, 1)
            elif sp.get("ribs"):
                cv.dot(X(-2 + fx * 3), Y(head_y - 6), (60, 40, 40), 5, 1)

    def draw_hair():
        if not hair:
            return
        st = sp["hair_style"]
        if st == "none":
            return
        h = Part(hair)
        if back:
            h.ellipse(X(0), Y(head_y + 0.5), (hr + 0.6) * s, (hr - 0.2) * s)
        else:
            h.ellipse(X(-fx * 1.5), Y(head_y + 5.5), (hr + 0.4) * s, 5.5 * s)
            # bangs: three points over the forehead, shifted to the facing side
            for k in (-6, -1.5, 3):
                bxp = k + fx * 2.5
                h.poly([(X(bxp - 2.5), Y(head_y + 4)), (X(bxp + 3), Y(head_y + 4)), (X(bxp + 1), Y(head_y - 1))])
            if side:  # back of head hair
                h.ellipse(X(-fx * 4.5), Y(head_y + 1), 7 * s, 8 * s)
        if st == "spiky":
            for k in (-7, -2, 3, 8):
                h.poly([(X(k - 3), Y(head_y + 7)), (X(k + 3), Y(head_y + 7)), (X(k + 1 - fx * 2), Y(head_y + 13))])
        if st == "messy":
            for k in (-8, 0, 7):
                h.poly([(X(k - 3), Y(head_y + 6)), (X(k + 3), Y(head_y + 6)), (X(k - 2), Y(head_y + 11))])
        cv.stamp(h)

    def draw_gear():
        g = sp.get("gear", "none")
        if g == "none":
            return
        gc = hx(sp["gear_c"]) if sp.get("gear_c") else (200, 200, 200)
        if g == "helm":
            pt = Part(gc, shiny=True)
            pt.ellipse(X(0), Y(head_y + 4.5), (hr + 0.8) * s, 6.8 * s)
            pt.poly([(X(-hr - 0.8), Y(head_y + 4)), (X(hr + 0.8), Y(head_y + 4)), (X(hr + 0.8), Y(head_y + 1)),
                     (X(-hr - 0.8), Y(head_y + 1))])
            cv.stamp(pt)
            plume = Part(hx("#e02a3a"))
            plume.ellipse(X(-fx * 2), Y(head_y + 12), 2.4 * s, 4.0 * s)
            cv.stamp(plume)
        elif g == "witch":
            brim = Part(gc)
            brim.ellipse(X(0), Y(head_y + 6), (hr + 5) * s, 2.8 * s)
            cv.stamp(brim)
            cone = Part(gc)
            cone.poly([(X(-7.5), Y(head_y + 6.5)), (X(7.5), Y(head_y + 6.5)), (X(-fx * 4 + 4), Y(head_y + 20)),
                       (X(-fx * 4 + 2), Y(head_y + 21))])
            cv.stamp(cone)
            cv.dot(X(-7), Y(head_y + 8), hx("#f0c040"), int(14 * s) + 1, 2)
        elif g == "ribbon":
            r = Part(gc)
            r.ellipse(X(5 - fx * 2), Y(head_y + 8), 3.2 * s, 2.3 * s)
            r.ellipse(X(9 - fx * 2), Y(head_y + 8), 3.2 * s, 2.3 * s)
            cv.stamp(r)
        elif g == "feather":
            cap = Part(gc)
            cap.ellipse(X(0), Y(head_y + 6.5), (hr - 0.5) * s, 4.5 * s)
            cv.stamp(cap)
            f = Part(hx("#f04a3a"))
            f.poly([(X(-fx * 3 + 4), Y(head_y + 8)), (X(-fx * 3 + 7), Y(head_y + 8)), (X(-fx * 6 + 12), Y(head_y + 17))])
            cv.stamp(f)
        elif g == "biretta":
            cap = Part(gc)
            cap.poly([(X(-8), Y(head_y + 6)), (X(8), Y(head_y + 6)), (X(6), Y(head_y + 12)), (X(-6), Y(head_y + 12))])
            cv.stamp(cap)
            cv.dot(X(-1), Y(head_y + 14), hx("#e02a3a"), 3, 2)
        elif g == "bandana":
            b = Part(gc)
            b.ellipse(X(0), Y(head_y + 6), (hr + 0.6) * s, 4.0 * s)
            b.poly([(X(-fx * 9), Y(head_y + 5)), (X(-fx * 9 - 2), Y(head_y + 4)), (X(-fx * 13 - 1), Y(head_y - 2)),
                    (X(-fx * 11 + 2), Y(head_y - 1))])
            cv.stamp(b)
            if not back:
                cv.dot(X(-3 + fx * 3), Y(head_y + 7), WHITE, 1, 1)
                cv.dot(X(2 + fx * 3), Y(head_y + 6), WHITE, 1, 1)
        elif g == "kafra":
            band = Part(gc)
            band.ellipse(X(0), Y(head_y + 7.5), (hr + 0.4) * s, 2.6 * s, shading=False)
            cv.stamp(band)
            bow = Part(hx("#2a4ac0"))
            bow.ellipse(X(7 - fx * 1), Y(head_y + 9), 3.0 * s, 2.4 * s)
            bow.ellipse(X(-7 - fx * 1), Y(head_y + 9), 3.0 * s, 2.4 * s)
            cv.stamp(bow)
            cv.dot(X(-1), Y(head_y + 10), hx("#e0303a"), 3, 2)
        elif g == "cap":
            c = Part(gc)
            c.ellipse(X(0), Y(head_y + 6.5), (hr + 0.3) * s, 5.0 * s)
            if not back:
                c.ellipse(X(fx * 7), Y(head_y + 3), 6 * s, 1.8 * s)
            cv.stamp(c)
            cv.dot(X(-1), Y(head_y + 10), hx("#f0c040"), 3, 2)
        elif g == "horns":
            hm = Part(gc, shiny=True)
            hm.ellipse(X(0), Y(head_y + 4.5), (hr + 0.8) * s, 6.5 * s)
            cv.stamp(hm)
            hn = Part(hx("#f4ecd0"))
            for sgn in (-1, 1):
                hn.poly([(X(sgn * 7), Y(head_y + 7)), (X(sgn * 10), Y(head_y + 5)), (X(sgn * 14), Y(head_y + 15))])
            cv.stamp(hn)

    def draw_cast_fx():
        if p["cast"] < 0:
            return
        col = {"mage": hx("#ff5a3a"), "acolyte": hx("#fff0a0")}.get(name, hx("#7ad8ff"))
        rr = 14 * s
        for k in range(40):
            a = 2 * math.pi * k / 40
            cv.put(bx + math.cos(a) * rr, by - 1 + math.sin(a) * rr * 0.4, mix(col, WHITE, 0.3))
        for k in range(6):
            a = 2 * math.pi * (k / 6 + p["cast"] / 3)
            px_, py_ = bx + math.cos(a) * rr, by - 1 + math.sin(a) * rr * 0.4
            cv.dot(px_ - 1, py_ - 1, col, 3, 2)
            cv.put(px_, py_ - 2, WHITE)
        sp_y = Y(36) - (p["cast"] * 10) % 10
        cv.dot(X(-9), sp_y, col, 1, 2)
        cv.dot(X(9), sp_y + 4, col, 1, 2)

    draw_cast_fx()
    if back:
        draw_weapon_hand()
        draw_legs()
        draw_body()
        draw_arm(-near, -p["arm"])
        draw_head()
        draw_hair()
        draw_long_hair()
        draw_pack()
        draw_shield()
        draw_gear()
    else:
        draw_long_hair()
        draw_pack()
        if side:
            draw_arm(-near, -p["arm"])
        draw_legs()
        draw_body()
        if not side and p["arms_up"] == 0:
            draw_arm(-near, -p["arm"])
        draw_shield()
        draw_head()
        draw_hair()
        draw_gear()
        draw_face()
        if not side and p["arms_up"] > 0:
            draw_arm(-near, -p["arm"])
        draw_weapon_hand()


# ---------------------------------------------------------------- blobs (poring family, lunatic)
BLOBS = {
    "poring": dict(c="#ff8fc0", eye="#2a1a2a"),
    "drops": dict(c="#ff9a2a", eye="#2a1a2a"),
    "poporing": dict(c="#6ad84a", eye="#2a1a2a", leaf=True),
    "lunatic": dict(c="#fbf6ff", eye="#e8203a", ears=True, small=True),
}


def draw_blob(cv, name, d, p, ax, ay):
    sp = BLOBS[name]
    fx, fy = FACE[d]
    back = d >= 3
    lx, ly = lunge_xy(p, fx, fy)
    c = hx(sp["c"])
    r = 9.5 if sp.get("small") else 11.5
    sx, sy = p["sx"], p["sy"]
    rx, ry = r * sx * 1.08, r * 0.9 * sy
    cx = ax + lx
    hop = p["hop"]
    cy = ay + ly - ry - hop
    # ground shadow
    sh_r = rx * (1 - hop / 20)
    for x in range(int(cx - sh_r), int(cx + sh_r) + 1):
        cv.put(x, ay + ly, (60, 50, 70))
    if sp.get("ears"):
        ears = Part(c)
        for sgn in (-1, 1):
            ex_ = cx + sgn * 4 - fx * 2
            ears.ellipse(ex_ + sgn * 1.5, cy - ry - 5, 2.6, 7.0)
        cv.stamp(ears)
        if not back:
            for sgn in (-1, 1):
                cv.dot(cx + sgn * 4 - fx * 2 + sgn * 1.5, cy - ry - 8, hx("#ffa0c0"), 1, 6)
    body = Part(c, shiny=True)
    body.ellipse(cx, cy, rx, ry)
    cv.stamp(body)
    if sp.get("leaf"):
        lf = Part(hx("#3a9a2a"))
        lf.ellipse(cx + 2 - fx * 2, cy - ry - 2, 3.5, 2.0)
        cv.stamp(lf)
    if not back:
        style = p["eyes"]
        draw_eyes(cv, cx, cy - 1, fx, back, style, hx(sp["eye"]) if name == "lunatic" else (60, 40, 60),
                  sep=3.6 if sp.get("small") else 4.2)
        mx = cx + fx * 4 - 1
        if p["mouth"] > 0 or style == "hurt":
            cv.dot(mx, cy + 3, (170, 40, 70), 3, 2)
        else:
            cv.put(mx, cy + 3, (120, 40, 60)); cv.put(mx + 2, cy + 3, (120, 40, 60)); cv.put(mx + 1, cy + 4, (120, 40, 60))
        if abs(fx) < 0.9:
            cv.dot(cx - 7 + fx * 2, cy + 1, mix(c, (255, 60, 110), 0.4), 2, 1)
            cv.dot(cx + 6 + fx * 2, cy + 1, mix(c, (255, 60, 110), 0.4), 2, 1)
    if name == "lunatic":
        if not back:
            cv.dot(cx + fx * 5 - 1, cy + 5, hx("#ffd0e0"), 2, 1)
        else:
            tail = Part(c)
            tail.ellipse(cx, cy + ry - 3, 3, 3)
            cv.stamp(tail)


# ---------------------------------------------------------------- bugs and larvae
def depth_sorted(items):
    return sorted(items, key=lambda it: it[0])


def draw_fabre(cv, d, p, ax, ay):
    fx, fy = FACE[d]
    back = d >= 3
    lx, ly = lunge_xy(p, fx, fy)
    c = hx("#8ed84a")
    phase = p["leg"]
    k = 7.0 + phase * 1.5
    segs = []
    for i, r in enumerate((8.5, 7.8, 6.8, 5.6)):
        u = (1.5 - i) * k * p["sx"]
        x = ax + lx + fx * u
        y = ay + ly - r - 1 + fy * u * 0.45 - (abs(phase) * 2.5 if i == 1 else 0) * 1.0
        segs.append((y, i, x, r))
    for y, i, x, r in depth_sorted(segs):
        part = Part(c if i else hx("#a8e860"), shiny=(i == 0))
        part.ellipse(x, y - (1 - p["sy"]) * 4, r * p["sx"], r * p["sy"])
        cv.stamp(part)
        if i == 0:
            hx_, hy_ = x, y
    if not back:
        draw_eyes(cv, hx_, hy_, fx, back, p["eyes"], (50, 30, 50), sep=2.6, big=False)
        if p["mouth"]:
            cv.dot(hx_ + fx * 3, hy_ + 3, (160, 40, 60), 2, 1)
    ant = Part(hx("#ffd84a"))
    for sgn in (-1, 1):
        ant.ellipse(hx_ + sgn * 3 + fx * 2, hy_ - 8, 1.6, 1.6)
    cv.stamp(ant)
    for sgn in (-1, 1):
        for t in range(3):
            cv.put(hx_ + sgn * 2.5 + fx * 1.5, hy_ - 5 + t, (60, 90, 30))


def draw_bug(cv, name, d, p, ax, ay):
    gold = name == "golden_thief_bug"
    s = 1.9 if gold else 1.3
    fx, fy = FACE[d]
    back = d >= 3
    lx, ly = lunge_xy(p, fx, fy)
    shell_c = hx("#f6c81a") if gold else hx("#8a4a2a")
    cx = ax + lx
    cy = ay + ly - 6 * s - p["bob"] * 0.6
    # legs
    legs = Part(hx("#2a1a1a") if not gold else hx("#b07a10"))
    for k in (-1, 0, 1):
        for sgn in (-1, 1):
            off = k * 4 * s
            lift = p["leg"] * k * sgn * 1.5
            bx_ = cx + fx * off + (-fy) * sgn * 6 * s
            by_ = cy + fy * off * 0.45 + fx * sgn * 2 * s
            ex = bx_ + (-fy) * sgn * 4 * s
            ey = by_ + 5 * s - lift
            legs.thick_line(bx_, by_, ex, ey, 1.6)
    cv.stamp(legs)
    head_x, head_y = cx + fx * 7 * s, cy + fy * 3 * s
    def head():
        h = Part(hx("#3a2a2a") if not gold else hx("#d8a010"))
        h.ellipse(head_x, head_y, 4.2 * s, 3.8 * s)
        cv.stamp(h)
        ant = Part(hx("#3a2a2a") if not gold else hx("#d8a010"))
        for sgn in (-1, 1):
            ant.thick_line(head_x + sgn * 2 * s, head_y - 2 * s, head_x + sgn * 4 * s + fx * 5 * s,
                           head_y - 7 * s, 1.2, shading=False)
        cv.stamp(ant)
        if not back:
            if gold and p["eyes"] == "open":  # red glowing eyes with a hot core
                side_ = abs(fx) > 0.9
                for ex in ([head_x + fx * 5] if side_ else [head_x - 2.2 * s + fx * 3, head_x + 2.2 * s + fx * 3]):
                    cv.dot(ex - 2, head_y - 2, (255, 70, 40), 4, 3)
                    cv.dot(ex - 1, head_y - 1, (255, 200, 120), 2, 1)
                    cv.put(ex - 2, head_y - 2, WHITE)
            else:
                draw_eyes(cv, head_x, head_y, fx, back, p["eyes"] if p["eyes"] != "open" else "hollow",
                          (255, 255, 255), sep=2.0 * s, big=False)
            if gold and p["mouth"]:
                cv.dot(head_x + fx * 4 - 1, head_y + 5, (150, 30, 50), 3, 2)
        if gold:  # little crown: three gold points and a ruby
            cr = Part(hx("#ffe040"), shiny=True)
            ccx, ccy = head_x - fx * 1.0, head_y - 4.6 * s
            cr.poly([(ccx - 5, ccy + 1), (ccx + 5, ccy + 1), (ccx + 5, ccy - 3), (ccx + 2.5, ccy - 1),
                     (ccx, ccy - 5), (ccx - 2.5, ccy - 1), (ccx - 5, ccy - 3)])
            cv.stamp(cr)
            cv.dot(ccx - 1, ccy - 1, (230, 30, 60), 2, 2)
            cv.put(ccx - 1, ccy - 1, (255, 190, 200))
    if back:
        head()
    shell = Part(shell_c, shiny=True)
    shell.ellipse(cx - fx * 1.5 * s, cy - 1.5 * s, (6.5 + 2.5 * abs(fx)) * s * p["sx"], 6.0 * s * p["sy"])
    cv.stamp(shell)
    split = shade_of(shell_c)
    if abs(fx) < 0.9:
        for t in range(int(-5 * s), int(4 * s)):
            cv.put(cx - fx * 1.5 * s, cy - 1.5 * s + t, split)
    else:
        for t in range(int(-6 * s), int(5 * s)):
            cv.put(cx + t, cy - 3 * s, split)
    if gold:
        # shining shell: a diagonal highlight band across the shell pixels plus sparkles
        bandc = (255, 252, 205)
        for (px_, py_), col in list(shell.px.items()):
            u = (px_ - (cx - fx * 1.5 * s)) * 0.8 + (py_ - (cy - 1.5 * s)) * 1.0
            if -9.5 < u < -6.5 and col != shell.line:
                cv.put(px_, py_, bandc)
            elif -5.0 < u < -4.0 and col != shell.line:
                cv.put(px_, py_, mix(col, bandc, 0.6))
        cv.dot(cx + 5 * s, cy - 6 * s, WHITE, 1, 3)
        cv.dot(cx + 4 * s, cy - 5 * s, WHITE, 3, 1)
        cv.dot(cx - 9 * s, cy + 2 * s, (255, 245, 150), 1, 2)
    if not back:
        head()


# ---------------------------------------------------------------- plants
def draw_willow(cv, d, p, ax, ay):
    fx, fy = FACE[d]
    back = d >= 3
    lx, ly = lunge_xy(p, fx, fy)
    sway = p["leg"] * 2
    cx = ax + lx
    by = ay + ly - p["bob"] - p["hop"] * 0.25
    roots = Part(hx("#7a4a22"))
    for sgn in (-1, 0, 1):
        roots.ellipse(cx + sgn * 5 + (p["leg"] * 2 * sgn), by - 2, 3, 2.2)
    cv.stamp(roots)
    trunk = Part(hx("#a0682e"))
    trunk.poly([(cx - 7 * p["sx"], by - 2), (cx + 7 * p["sx"], by - 2), (cx + 5 + sway, by - 26 * p["sy"]),
                (cx - 5 + sway, by - 26 * p["sy"])])
    cv.stamp(trunk)
    arms = Part(hx("#8a5a26"))
    reach = p["arm"]
    for sgn in (-1, 1):
        arms.thick_line(cx + sgn * 5 + sway * 0.5, by - 17, cx + sgn * 12 + fx * reach * 4, by - 22 - reach * sgn * 3,
                        2.4)
    cv.stamp(arms)
    crown = Part(hx("#4ac040"), shiny=True)
    top = by - 26 * p["sy"]
    for ox, oy, r in ((-7, -4, 7), (7, -4, 7), (0, -10, 8), (-4, 2, 6), (5, 2, 6)):
        crown.ellipse(cx + sway + ox, top + oy, r, r * 0.85)
    cv.stamp(crown)
    if not back:
        fy_ = by - 15
        style = p["eyes"]
        ex = cx + sway * 0.6 + fx * 2
        if style in ("hurt", "dead"):
            draw_eyes(cv, ex, fy_, fx, back, style, EYE_DARK, sep=2.8, big=False)
        else:
            for sgn in ((-1, 1) if abs(fx) < 0.9 else (1,)):
                xx = ex + sgn * 2.8 if abs(fx) < 0.9 else ex + fx * 3
                cv.dot(xx - 1, fy_ - 1, (40, 20, 10), 2, 3)
                cv.put(xx, fy_ - 1, (255, 220, 120))
        cv.dot(ex - 1 + fx * 2, fy_ + 5, (40, 20, 10), 3, 2 if p["mouth"] else 1)


def draw_spore(cv, d, p, ax, ay):
    fx, fy = FACE[d]
    back = d >= 3
    lx, ly = lunge_xy(p, fx, fy)
    cx = ax + lx
    by = ay + ly - p["hop"] * 0.8
    sx, sy = p["sx"], p["sy"]
    for x in range(int(cx - 8), int(cx + 9)):
        cv.put(x, ay + ly, (60, 50, 70))
    stem = Part(hx("#fff0d0"))
    stem.ellipse(cx, by - 7 * sy, 7 * sx, 7.5 * sy)
    cv.stamp(stem)
    if not back:
        draw_eyes(cv, cx, by - 8 * sy, fx, back, p["eyes"], (60, 40, 60), sep=3.0, big=True)
        draw_mouth(cv, cx + fx * 3 - 1, by - 4 * sy, p["mouth"] > 0)
    cap = Part(hx("#e8402a"), shiny=True)
    cap.ellipse(cx, by - 16 * sy, 12 * sx, 8 * sy)
    cap.poly([(cx - 12 * sx, by - 16 * sy), (cx + 12 * sx, by - 16 * sy), (cx + 10 * sx, by - 12 * sy),
              (cx - 10 * sx, by - 12 * sy)])
    cv.stamp(cap)
    for ox, oy, r in ((-6, -18, 2), (3, -21, 2.2), (6, -15, 1.6), (-2, -14, 1.4)):
        spot = Part(WHITE, line=hx("#a02a1a"))
        spot.ellipse(cx + ox * sx - fx * 2, by + oy * sy, r, r * 0.9, shading=False)
        cv.stamp(spot)


# ---------------------------------------------------------------- critters
def draw_rocker(cv, d, p, ax, ay):
    fx, fy = FACE[d]
    back = d >= 3
    lx, ly = lunge_xy(p, fx, fy)
    cx = ax + lx
    by = ay + ly - p["bob"] - p["hop"] * 0.3
    c = hx("#7ac83a")
    legs = Part(hx("#5a9a2a"))
    for sgn in (-1, 1):
        kx = cx + sgn * 6 - fx * 2
        legs.thick_line(cx + sgn * 3, by - 9, kx, by - 14 + p["leg"] * sgn * 2, 2.6)
        legs.thick_line(kx, by - 14 + p["leg"] * sgn * 2, cx + sgn * 4, by - 1, 2.0)
    cv.stamp(legs)
    violin = Part(hx("#b0602a"), shiny=True)
    vx = cx + (6 if not back else -6) + fx * 3
    violin.ellipse(vx, by - 14, 3.2, 4.5)
    if not back:
        pass
    body = Part(c)
    body.ellipse(cx, by - 13 * p["sy"], 5.5 * p["sx"], 8 * p["sy"])
    if back:
        cv.stamp(violin)
    cv.stamp(body)
    if not back:
        for k in range(3):
            cv.dot(cx - 2 + fx * 2, by - 16 + k * 3, hx("#d8f080"), 4, 1)
    head = Part(c, shiny=True)
    hy_ = by - 25 * p["sy"]
    head.ellipse(cx + fx * 2, hy_, 6.5, 5.5)
    cv.stamp(head)
    ant = Part(hx("#5a9a2a"))
    for sgn in (-1, 1):
        ant.thick_line(cx + sgn * 2, hy_ - 4, cx + sgn * 6 - fx * 3, hy_ - 12, 1.2, shading=False)
    cv.stamp(ant)
    if not back:
        style = p["eyes"]
        if style == "open":
            for sgn in ((-1, 1) if abs(fx) < 0.9 else (1,)):
                ex_ = cx + fx * 4 + (sgn * 3.5 if abs(fx) < 0.9 else 0)
                e = Part(hx("#e83a2a"), shiny=True)
                e.ellipse(ex_, hy_ - 1, 2.4, 2.8)
                cv.stamp(e)
        else:
            draw_eyes(cv, cx + fx * 2, hy_ - 1, fx, back, style, EYE_DARK, sep=3.5)
        cv.stamp(violin)
        bow_ang = p["arm"] * 0.6
        bowp = Part(hx("#f0e0c0"))
        bowp.thick_line(vx - 6, by - 18 + bow_ang * 4, vx + 6, by - 12 - bow_ang * 4, 1.0, shading=False)
        cv.stamp(bowp)


def draw_hode(cv, d, p, ax, ay):
    fx, fy = FACE[d]
    back = d >= 3
    lx, ly = lunge_xy(p, fx, fy)
    cx = ax + lx
    by = ay + ly
    mound = Part(hx("#a0703a"))
    mound.ellipse(cx, by - 3, 12, 4.5)
    cv.stamp(mound)
    for ox in (-8, -3, 4, 9):
        cv.put(cx + ox, by - 5, hx("#c89a5a"))
    rise = 14 + p["bob"] * 1.5 + p["hop"] * 0.5 + p["lunge"] * 0.6
    worm = Part(hx("#e8c060"), shiny=True)
    worm.ellipse(cx + fx * 1.5, by - 5 - rise * 0.45 * p["sy"], 5.5 * p["sx"], rise * 0.55 * p["sy"])
    cv.stamp(worm)
    hy_ = by - 5 - rise * 0.8 * p["sy"]
    for k in range(3):
        cv.dot(cx - 4 + fx * 2, hy_ + 6 + k * 3, hx("#c89a40"), 8, 1)
    if not back:
        draw_eyes(cv, cx + fx * 1.5, hy_, fx, back, p["eyes"] if p["eyes"] != "open" else "dots",
                  EYE_DARK, sep=2.4, big=False)
        nose = Part(hx("#ff7a9a"))
        nose.ellipse(cx + fx * 4, hy_ + 2.5, 1.8, 1.4)
        cv.stamp(nose)
        if p["mouth"]:
            cv.dot(cx + fx * 4 - 1, hy_ + 5, (150, 40, 60), 2, 1)
    front_dirt = Part(hx("#8a5a2a"))
    front_dirt.ellipse(cx, by - 1.5, 9, 2.5)
    cv.stamp(front_dirt)


QUADS = {
    "wolf": dict(c="#8a9ac8", belly="#e8eef8", s=1.3, ears="point", eye="#ffd020", tail="bushy", snout=True),
    "bigfoot": dict(c="#a0642a", belly="#d8a060", s=1.75, ears="round", eye="#2a1a1a", tail="stub", snout=True),
    "eddga": dict(c="#ff9a1a", belly="#fff4d8", s=1.8, ears="round", eye="#30c040", tail="bush", snout=True,
                  stripes=True, mvp=True),
}


def draw_quad(cv, name, d, p, ax, ay):
    sp = QUADS[name]
    s = sp["s"]
    fx, fy = FACE[d]
    back = d >= 3
    lx, ly = lunge_xy(p, fx, fy)
    c = hx(sp["c"])
    L, R, H = 8 * s, 6 * s, 6 * s
    cx = ax + lx
    gy = ay + ly
    bob = p["bob"] * s
    leg = p["leg"]

    def scr(u, v, h):
        return cx + u * fx - v * fy, gy + (u * fy + v * fx) * 0.35 - h

    items = []
    for u, v, ph in ((0.6, -0.55, 1), (0.6, 0.55, -1), (-0.6, -0.55, -1), (-0.6, 0.55, 1)):
        x, y = scr(u * L, v * R, 0)
        lift = max(0, leg * ph) * 2.5 * s
        items.append(("leg", y, x, lift))
    tx, ty = scr(-L * 1.05, 0, H + R * 0.6 - bob)
    items.append(("tail", (-L) * fy * 0.35 + gy, tx, ty))
    hxp, hyp = scr(L * 0.95, 0, H + R * (0.95 if back else 0.55) - bob)
    head_depth = gy + L * fy * 0.35 + (0.1 if fy >= 0 else -0.1)
    items.append(("head", head_depth, hxp, hyp))
    body_x, body_y = cx, gy - H - R * 0.55 - bob
    items.append(("body", gy, body_x, body_y))

    def draw(kind, a, b):
        if kind == "leg":
            lg = Part(shade_of(c))
            lg.ellipse(a, gy + 0 - b - 3 * s + (0), 2.4 * s, 3.8 * s)
            cv.stamp(lg)
        elif kind == "tail":
            t = Part(c)
            if sp["tail"] == "bushy":
                t.ellipse(a, b - 2 * s, 3.2 * s, 4.5 * s)
            elif sp["tail"] == "bush":  # big bushy striped tail curling up behind
                t.thick_line(a, b, a - fx * 6 * s, b - 5 * s, 4.2 * s)
                t.ellipse(a - fx * 7 * s, b - 8 * s, 3.6 * s, 4.6 * s)
                cv.stamp(t)
                for k in range(3):
                    ty_ = b - 4 * s - k * 2.6 * s
                    tx_ = a - fx * (4 + k * 1.2) * s
                    cv.dot(tx_ - 2, ty_, (40, 24, 20), 5, 1)
                t = Part(hx("#fff4d8"))
                t.ellipse(a - fx * 7.5 * s, b - 11.5 * s, 1.8 * s, 1.6 * s, shading=False)
            elif sp["tail"] == "long":
                t.thick_line(a, b, a - fx * 5 * s, b - 6 * s, 2.4 * s)
                t.ellipse(a - fx * 5 * s, b - 6 * s, 1.8 * s, 1.8 * s)
            else:
                t.ellipse(a, b, 2 * s, 2 * s)
            cv.stamp(t)
        elif kind == "body":
            bd = Part(c)
            bd.ellipse(a, b, (R * 0.95 + (L - R * 0.95 + 2) * abs(fx)) * p["sx"], R * 0.9 * p["sy"])
            cv.stamp(bd)
            if sp.get("stripes"):
                for k in (-3, -1.5, 0, 1.5, 3):
                    for t_ in range(int(3.4 * s)):
                        xx = a + k * 2.6 * s * (abs(fx) * 0.8 + 0.4)
                        if cv.get(int(xx), int(b - R * 0.8 + t_)) not in (None,):
                            cv.put(xx, b - R * 0.8 + t_, (40, 24, 20))
                            cv.put(xx + 1, b - R * 0.8 + t_ + 1, (40, 24, 20))
            if not back:
                bl = Part(hx(sp["belly"]))
                bl.ellipse(a + fx * L * 0.3, b + R * 0.35, R * 0.5, R * 0.35, shading=False)
                cv.stamp(bl)
        elif kind == "head":
            ears = Part(c)
            for sgn in (-1, 1):
                ex_ = a + sgn * 4 * s * (1 - abs(fx) * 0.5) - fx * 1.5 * s
                if sp["ears"] == "point":
                    ears.poly([(ex_ - 2.2 * s, b - 3 * s), (ex_ + 2.2 * s, b - 3 * s), (ex_, b - 9 * s)])
                else:
                    ears.ellipse(ex_, b - 5 * s, 2.4 * s, 2.4 * s)
            cv.stamp(ears)
            hd = Part(c, shiny=True)
            hd.ellipse(a, b, 5.5 * s, 4.8 * s)
            cv.stamp(hd)
            if sp.get("stripes"):
                cv.dot(a - 1, b - 4.5 * s, (40, 24, 20), 2, int(2 * s))
                for k in (-3.5, 3.5):
                    cv.dot(a + k * s - 1, b - 3.5 * s, (40, 24, 20), 2, int(1.6 * s))
            if not back and sp["snout"]:
                sn = Part(hx(sp["belly"]))
                sn.ellipse(a + fx * 3.5 * s, b + 2 * s, 3 * s, 2.2 * s)
                cv.stamp(sn)
                cv.dot(a + fx * 5 * s - 0.5, b + 1 * s, (30, 20, 30), 2, 1)
                if p["mouth"] and sp.get("mvp"):  # fierce roar: wide open maw, fangs and tongue
                    mx_, my_ = a + fx * 3.0 * s - 3, b + 2.6 * s
                    cv.dot(mx_ - 1, my_ - 1, (40, 10, 20), 8, 7)
                    cv.dot(mx_, my_ + 3, (230, 80, 100), 6, 2)
                    cv.dot(mx_ - 1, my_ - 1, WHITE, 2, 3)
                    cv.dot(mx_ + 5, my_ - 1, WHITE, 2, 3)
                    cv.dot(mx_ + 1, my_ + 4, WHITE, 1, 2)
                    cv.dot(mx_ + 4, my_ + 4, WHITE, 1, 2)
                elif p["mouth"]:
                    cv.dot(a + fx * 3.5 * s - 1, b + 3.5 * s, (170, 30, 50), 3, 2)
                    cv.put(a + fx * 3.5 * s - 1, b + 3.5 * s, WHITE)
            if not back:
                style = p["eyes"]
                draw_eyes(cv, a, b - 1 * s, fx, back, style, hx(sp["eye"]), sep=2.6 * s, big=s > 1.2)
                if sp.get("mvp"):
                    # eye patch over one eye (screen-right), a scar over the other, brow ridge for a fierce look
                    side_ = abs(fx) > 0.9
                    pex = a + (fx * 5 if side_ else 2.6 * s + fx * 3)
                    if not side_ or fx < 0:
                        pt_ = Part(hx("#2a1a2a"))
                        pt_.ellipse(pex, b - 1 * s, 1.7 * s, 1.6 * s, shading=False)
                        cv.stamp(pt_)
                        cv.dot(pex - 1, b - 1 * s - 1, hx("#6a5a7a"), 1, 1)
                        for k in range(4):  # strap up over the brow
                            cv.put(pex - 2 + k, b - 1 * s - 3 - k * 0.5, (30, 20, 30))
                    scx = a - 2.6 * s + fx * 3 if not side_ else a + fx * 5
                    for k in range(int(3.5 * s)):  # scar slash across the eye
                        cv.put(scx - 2 + k * 0.7, b - 1 * s - 4 + k, (255, 90, 90))
                        cv.put(scx - 1 + k * 0.7, b - 1 * s - 4 + k, (150, 30, 40))
                    cv.dot(a - 6 * s * (1 - 0.5 * abs(fx)) , b - 4 * s, (40, 24, 20), int(4 * s), 1)

    for kind, _depth, a, b in sorted(items, key=lambda it: (it[1], {"leg": 0, "tail": 1, "body": 2, "head": 3}[it[0]])):
        if kind == "head" and not back and fy >= 0:
            continue
        draw(kind, a, b)
    if not back and fy >= 0:
        draw("head", hxp, hyp)
    if name == "eddga" and not back:
        # MVP aura sparkle
        cv.dot(cx - 14 * s, gy - 22 * s, (255, 240, 120), 1, 2)


def draw_peco(cv, d, p, ax, ay):
    fx, fy = FACE[d]
    back = d >= 3
    lx, ly = lunge_xy(p, fx, fy)
    cx = ax + lx
    gy = ay + ly
    bob = p["bob"] * 1.5
    c = hx("#ffb02a")
    legs = Part(hx("#e8701a"))
    for sgn in (-1, 1):
        stride = p["leg"] * sgn * 3
        kx = cx + sgn * 3.5 * (1 - abs(fx) * 0.7) + fx * stride
        legs.thick_line(cx + sgn * 3 * (1 - abs(fx) * 0.7), gy - 16 - bob, kx, gy - 1 - max(0, p["leg"] * sgn) * 2, 2.0)
        legs.ellipse(kx + fx * 1.5, gy - 1 - max(0, p["leg"] * sgn) * 2, 2.6, 1.3, shading=False)
    cv.stamp(legs)
    tail = Part(hx("#3a8adf"))
    tail.ellipse(cx - fx * 10, gy - 24 - bob + (0 if not back else 4), 4, 5)
    body = Part(c)
    body.ellipse(cx, gy - 22 * p["sy"] - bob, (10 + 3 * abs(fx)) * p["sx"], 9 * p["sy"])
    if not back:
        cv.stamp(tail)
    cv.stamp(body)
    if back:
        cv.stamp(tail)
    wing = Part(hx("#ffd85a"))
    wing.ellipse(cx - fx * 2 + (0 if abs(fx) > 0.5 else 8), gy - 23 - bob - p["arm"] * 2, 4, 6)
    cv.stamp(wing)
    neck = Part(c)
    hx_, hy_ = cx + fx * 7, gy - 38 - bob + fy * 1 - p["lunge"] * 0.2
    neck.thick_line(cx + fx * 4, gy - 26 - bob, hx_, hy_ + 3, 6)
    cv.stamp(neck)
    head = Part(c, shiny=True)
    head.ellipse(hx_, hy_, 6.5, 6)
    cv.stamp(head)
    crest = Part(hx("#e83a2a"))
    crest.ellipse(hx_ - fx * 3, hy_ - 6, 2.5, 3.5)
    cv.stamp(crest)
    if not back:
        beak = Part(hx("#ffe060"))
        open_ = 2 if p["mouth"] else 0
        beak.poly([(hx_ + fx * 3 - 2.5, hy_ + 1), (hx_ + fx * 3 + 2.5, hy_ + 1), (hx_ + fx * 9, hy_ + 3 + open_)])
        cv.stamp(beak)
        draw_eyes(cv, hx_, hy_ - 1, fx, back, p["eyes"], (40, 30, 60), sep=2.8)


# ---------------------------------------------------------------- kinds
def kind_spec(name):
    """-> (frame_w, frame_h, anchor_x, anchor_y, draw_fn, dead_style, actions)"""
    if name in JOBS:
        return 48, 64, 24, 60, lambda cv, d, p, ax, ay: draw_humanoid(cv, name, d, p, ax, ay), "fall", PLAYER_ACTIONS
    if name in NPCS:
        return 48, 64, 24, 60, lambda cv, d, p, ax, ay: draw_humanoid(cv, name, d, p, ax, ay), "fall", NPC_ACTIONS
    if name in ("zombie", "skeleton", "soldier_skeleton", "mummy"):
        return 48, 64, 24, 60, lambda cv, d, p, ax, ay: draw_humanoid(cv, name, d, p, ax, ay), "fall", MONSTER_ACTIONS
    if name == "orc_warrior":
        return 56, 72, 28, 68, lambda cv, d, p, ax, ay: draw_humanoid(cv, name, d, p, ax, ay), "fall", MONSTER_ACTIONS
    if name in BLOBS:
        return 40, 40, 20, 36, lambda cv, d, p, ax, ay: draw_blob(cv, name, d, p, ax, ay), "flat", MONSTER_ACTIONS
    if name == "fabre":
        return 48, 40, 24, 36, draw_fabre, "flat", MONSTER_ACTIONS
    if name == "thief_bug":
        return 48, 40, 24, 33, lambda cv, d, p, ax, ay: draw_bug(cv, name, d, p, ax, ay), "flip", MONSTER_ACTIONS
    if name == "golden_thief_bug":
        return 72, 56, 36, 46, lambda cv, d, p, ax, ay: draw_bug(cv, name, d, p, ax, ay), "flip", MONSTER_ACTIONS
    if name == "willow":
        return 48, 64, 24, 60, draw_willow, "fall", MONSTER_ACTIONS
    if name == "spore":
        return 40, 48, 20, 44, draw_spore, "flat", MONSTER_ACTIONS
    if name == "rocker":
        return 48, 56, 24, 52, draw_rocker, "fall", MONSTER_ACTIONS
    if name == "hode":
        return 40, 40, 20, 36, draw_hode, "flat", MONSTER_ACTIONS
    if name == "wolf":
        return 56, 48, 28, 44, lambda cv, d, p, ax, ay: draw_quad(cv, name, d, p, ax, ay), "fall", MONSTER_ACTIONS
    if name == "bigfoot":
        return 72, 64, 36, 58, lambda cv, d, p, ax, ay: draw_quad(cv, name, d, p, ax, ay), "fall", MONSTER_ACTIONS
    if name == "eddga":
        return 80, 64, 40, 58, lambda cv, d, p, ax, ay: draw_quad(cv, name, d, p, ax, ay), "fall", MONSTER_ACTIONS
    if name == "peco_peco":
        return 56, 64, 28, 60, draw_peco, "fall", MONSTER_ACTIONS
    raise KeyError(name)


def render_frame(name, action, d, i, n):
    fw, fh, ax, ay, fn, dead_style, _ = kind_spec(name)
    p = pose_for(action, i, n)
    upright = dict(p)
    if action == "dead" and dead_style == "fall":
        upright["sx"], upright["sy"] = 1.0, 1.0  # the fall is a rotation, not a squash
    if action == "dead" and dead_style == "flip":
        upright["sx"], upright["sy"] = 1.0, max(0.6, p["sy"])
    cv = Canvas(fw, fh)
    fn(cv, d, upright, ax, ay)
    if action == "dead" and dead_style in ("fall", "flip") and p["fall"] > 0:
        bb = cv.bbox()
        if bb:
            sign = 1 if d <= 2 else -1
            angle = sign * 90 * p["fall"] if dead_style == "fall" else 180 * p["fall"]
            pivot_y = (bb[1] + bb[3]) / 2
            cv = cv.rotated(angle, ax, pivot_y)
            bb2 = cv.bbox()
            if bb2:
                cv = cv.shifted(int(round(ax - (bb2[0] + bb2[2]) / 2)), int(round(ay - bb2[3])))
    if p["flash"]:
        cv.tint(p["flash"][0], p["flash"][1])
    return cv


def build_kind(name, out_dir):
    fw, fh, ax, ay, _fn, _dead, actions = kind_spec(name)
    kdir = os.path.join(out_dir, name)
    os.makedirs(kdir, exist_ok=True)
    act = {"kind": name, "frame_w": fw, "frame_h": fh, "anchor": [ax, ay], "atk_frame": ATK_FRAME,
           "rows": N_ROWS, "directions": ["S", "SW", "W", "NW", "N", "NE", "E", "SE"],
           "mirror_rows": {"5": 3, "6": 2, "7": 1}, "actions": {}}
    for action in actions:
        n, delay, loop = CLIP[action]
        W, H = fw * n, fh * N_ROWS
        pix = [None] * (W * H)
        for d in range(N_ROWS):
            for i in range(n):
                cv = render_frame(name, action, d, i, n)
                for y in range(fh):
                    row = cv.p[y * fw:(y + 1) * fw]
                    base = (d * fh + y) * W + i * fw
                    pix[base:base + fw] = row
        fname = "%s.png" % action
        write_png(os.path.join(kdir, fname), W, H, pix)
        entry = {"file": fname, "image": "res://assets/anim/%s/%s" % (name, fname), "row0": 0, "frames": n, "delay_ms": delay, "mirror": True, "loop": loop}
        if action == "attack":
            entry["atk_frame"] = ATK_FRAME
        act["actions"][action] = entry
    with open(os.path.join(kdir, "act.json"), "w", newline="\n") as f:
        json.dump(act, f, indent=1)
        f.write("\n")
    return act


# ---------------------------------------------------------------- contact sheet
def contact_sheet(out_dir, kinds):
    """All kinds x idle f0 x dir 0 (jobs row, then monsters 7 per row), novice 8 directions, walk and attack
    strips; each cell 80x80 on a grass tile, scaled x2."""
    cell = 80
    rows = []
    jobs = [k for k in kinds if k in JOBS]
    mons = [k for k in kinds if k not in JOBS and k not in NPCS]
    rows.append([(k, "idle", 0, 0) for k in jobs])
    for i in range(0, len(mons), 7):
        rows.append([(k, "idle", 0, 0) for k in mons[i:i + 7]])
    if "novice" in kinds:
        rows.append([("novice", "idle", d, 0) for d in range(8)])
        rows.append([("novice", "walk", 2, i) for i in range(8)])
        rows.append([("swordsman", "attack", 1, i) for i in range(6)] + [("mage", "cast", 0, 0), ("mage", "cast", 0, 3)])
        rows.append([("thief", a, 0, f) for a, f in (("hurt", 0), ("hurt", 1), ("dead", 2), ("dead", 4),
                                                     ("sit", 0), ("pickup", 1), ("standby", 0))])
    if "poring" in kinds:
        rows.append([("poring", "walk", 0, i) for i in range(8)])
        rows.append([("poring", a, 0, f) for a, f in (("attack", 3), ("hurt", 0), ("dead", 4))]
                    + [("wolf", "walk", 2, i) for i in (0, 2, 4)])
        rows.append([("orc_warrior", "hurt", 0, i) for i in range(4)]
                    + [("poring", "hurt", 0, i) for i in range(2)] + [("thief", "hurt", 0, i) for i in range(2)])
        rows.append([("orc_warrior", "dead", 0, 0), ("poring", "dead", 0, 0), ("thief", "dead", 0, 0)])
    for mvp in ("golden_thief_bug", "eddga"):
        if mvp in kinds:
            rows.append([(mvp, "idle", 0, 0), (mvp, "idle", 2, 0), (mvp, "walk", 2, 2), (mvp, "walk", 0, 2),
                         (mvp, "attack", 2, 3), (mvp, "attack", 0, 3), (mvp, "dead", 2, 4), (mvp, "hurt", 2, 0)])
    if any(k in kinds for k in NPCS):
        rows.append([(k, a, d, f) for k in NPCS if k in kinds for d in (0, 2) for a, f in (("idle", 0), ("talk", 2))])
    ncols = max(len(r) for r in rows)
    W, H = ncols * cell, len(rows) * cell
    pix = [None] * (W * H)
    grass_a, grass_b = (120, 190, 90), (110, 178, 84)
    for y in range(H):
        for x in range(W):
            pix[y * W + x] = grass_a if ((x // 8) + (y // 8)) % 2 else grass_b
    cache = {}
    for r, row in enumerate(rows):
        for c, (k, a, d, f) in enumerate(row):
            n = CLIP[a][0]
            dd = d if d < N_ROWS else 8 - d
            key = (k, a, dd, f)
            if key not in cache:
                cache[key] = render_frame(k, a, dd, f, n)
            cv = cache[key]
            if d >= N_ROWS:
                cv = cv.flipped()
            fw, fh, ax, ay = kind_spec(k)[:4]
            ox = c * cell + cell // 2 - ax
            oy = r * cell + cell - 6 - ay
            for y in range(fh):
                for x in range(fw):
                    v = cv.p[y * fw + x]
                    X, Y = ox + x, oy + y
                    if v is not None and 0 <= X < W and 0 <= Y < H:
                        pix[Y * W + X] = v
    scaled = [None] * (W * 2 * H * 2)
    for y in range(H * 2):
        src = (y // 2) * W
        for x in range(W * 2):
            scaled[y * W * 2 + x] = pix[src + x // 2]
    review = os.path.join(out_dir, "_review")  # .gdignore: Godot neither imports nor exports it
    os.makedirs(review, exist_ok=True)
    open(os.path.join(review, ".gdignore"), "w").close()
    path = os.path.join(review, "contact_sheet.png")
    write_png(path, W * 2, H * 2, scaled)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", action="append", help="build only these kinds (repeatable)")
    ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--jobs", type=int, default=4, help="worker processes")
    a = ap.parse_args(argv)
    kinds = a.only or (JOBS + MONSTERS + NPCS)
    os.makedirs(a.out, exist_ok=True)
    if a.jobs > 1 and len(kinds) > 1:
        from multiprocessing import Pool
        with Pool(min(a.jobs, len(kinds))) as pool:
            pool.starmap(build_kind, [(k, a.out) for k in kinds])
    else:
        for k in kinds:
            build_kind(k, a.out)
    index = {"version": 1, "jobs": JOBS, "monsters": MONSTERS,
             "npcs": NPCS, "player_actions": PLAYER_ACTIONS, "monster_actions": MONSTER_ACTIONS,
             "npc_actions": NPC_ACTIONS}
    if not a.only:
        with open(os.path.join(a.out, "index.json"), "w", newline="\n") as f:
            json.dump(index, f, indent=1)
            f.write("\n")
    if not a.no_contact:
        print("contact sheet:", contact_sheet(a.out, kinds))
    print("built %d kinds into %s" % (len(kinds), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
