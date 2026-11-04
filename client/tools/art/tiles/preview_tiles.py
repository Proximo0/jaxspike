"""Compose a map crop from the tileset with the same rules as map_renderer.gd (offline review, Pillow).

    python client/tools/art/tiles/preview_tiles.py TERRAIN_BIN X0 Y0 W H OUT.png [--px 32]

TERRAIN_BIN: raw 512x512 uint8 terrain, row-major y * 512 + x (as the welcome map). Also writes the atlas sheet
`_atlas_review.png` next to OUT. Mirrors ViewMath.tile_hash / autotile rows, so what this shows is what Godot draws.
"""
import argparse
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.normpath(os.path.join(HERE, "..", "..", "..", "assets", "tiles"))
N = 512


def tile_hash(x, y):
    h = (x * 73856093) ^ (y * 19349663)
    h = ((h ^ (h >> 13)) & 0x7FFFFFFF) * 1274126177  # masked first: GDScript ints are 64-bit
    return (h ^ (h >> 16)) & 0x7FFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("terrain")
    ap.add_argument("x0", type=int)
    ap.add_argument("y0", type=int)
    ap.add_argument("w", type=int)
    ap.add_argument("h", type=int)
    ap.add_argument("out")
    ap.add_argument("--px", type=int, default=32)
    ap.add_argument("--tiles", default=TILES, help="folder with tiles.json, terrain.png, props.png")
    a = ap.parse_args()
    tiles = a.tiles
    t = open(a.terrain, "rb").read()
    idx = json.load(open(os.path.join(tiles, "tiles.json")))
    cp = idx["cell_px"]
    atlas = Image.open(os.path.join(tiles, "terrain.png")).convert("RGBA")
    props = Image.open(os.path.join(tiles, "props.png")).convert("RGBA")

    def at(x, y):
        return t[y * N + x] if 0 <= x < N and 0 <= y < N else 255

    img = Image.new("RGBA", (a.w * cp, a.h * cp), (0, 0, 0, 255))
    over = []
    pmap = idx["props"]
    for y in range(a.y0, a.y0 + a.h):
        for x in range(a.x0, a.x0 + a.w):
            tid = at(x, y)
            e = idx["terrains"][tid]
            if e["mode"] == "blob47":
                c = e["connects"]
                m = 0
                for bit, (dx, dy) in ((1, (0, -1)), (2, (1, 0)), (4, (0, 1)), (8, (-1, 0)),
                                      (16, (1, -1)), (32, (1, 1)), (64, (-1, 1)), (128, (-1, -1))):
                    if at(x + dx, y + dy) in c:
                        m |= bit
                for diag, s1, s2 in ((16, 1, 2), (32, 4, 2), (64, 4, 8), (128, 1, 8)):
                    if m & diag and not (m & s1 and m & s2):
                        m &= ~diag
                row = idx["blob47_masks"].index(m)
            elif e["mode"] == "blob4":
                c = e["connects"]
                row = (1 if at(x, y - 1) in c else 0) | (2 if at(x + 1, y) in c else 0) | \
                      (4 if at(x, y + 1) in c else 0) | (8 if at(x - 1, y) in c else 0)
            elif e["mode"] == "variants":
                w = e["weights"]
                r = tile_hash(x, y) % sum(w)
                row = 0
                while r >= w[row]:
                    r -= w[row]
                    row += 1
            else:
                row = 0
            near = e.get("near")
            if near and any(at(x + dx, y + dy) in near["ids"] for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                row += near["offset"]
            img.paste(atlas.crop((tid * cp, row * cp, tid * cp + cp, row * cp + cp)), ((x - a.x0) * cp, (y - a.y0) * cp))
            names = pmap["on_terrain"].get(str(tid))
            if names:
                k = x if tid == 12 else tile_hash(x, y) >> 3
                over.append((x, y, names[k % len(names)]))
            bc = pmap["block_center"].get(str(tid))
            if bc and at(x - 1, y) != tid and at(x, y - 1) != tid:
                bw = 0
                while at(x + bw, y) == tid:
                    bw += 1
                bh = 0
                while at(x, y + bh) == tid:
                    bh += 1
                over.append((x + (bw - 1) // 2, y + (bh - 1) // 2, bc))
    for x, y, name in over:
        cell = pmap["cells"][name]
        c, r = cell["atlas"]
        w, h = cell["size"]
        spr = props.crop((c * cp, r * cp, (c + w) * cp, (r + h) * cp))
        # Godot: centred on the cell centre, minus texture_origin
        cx, cy = (x - a.x0) * cp + cp // 2, (y - a.y0) * cp + cp // 2
        px = cx - w * cp // 2 - cell["origin"][0]
        py = cy - h * cp // 2 - cell["origin"][1]
        img.alpha_composite(spr, (max(0, px), max(0, py)), (max(0, -px), max(0, -py)))
    if a.px != cp:
        img = img.resize((a.w * a.px, a.h * a.px), Image.LANCZOS)
    img.save(a.out)
    sheet = Image.new("RGBA", (atlas.width + props.width + cp, max(atlas.height, props.height)), (60, 60, 70, 255))
    sheet.alpha_composite(atlas, (0, 0))
    sheet.alpha_composite(props, (atlas.width + cp, 0))
    sheet.save(os.path.join(os.path.dirname(os.path.abspath(a.out)), "_atlas_review.png"))
    print("PREVIEW done %s" % a.out)


if __name__ == "__main__":
    main()
