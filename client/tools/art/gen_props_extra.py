"""Extra town props in the cel look, drawn with Pillow and appended to client/assets/tiles/props.png as new rows
(existing cells untouched): a lamp post, a well, and the four shop stalls redrawn with striped awnings. Updates
tiles.json `props_rows` and `props.cells` (new cells appended, the stall cells repointed to the awning versions).

    python client/tools/art/gen_props_extra.py

Procedural, CC0. Cells are 64 px; the props are 1x2 (64x128) standing on their cell (base_px 32, origin [0, 32]).
"""
import json
import os

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "tiles"))
ATLAS = os.path.join(TILES, "props.png")
INDEX = os.path.join(TILES, "tiles.json")
CELL = 64
OUTLINE = (45, 30, 40, 255)
SS = 2


def tile(draw_fn):
    s = SS
    img = Image.new("RGBA", (CELL * s, CELL * 2 * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    draw_fn(d, img, CELL * s, CELL * 2 * s, s)
    return img.resize((CELL, CELL * 2), Image.LANCZOS)


def shadow(img, w, h, s):
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse((w * 0.18, h * 0.9, w * 0.82, h * 0.99), fill=(20, 20, 40, 90))
    return Image.alpha_composite(sh.filter(ImageFilter.GaussianBlur(2 * s)), img)


def lamp(d, img, w, h, s):
    lw = 3 * s
    base_y = h * 0.95
    # post
    d.rectangle((w * 0.46, h * 0.3, w * 0.54, base_y), fill=(60, 60, 75, 255), outline=OUTLINE, width=lw)
    d.polygon([(w * 0.34, base_y), (w * 0.66, base_y), (w * 0.6, base_y - h * 0.05), (w * 0.4, base_y - h * 0.05)], fill=(70, 70, 85, 255), outline=OUTLINE, width=lw)
    # lantern
    d.polygon([(w * 0.3, h * 0.3), (w * 0.7, h * 0.3), (w * 0.62, h * 0.12), (w * 0.38, h * 0.12)], fill=(80, 80, 95, 255), outline=OUTLINE, width=lw)
    d.rectangle((w * 0.34, h * 0.14, w * 0.66, h * 0.29), fill=(255, 225, 140, 255), outline=OUTLINE, width=lw)
    d.ellipse((w * 0.42, h * 0.17, w * 0.58, h * 0.27), fill=(255, 250, 220, 255))
    d.rectangle((w * 0.47, h * 0.06, w * 0.53, h * 0.12), fill=(60, 60, 75, 255))
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((w * 0.18, h * 0.04, w * 0.82, h * 0.4), fill=(255, 220, 120, 90))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(6 * s)))
    d2 = ImageDraw.Draw(img)
    d2.rectangle((w * 0.34, h * 0.14, w * 0.66, h * 0.29), outline=OUTLINE, width=lw)


def well(d, img, w, h, s):
    lw = 3 * s
    # roof
    d.polygon([(w * 0.1, h * 0.36), (w * 0.9, h * 0.36), (w * 0.5, h * 0.12)], fill=(200, 100, 70, 255), outline=OUTLINE, width=lw)
    d.line((w * 0.2, h * 0.33, w * 0.8, h * 0.33), fill=(150, 66, 50, 255), width=lw)
    # posts
    for x in (0.22, 0.78):
        d.rectangle((w * (x - 0.03), h * 0.36, w * (x + 0.03), h * 0.7), fill=(120, 80, 45, 255), outline=OUTLINE, width=lw)
    # crossbar and bucket
    d.rectangle((w * 0.2, h * 0.44, w * 0.8, h * 0.48), fill=(110, 70, 40, 255), outline=OUTLINE, width=lw)
    d.line((w * 0.5, h * 0.48, w * 0.5, h * 0.6), fill=(70, 50, 40, 255), width=2 * s)
    d.rectangle((w * 0.42, h * 0.6, w * 0.58, h * 0.7), fill=(130, 90, 50, 255), outline=OUTLINE, width=lw)
    # stone ring
    d.ellipse((w * 0.12, h * 0.6, w * 0.88, h * 0.95), fill=(150, 150, 165, 255), outline=OUTLINE, width=lw)
    d.ellipse((w * 0.22, h * 0.64, w * 0.78, h * 0.82), fill=(70, 110, 170, 255), outline=OUTLINE, width=lw)
    d.ellipse((w * 0.3, h * 0.67, w * 0.5, h * 0.74), fill=(140, 190, 240, 255))
    for x in (0.2, 0.4, 0.6):
        d.line((w * x, h * 0.84, w * (x + 0.08), h * 0.93), fill=(110, 110, 125, 255), width=2 * s)


def stall(colors):
    def fn(d, img, w, h, s):
        lw = 3 * s
        c1, c2 = colors
        # counter
        d.rectangle((w * 0.1, h * 0.62, w * 0.9, h * 0.94), fill=(150, 100, 55, 255), outline=OUTLINE, width=lw)
        d.rectangle((w * 0.1, h * 0.62, w * 0.9, h * 0.68), fill=(190, 140, 80, 255))
        # goods
        for k, (x, col) in enumerate([(0.25, (230, 80, 80)), (0.45, (240, 200, 70)), (0.65, (90, 180, 110))]):
            d.ellipse((w * (x - 0.07), h * 0.56, w * (x + 0.07), h * 0.65), fill=col + (255,), outline=OUTLINE, width=max(1, lw // 2))
        # posts
        for x in (0.14, 0.86):
            d.rectangle((w * (x - 0.025), h * 0.28, w * (x + 0.025), h * 0.62), fill=(120, 80, 45, 255), outline=OUTLINE, width=lw)
        # awning: scalloped striped canopy
        stripes = 6
        for k in range(stripes):
            x0 = w * (0.04 + 0.92 * k / stripes)
            x1 = w * (0.04 + 0.92 * (k + 1) / stripes)
            d.rectangle((x0, h * 0.14, x1, h * 0.3), fill=(c1 if k % 2 == 0 else c2) + (255,))
            d.ellipse((x0, h * 0.26, x1, h * 0.34), fill=(c1 if k % 2 == 0 else c2) + (255,))
        d.polygon([(w * 0.04, h * 0.14), (w * 0.96, h * 0.14), (w * 0.9, h * 0.06), (w * 0.1, h * 0.06)], fill=c1 + (255,), outline=OUTLINE, width=lw)
        d.line((w * 0.04, h * 0.14, w * 0.04, h * 0.3), fill=OUTLINE, width=lw)
        d.line((w * 0.96, h * 0.14, w * 0.96, h * 0.3), fill=OUTLINE, width=lw)
        d.line((w * 0.04, h * 0.14, w * 0.96, h * 0.14), fill=OUTLINE, width=lw)
    return fn


NEW = [
    ("lamp", lamp), ("well", well),
    ("stall0", stall(((225, 70, 70), (250, 245, 235)))), ("stall1", stall(((70, 110, 220), (250, 245, 235)))),
    ("stall2", stall(((70, 170, 100), (250, 245, 235)))), ("stall3", stall(((240, 190, 60), (250, 245, 235)))),
]


def main():
    atlas = Image.open(ATLAS).convert("RGBA")
    cols = atlas.width // CELL
    rows = atlas.height // CELL
    with open(INDEX, "r", encoding="utf-8") as fh:  # re-read right before the edit (shared file)
        index = json.load(fh)
    new_row = rows  # append a fresh 1x2 row pair
    out = Image.new("RGBA", (atlas.width, (rows + 2) * CELL), (0, 0, 0, 0))
    out.paste(atlas, (0, 0))
    cells = index["props"]["cells"]
    for i, (name, fn) in enumerate(NEW):
        img = tile(fn)
        img = shadow(img, img.width, img.height, 1)
        out.paste(img, (i * CELL, new_row * CELL), img)
        cells[name] = {"atlas": [i, new_row], "size": [1, 2], "origin": [0, 32], "base_px": 32}
    out.save(ATLAS, optimize=True)
    index["props_rows"] = rows + 2
    with open(INDEX, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    print("PROPS_EXTRA added=%d atlas=%dx%d rows=%d" % (len(NEW), out.width, out.height, rows + 2))


if __name__ == "__main__":
    main()
