"""Roof overlay tiles for the big decor wall blocks (Solmera, Kharsand): a 16-variant blob4 set drawn with Pillow in
the cel look, one 64x128 tile per wall cell (the lower cell is the wall's own cell, the upper 64 px is the roof
that rises above it). The variant index is the 4-bit mask of wall neighbours (N=1, E=2, S=4, W=8, as tiles.json
mask_bits): edges get eaves and a highlight ridge, free corners round off, a chimney sits on some interior tiles.
MapRenderer places them on the y-sorted Props layer (sort origin = the wall cell's bottom) so characters walk
behind a building they stand north of and in front of one they stand south of.

    python client/tools/art/gen_roofs.py [--out client/assets/tiles/roofs.png]

Three styles (tile red, slate blue, thatch) x three feature rows (plain, door, window) = 9 rows of 16 masks.

Writes <out> (16 x 1 tiles of 64x128) and prints one line. Procedural, CC0.
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "tiles", "roofs.png"))
CELL = 64
TILE_H = 128
LIFT = 44  # px the roof plane sits above the wall cell's top (the wall face shows below the eave)
OUTLINE = (45, 30, 40, 255)

STYLES = {
    "terracotta": dict(base=(214, 106, 72), light=(240, 150, 110), dark=(150, 66, 50), ridge=(250, 200, 160), eave=(90, 40, 35), course=10, stagger=True),
    "slate": dict(base=(96, 110, 140), light=(140, 155, 185), dark=(60, 70, 95), ridge=(200, 210, 230), eave=(40, 45, 60), course=8, stagger=True),
    "thatch": dict(base=(205, 170, 95), light=(235, 205, 130), dark=(150, 115, 55), ridge=(240, 220, 160), eave=(110, 80, 40), course=6, stagger=False),
}
STYLE_ORDER = ["terracotta", "slate", "thatch"]  # roofs.styles index 0..2 (tile red, slate blue, thatch)
N, E, S, W = 1, 2, 4, 8


def draw_tile(mask, st, s):
    """One tile at supersample s: the roof plane covers the full 64 px width and from y=LIFT-64.. (shifted up)."""
    W_, H_ = CELL * s, TILE_H * s
    img = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    top = (CELL - LIFT) * s  # the roof plane's top edge within the tile (the plane is one cell tall, lifted)
    bot = top + CELL * s
    lw = max(2, 3 * s)
    # roof plane: base fill with tile rows (horizontal courses) and a slight gradient top->bottom
    d.rectangle((0, top, W_, bot), fill=st["base"])
    course = st.get("course", 10) * s
    y = top + course
    while y < bot:
        d.line((0, y, W_, y), fill=st["dark"], width=max(1, s))
        y += course
    if st.get("stagger", True):
        # staggered tile seams
        y = top
        row = 0
        while y < bot:
            x = (row % 2) * course
            while x < W_:
                d.line((x, y, x, min(y + course, bot)), fill=st["dark"], width=max(1, s))
                x += course * 2
            y += course
            row += 1
    else:
        # thatch: short diagonal strokes per course
        y = top
        k = 0
        while y < bot:
            x = (k % 3) * 4 * s
            while x < W_:
                d.line((x, y + course - s, x + 3 * s, y + 2 * s), fill=st["light"], width=max(1, s))
                x += 9 * s
            y += course
            k += 1
    # highlights near the top of the plane (sun from the upper left)
    d.rectangle((0, top, W_, top + 6 * s), fill=st["light"])
    # edges: where no neighbour continues the roof
    if not mask & N:
        d.rectangle((0, top, W_, top + 5 * s), fill=st["ridge"])
        d.line((0, top, W_, top), fill=OUTLINE, width=lw)
    if not mask & S:
        # eave: a dark band at the bottom of the plane and a shadow on the wall face below
        d.rectangle((0, bot - 7 * s, W_, bot), fill=st["eave"])
        d.line((0, bot, W_, bot), fill=OUTLINE, width=lw)
        shadow = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rectangle((0, bot, W_, bot + 14 * s), fill=(20, 10, 20, 110))
        shadow = shadow.filter(ImageFilter.GaussianBlur(3 * s))
        img = Image.alpha_composite(img, shadow)
        d = ImageDraw.Draw(img)
    if not mask & W:
        d.line((0, top, 0, bot), fill=OUTLINE, width=lw)
        d.rectangle((0, top, 4 * s, bot), fill=st["light"])
    if not mask & E:
        d.line((W_ - 1, top, W_ - 1, bot), fill=OUTLINE, width=lw)
        d.rectangle((W_ - 5 * s, top, W_, bot), fill=st["dark"])
    return img


def draw_feature(img, feature, s):
    """A house front on the wall face below the eave: `door` (arched wooden door) or `window` (shutters + box)."""
    W_, H_ = img.size
    d = ImageDraw.Draw(img)
    lw = max(2, 3 * s)
    top = (CELL - LIFT) * s + CELL * s  # the eave's bottom edge; the wall face is below it
    face_bot = TILE_H * s - 4 * s
    if feature == "door":
        x0, x1 = W_ * 0.3, W_ * 0.7
        y0 = top + 4 * s
        d.rounded_rectangle((x0, y0, x1, face_bot), radius=int((x1 - x0) / 2), fill=(120, 75, 40, 255), outline=OUTLINE, width=lw)
        d.rectangle((x0, y0 + (face_bot - y0) * 0.45, x1, face_bot), fill=(120, 75, 40, 255), outline=OUTLINE, width=lw)
        for k in range(3):
            x = x0 + (x1 - x0) * (0.25 + 0.25 * k)
            d.line((x, y0 + 8 * s, x, face_bot - 3 * s), fill=(80, 50, 30, 255), width=max(1, s))
        d.ellipse((x1 - 12 * s, (y0 + face_bot) / 2, x1 - 7 * s, (y0 + face_bot) / 2 + 5 * s), fill=(240, 200, 90, 255))
    elif feature == "window":
        x0, x1 = W_ * 0.28, W_ * 0.72
        y0, y1 = top + 8 * s, top + 30 * s
        d.rectangle((x0, y0, x1, y1), fill=(150, 200, 235, 255), outline=OUTLINE, width=lw)
        d.line(((x0 + x1) / 2, y0, (x0 + x1) / 2, y1), fill=OUTLINE, width=max(1, lw // 2))
        d.line((x0, (y0 + y1) / 2, x1, (y0 + y1) / 2), fill=OUTLINE, width=max(1, lw // 2))
        for sx in (x0 - 9 * s, x1 + 1 * s):
            d.rectangle((sx, y0, sx + 8 * s, y1), fill=(90, 130, 90, 255), outline=OUTLINE, width=max(1, lw // 2))
        d.rectangle((x0 - 3 * s, y1, x1 + 3 * s, y1 + 7 * s), fill=(130, 85, 50, 255), outline=OUTLINE, width=max(1, lw // 2))
        for k in range(4):
            fx = x0 + (x1 - x0) * (0.12 + 0.25 * k)
            d.ellipse((fx - 3 * s, y1 - 5 * s, fx + 3 * s, y1 + 1 * s), fill=((240, 90, 110) if k % 2 else (250, 210, 80)) + (255,))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ss", type=int, default=2)
    args = ap.parse_args()
    features = ["plain", "door", "window"]
    # rows: style 0 features 0..2, style 1 features 0..2, style 2 features 0..2 (9 rows of 128 px)
    sheet = Image.new("RGBA", (CELL * 16, TILE_H * len(features) * len(STYLE_ORDER)), (0, 0, 0, 0))
    for si, sname in enumerate(STYLE_ORDER):
        st = STYLES[sname]
        for r, feature in enumerate(features):
            for mask in range(16):
                big = draw_tile(mask, st, args.ss)
                if feature != "plain" and not mask & S:
                    big = draw_feature(big, feature, args.ss)
                tile = big.resize((CELL, TILE_H), Image.LANCZOS)
                sheet.paste(tile, (mask * CELL, (si * len(features) + r) * TILE_H), tile)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sheet.save(args.out, optimize=True)
    print("ROOFS written=%s styles=%d features=%d variants=16 tile=%dx%d lift=%d" % (args.out, len(STYLE_ORDER), len(features), CELL, TILE_H, LIFT))


if __name__ == "__main__":
    main()
