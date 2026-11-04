"""Compose the framed card art (round 4, ro4-cards): one 192x256 RGBA card per card item in client/assets/cards.

The portrait comes from the art pipeline (client/tools/art, owned by the art session), read-only:
  1. D:/jaxspike-art/portraits/<snake>.png (render_kind.py --portrait, a 3/4 bust close-up), when it exists;
  2. else frame 0 of the SW row of client/assets/anim/<snake>/idle.png (the chibi sprite, act.json layout).
Cards whose monster has no sprite in Aurelune (the seven card-only legacy cards) get a "mystery" silhouette.
The card is text-free (the client draws names): a gold frame (MVP: rose-gold with gems), a sunburst window tinted
by the monster's own colours, the portrait with a soft shadow, and a plate with the socket-type gem and rarity stars.

Usage (from client/): python tools/cards/compose_cards.py [--only blobbin,...] [--portraits DIR]
Pillow is the only dependency (like tools/art/check_sheets.py). Ends with `CARDS done cards=N portraits=P sprites=S`.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]  # client/
OUT = ROOT / "assets" / "cards"
ANIM = ROOT / "assets" / "anim"
PORTRAITS = Path("D:/jaxspike-art/portraits")
W, H = 192, 256
SS = 2  # supersample factor

# (snake, socket slot 0..6, mvp) for every card item, in item-id order (game/data/items.py card rows).
# slot: 0 weapon, 1 head, 2 armor, 3 shield, 4 garment, 5 shoes, 6 accessory
CARDS = [
    ("blobbin", 2, False), ("wrigglet", 0, False), ("cocoonling", 2, False), ("moonbun", 0, False),
    ("fiddlehop", 2, False), ("pilferbug", 2, False), ("grukk_brute", 3, False), ("antlet", 0, False),
    ("emberblob", 0, False), ("buzzeye", 5, False), ("peepit", 2, False), ("croakle", 2, False),
    ("capling", 6, False), ("needlebuzz", 0, False), ("cragwing", 4, False), ("bramblet", 1, False),
    ("vexblob", 6, False), ("rotling", 5, False), ("snarlfang", 0, False), ("clackbone", 0, False),
    ("gallopeck", 2, False), ("sandgulper", 4, False), ("clackbone_guard", 0, False), ("grumblebear", 3, False),
    ("tombwrap", 0, False), ("gildscarab", 3, True), ("scorchstripe", 5, True),
]
SLOT_GEM = [(235, 70, 70), (255, 150, 40), (70, 140, 255), (60, 190, 90), (170, 90, 230), (170, 110, 60),
            (40, 200, 220)]
INK = (52, 36, 30, 255)


def _title(snake: str) -> str:
    return " ".join(w.capitalize() for w in snake.split("_"))


def _sprite_frame(snake: str):
    """Frame 0 of the SW row (row 1) of idle.png, cropped to its alpha box; None without a sheet."""
    act = ANIM / snake / "act.json"
    if not act.is_file():
        return None
    meta = json.loads(act.read_text(encoding="utf-8"))
    idle = meta.get("actions", {}).get("idle", {})
    sheet_path = ANIM / snake / idle.get("file", "idle.png")
    if not sheet_path.is_file():
        return None
    fw, fh = int(meta["frame_w"]), int(meta["frame_h"])
    row = 1 if int(meta.get("rows", 1)) > 1 else 0
    with Image.open(sheet_path) as sheet_img:
        sheet = sheet_img.convert("RGBA")
    frame = sheet.crop((0, row * fh, fw, (row + 1) * fh))
    box = frame.getbbox()
    return frame.crop(box) if box else None


def _portrait(snake: str, portraits: Path):
    for name in (snake + ".png", _title(snake) + ".png"):
        p = portraits / name
        if p.is_file():
            with Image.open(p) as im:
                img = im.convert("RGBA")
            return img, "portrait"  # a bust framed by the art pipeline: keep its square, cut edges sit on the frame
    f = _sprite_frame(snake)
    return (f, "sprite") if f is not None else (None, "mystery")


def _palette(img):
    """Two bright tints from the portrait's saturated opaque pixels (sky and glow of the window)."""
    if img is None:
        return (150, 140, 200), (235, 230, 255)
    px = [p for p in img.convert("RGBA").resize((32, 32)).getdata() if p[3] > 200]
    sat = [p for p in px if max(p[:3]) - min(p[:3]) > 40] or px or [(200, 200, 200, 255)]
    r = sum(p[0] for p in sat) / len(sat)
    g = sum(p[1] for p in sat) / len(sat)
    b = sum(p[2] for p in sat) / len(sat)
    base = tuple(int(c + (255 - c) * 0.35) for c in (r, g, b))
    light = tuple(int(c + (255 - c) * 0.8) for c in (r, g, b))
    return base, light


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _star(d: ImageDraw.ImageDraw, cx, cy, r, fill):
    k = r * 0.3
    d.polygon([(cx, cy - r), (cx + k, cy - k), (cx + r, cy), (cx + k, cy + k), (cx, cy + r), (cx - k, cy + k),
               (cx - r, cy), (cx - k, cy - k)], fill=fill)


def compose(snake: str, slot: int, mvp: bool, portraits: Path):
    s = SS
    img, source = _portrait(snake, portraits)
    base, light = _palette(img)
    card = Image.new("RGBA", (W * s, H * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    # frame
    frame_hi, frame_lo = ((255, 150, 220), (210, 120, 60)) if mvp else ((255, 222, 110), (196, 136, 40))
    d.rounded_rectangle((0, 0, W * s - 1, H * s - 1), radius=16 * s, fill=INK)
    d.rounded_rectangle((3 * s, 3 * s, W * s - 3 * s, H * s - 3 * s), radius=14 * s, fill=frame_lo)
    for y in range(5 * s, H * s - 5 * s):  # vertical sheen on the frame
        t = abs((y / (H * s)) - 0.3)
        d.line((6 * s, y, W * s - 6 * s, y), fill=_lerp(frame_hi, frame_lo, min(t * 1.6, 1.0)))
    # window: gradient + sunburst rays + sparkles
    win = (12 * s, 12 * s, W * s - 12 * s, 202 * s)
    wimg = Image.new("RGBA", card.size, (0, 0, 0, 0))
    wd = ImageDraw.Draw(wimg)
    for y in range(win[1], win[3]):
        t = (y - win[1]) / (win[3] - win[1])
        wd.line((win[0], y, win[2], y), fill=_lerp(light, base, t) + (255,))
    cx, cy = W * s // 2, 112 * s
    for k in range(16):
        if k % 2:
            continue
        a0, a1 = k * math.pi / 8, (k + 1) * math.pi / 8
        rr = 300 * s
        wd.polygon([(cx, cy), (cx + rr * math.cos(a0), cy + rr * math.sin(a0)),
                    (cx + rr * math.cos(a1), cy + rr * math.sin(a1))], fill=(255, 255, 255, 46))
    for (sx, sy, sr) in ((34, 34, 7), (158, 46, 5), (150, 160, 4), (40, 150, 5), (96, 26, 4)):
        _star(wd, sx * s, sy * s, sr * s, (255, 255, 255, 220))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(win, radius=10 * s, fill=255)
    card.paste(wimg, (0, 0), Image.composite(wimg, Image.new("RGBA", card.size), mask).getchannel("A"))
    d.rounded_rectangle(win, radius=10 * s, outline=INK, width=2 * s)
    # portrait
    if img is not None:
        max_w, max_h = 160 * s, 172 * s
        sc = min(max_w / img.width, max_h / img.height)
        if source == "portrait":  # fill the window width, bottom on the window edge
            sc = (win[2] - win[0]) / img.width
        pw, ph = max(1, int(img.width * sc)), max(1, int(img.height * sc))
        resample = Image.NEAREST if sc >= 3 * s and source == "sprite" else Image.LANCZOS
        por = img.resize((pw, ph), resample)
        if source == "portrait":  # soften the bust's cut edges (top and sides) into the window's sky
            fade = Image.new("L", por.size, 255)
            fd = ImageDraw.Draw(fade)
            band = max(1, int(ph * 0.14))
            for y in range(band):
                fd.line((0, y, pw, y), fill=int(255 * (y / band) ** 1.5))
            side = max(1, int(pw * 0.05))
            for x in range(side):
                v = int(255 * x / side)
                fd.line((x, band, x, ph), fill=v)
                fd.line((pw - 1 - x, band, pw - 1 - x, ph), fill=v)
            por.putalpha(ImageChops.multiply(por.getchannel("A"), fade))
        px, py = cx - pw // 2, win[3] - (0 if source == "portrait" else 6 * s) - ph
        shadow = Image.new("RGBA", card.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).ellipse((cx - pw * 0.38, win[3] - 18 * s, cx + pw * 0.38, win[3] - 4 * s),
                                       fill=(40, 20, 40, 90))
        card.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(3 * s)))
        layer = Image.new("RGBA", card.size, (0, 0, 0, 0))
        layer.paste(por, (px, py))
        layer = Image.composite(layer, Image.new("RGBA", card.size), mask)  # clip to the window
        card.alpha_composite(layer)
    else:  # mystery card: a dark silhouette blob with a question mark
        d.ellipse((cx - 50 * s, 70 * s, cx + 50 * s, 190 * s), fill=(70, 60, 100, 200))
        d.arc((cx - 22 * s, 92 * s, cx + 22 * s, 136 * s), start=180, end=90, fill=(255, 245, 210), width=9 * s)
        d.line((cx, 136 * s, cx, 150 * s), fill=(255, 245, 210), width=9 * s)
        d.ellipse((cx - 6 * s, 160 * s, cx + 6 * s, 172 * s), fill=(255, 245, 210))
    # plate: socket gem + rarity stars
    plate = (18 * s, 210 * s, W * s - 18 * s, 244 * s)
    d.rounded_rectangle(plate, radius=8 * s, fill=(255, 246, 220), outline=INK, width=2 * s)
    gx, gy, gr = cx, 227 * s, 10 * s
    gem = SLOT_GEM[slot]
    d.polygon([(gx, gy - gr), (gx + gr, gy), (gx, gy + gr), (gx - gr, gy)], fill=gem, outline=INK)
    d.polygon([(gx, gy - gr + 3 * s), (gx + gr // 2, gy), (gx, gy - 2 * s)], fill=_lerp(gem, (255, 255, 255), 0.6))
    star_col = (255, 110, 200) if mvp else (255, 190, 40)
    for k in range(3 if mvp else 2):  # rarity: two stars a side, three on an MVP card
        for side in (-1, 1):
            _star(d, gx + side * (24 + 16 * k) * s, gy, 6 * s, star_col)
    if mvp:  # corner gems
        for (ex, ey) in ((10, 10), (W - 10, 10), (10, H - 10), (W - 10, H - 10)):
            d.ellipse((ex * s - 5 * s, ey * s - 5 * s, ex * s + 5 * s, ey * s + 5 * s), fill=(255, 90, 170), outline=INK)
    return card.resize((W, H), Image.LANCZOS), source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--portraits", default=str(PORTRAITS))
    args = ap.parse_args()
    only = {x for x in args.only.split(",") if x}
    OUT.mkdir(parents=True, exist_ok=True)
    review = OUT / "_review"
    review.mkdir(exist_ok=True)
    (review / ".gdignore").write_text("", encoding="utf-8")
    counts = {"portrait": 0, "sprite": 0, "mystery": 0}
    made = []
    for snake, slot, mvp in CARDS:
        if only and snake not in only:
            continue
        img, source = compose(snake, slot, mvp, Path(args.portraits))
        img.save(OUT / f"{snake}.png", optimize=True)
        counts[source] += 1
        made.append(img)
    if made:
        cols = 9
        rows = (len(made) + cols - 1) // cols
        sheet = Image.new("RGBA", (cols * (W + 8), rows * (H + 8)), (40, 44, 60, 255))
        for i, im in enumerate(made):
            sheet.alpha_composite(im, ((i % cols) * (W + 8) + 4, (i // cols) * (H + 8) + 4))
        sheet.save(review / "_all_cards.png")
    print(f"CARDS done cards={len(made)} portraits={counts['portrait']} sprites={counts['sprite']} "
          f"mystery={counts['mystery']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
