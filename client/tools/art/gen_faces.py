"""Procedural anime face decals for the sprite pipeline (Pillow). Writes RGBA PNGs the Blender renderer maps on a
plane shrink-wrapped to a model's head (or a blob's front): big eyes with iris, pupil and two highlights, brows,
a small mouth. Styles are parameterised by eye colour and expression so every kind can get its own look.

    python client/tools/art/gen_faces.py [--out D:/jaxspike-art/faces]

Produces <out>/<style>.png (512x512). Styles come from FACES below; the manifest's "face": {"style": ...} picks one.
"""
import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFilter

DEFAULT_OUT = "D:/jaxspike-art/faces"
S = 512

# style -> dict(iris=(r,g,b), mouth="smile"|"neutral"|"grin"|"angry"|"o", brows=bool, eye_w, eye_h, spacing)
FACES = {
    "novice": dict(skin=(246, 198, 152), iris=(60, 110, 200), mouth="smile", brows=True),
    "swordsman": dict(skin=(246, 198, 152), iris=(40, 90, 180), mouth="neutral", brows=True, brow_angle=12),
    "mage": dict(skin=(246, 198, 152), iris=(130, 70, 200), mouth="smile", brows=True),
    "archer": dict(skin=(246, 198, 152), iris=(60, 150, 90), mouth="neutral", brows=True, brow_angle=6),
    "acolyte": dict(skin=(246, 198, 152), iris=(80, 160, 220), mouth="smile", brows=True),
    "thief": dict(skin=(246, 198, 152), iris=(200, 70, 70), mouth="grin", brows=True, brow_angle=14),
    "merchant": dict(skin=(246, 198, 152), iris=(160, 100, 40), mouth="grin", brows=True),
    "blob_happy": dict(iris=(40, 30, 50), mouth="smile", brows=False, eye_w=0.17, eye_h=0.24, pupil=0.75),
    "blob_grumpy": dict(iris=(40, 30, 50), mouth="neutral", brows=True, brow_angle=18, eye_w=0.17, eye_h=0.22, pupil=0.75),
    "monster_cute": dict(iris=(40, 30, 50), mouth="smile", brows=False, eye_w=0.16, eye_h=0.22, pupil=0.7),
    "monster_mean": dict(iris=(200, 40, 40), mouth="angry", brows=True, brow_angle=22, eye_w=0.15, eye_h=0.17, pupil=0.6),
    "undead": dict(iris=(120, 220, 120), mouth="o", brows=False, eye_w=0.15, eye_h=0.18, pupil=0.8),
}


def draw_eye(d, cx, cy, w, h, iris, pupil_frac, hl_scale=1.0):
    # white
    d.ellipse((cx - w, cy - h, cx + w, cy + h), fill=(255, 255, 255, 255), outline=(30, 20, 40, 255), width=max(2, S // 90))
    # iris (slightly lower than centre, big)
    iw, ih = w * 0.78, h * 0.78
    icy = cy + h * 0.12
    d.ellipse((cx - iw, icy - ih, cx + iw, icy + ih), fill=iris)
    # darker ring + pupil
    dark = tuple(max(0, c - 70) for c in iris)
    d.ellipse((cx - iw, icy - ih, cx + iw, icy + ih), outline=dark, width=max(2, S // 70))
    pw, ph = iw * pupil_frac * 0.55, ih * pupil_frac * 0.7
    d.ellipse((cx - pw, icy - ph + ih * 0.1, cx + pw, icy + ph + ih * 0.1), fill=(25, 15, 35, 255))
    # highlights
    hw = w * 0.32 * hl_scale
    d.ellipse((cx - iw * 0.55 - hw, icy - ih * 0.55 - hw, cx - iw * 0.55 + hw, icy - ih * 0.55 + hw), fill=(255, 255, 255, 255))
    hw2 = hw * 0.45
    d.ellipse((cx + iw * 0.35 - hw2, icy + ih * 0.35 - hw2, cx + iw * 0.35 + hw2, icy + ih * 0.35 + hw2), fill=(255, 255, 255, 230))
    # upper lid line (thicker, anime)
    d.arc((cx - w * 1.05, cy - h * 1.15, cx + w * 1.05, cy + h * 0.9), start=200, end=340, fill=(30, 20, 40, 255), width=max(3, S // 45))


def draw_brow(d, cx, cy, w, angle_deg, thickness):
    a = math.radians(angle_deg)
    dx, dy = math.cos(a) * w, math.sin(a) * w
    d.line((cx - dx, cy + dy, cx + dx, cy - dy), fill=(45, 30, 40, 255), width=thickness)


def draw_mouth(d, cx, cy, kind, w):
    col = (120, 40, 60, 255)
    lw = max(3, S // 60)
    if kind == "smile":
        d.arc((cx - w, cy - w * 0.9, cx + w, cy + w * 0.5), start=15, end=165, fill=col, width=lw)
    elif kind == "grin":
        d.chord((cx - w, cy - w * 0.6, cx + w, cy + w * 0.6), start=0, end=180, fill=(90, 30, 50, 255), outline=col, width=lw)
        d.line((cx - w * 0.9, cy, cx + w * 0.9, cy), fill=(255, 255, 255, 255), width=max(2, lw // 2))
    elif kind == "angry":
        d.chord((cx - w, cy - w * 0.5, cx + w, cy + w * 0.7), start=180, end=360, fill=(90, 30, 50, 255), outline=col, width=lw)
        for i in range(4):
            x = cx - w * 0.6 + i * w * 0.4
            d.polygon((x - w * 0.12, cy, x + w * 0.12, cy, x, cy + w * 0.3), fill=(255, 255, 255, 255))
    elif kind == "o":
        d.ellipse((cx - w * 0.45, cy - w * 0.5, cx + w * 0.45, cy + w * 0.5), fill=(60, 20, 40, 255), outline=col, width=lw)
    else:  # neutral: short line with a slight curve
        d.arc((cx - w * 0.8, cy - w * 0.3, cx + w * 0.8, cy + w * 0.5), start=20, end=160, fill=col, width=lw)


def render(style):
    p = dict(FACES[style])
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    skin = p.get("skin")
    if skin:
        # a soft skin-toned oval behind the features: covers the model's own painted eyes
        mask = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        md = ImageDraw.Draw(mask)
        md.ellipse((S * 0.12, S * 0.2, S * 0.88, S * 0.95), fill=tuple(skin) + (255,))
        mask = mask.filter(ImageFilter.GaussianBlur(S // 30))
        img = Image.alpha_composite(img, mask)
    d = ImageDraw.Draw(img)
    ew = S * p.get("eye_w", 0.15)
    eh = S * p.get("eye_h", 0.2)
    spacing = S * p.get("spacing", 0.21)
    ey = S * 0.47
    for sx in (-1, 1):
        draw_eye(d, S / 2 + sx * spacing, ey, ew, eh, tuple(p["iris"]) + (255,), p.get("pupil", 0.6))
    if p.get("brows", True):
        ang = p.get("brow_angle", 8)
        for sx in (-1, 1):
            draw_brow(d, S / 2 + sx * spacing, ey - eh * 1.55, ew * 0.9, -ang * sx, max(4, S // 40))
    draw_mouth(d, S / 2, S * 0.76, p.get("mouth", "smile"), S * 0.09)
    # faint blush
    blush = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(blush)
    for sx in (-1, 1):
        cx = S / 2 + sx * S * 0.3
        bd.ellipse((cx - S * 0.08, S * 0.6 - S * 0.045, cx + S * 0.08, S * 0.6 + S * 0.045), fill=(255, 120, 140, 110))
    blush = blush.filter(ImageFilter.GaussianBlur(S // 40))
    img = Image.alpha_composite(blush, img)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for style in FACES:
        render(style).save(os.path.join(args.out, style + ".png"))
    # a review strip
    strip = Image.new("RGBA", (S // 2 * len(FACES), S // 2), (90, 150, 90, 255))
    for i, style in enumerate(FACES):
        strip.paste(render(style).resize((S // 2, S // 2), Image.LANCZOS), (i * S // 2, 0), render(style).resize((S // 2, S // 2), Image.LANCZOS))
    strip.save(os.path.join(args.out, "_all_faces.png"))
    print("FACES written=%d out=%s" % (len(FACES), args.out))


if __name__ == "__main__":
    main()
