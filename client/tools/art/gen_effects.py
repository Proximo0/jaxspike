"""Procedural 2D effect sprites in the chibi cel look (Pillow): one row of frames per effect, RGBA, premultiplied
look with soft glow, written to client/assets/anim/fx/<name>.png plus fx.json (frames, delay_ms, anchor, loop).

    python client/tools/art/gen_effects.py [--out client/assets/anim/fx] [--size 128]

Effects: cast_circle (looping magic circle with runes, rotates and pulses, 8 frames), bolt (6), arrow_shower (8),
magnum_break (6) per skill, hit_spark (one-shot star burst
with motion streaks, 6 frames), level_up (one-shot rising golden rays and sparkles, 10 frames), heal (one-shot
green-white cross sparkles, 8 frames). Anchor = the effect centre on the entity's feet (cast) or body (hit, heal).
"""
import argparse
import json
import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "anim", "fx"))


def glow(img, radius, strength=1.0):
    g = img.filter(ImageFilter.GaussianBlur(radius))
    r, gg, b, a = g.split()
    a = a.point(lambda v: min(255, int(v * strength)))
    g = Image.merge("RGBA", (r, gg, b, a))
    return Image.alpha_composite(g, img)


def cast_circle(size, n):
    frames = []
    for i in range(n):
        t = i / n
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx, cy = size / 2, size * 0.62
        rx, ry = size * 0.42, size * 0.2  # flat ellipse on the ground
        pulse = 1.0 + 0.06 * math.sin(t * 2 * math.pi)
        col = (120, 200, 255, 255)
        col2 = (200, 240, 255, 255)
        lw = max(2, size // 40)
        d.ellipse((cx - rx * pulse, cy - ry * pulse, cx + rx * pulse, cy + ry * pulse), outline=col, width=lw)
        d.ellipse((cx - rx * 0.72, cy - ry * 0.72, cx + rx * 0.72, cy + ry * 0.72), outline=col2, width=max(1, lw // 2))
        # rotating runes: 8 ticks on the outer ring
        for k in range(8):
            a = t * 2 * math.pi + k * math.pi / 4
            x, y = cx + rx * 0.86 * math.cos(a), cy + ry * 0.86 * math.sin(a)
            d.ellipse((x - lw, y - lw, x + lw, y + lw), fill=col2)
        # inner star (two triangles, rotating the other way)
        pts = []
        for k in range(6):
            a = -t * 2 * math.pi + k * math.pi / 3
            pts.append((cx + rx * 0.55 * math.cos(a), cy + ry * 0.55 * math.sin(a)))
        d.polygon([pts[0], pts[2], pts[4]], outline=col, width=max(1, lw // 2))
        d.polygon([pts[1], pts[3], pts[5]], outline=col, width=max(1, lw // 2))
        frames.append(glow(img, size // 24, 1.4))
    return frames, 90, True, [size // 2, int(size * 0.62)]


def hit_spark(size, n):
    frames = []
    rnd = random.Random(7)
    rays = [(rnd.uniform(0, 2 * math.pi), rnd.uniform(0.6, 1.0)) for _ in range(10)]
    for i in range(n):
        t = (i + 0.5) / n
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx, cy = size / 2, size / 2
        spread = size * 0.45 * (0.3 + 0.7 * t)
        fade = int(255 * (1.0 - t * 0.9))
        for a, l in rays:
            x0, y0 = cx + spread * 0.35 * l * math.cos(a), cy + spread * 0.35 * l * math.sin(a)
            x1, y1 = cx + spread * l * math.cos(a), cy + spread * l * math.sin(a)
            d.line((x0, y0, x1, y1), fill=(255, 240, 120, fade), width=max(2, int(size / 28 * (1 - t) + 1)))
        # central star
        r0 = size * 0.22 * (1.0 - t * 0.6)
        pts = []
        for k in range(8):
            a = k * math.pi / 4 + t
            rr = r0 if k % 2 == 0 else r0 * 0.4
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, fill=(255, 255, 230, fade), outline=(255, 160, 60, fade))
        frames.append(glow(img, size // 20, 1.2))
    return frames, 60, False, [size // 2, size // 2]


def level_up(size, n):
    frames = []
    rnd = random.Random(3)
    sparks = [(rnd.uniform(0.15, 0.85), rnd.uniform(0.0, 1.0), rnd.uniform(0.6, 1.0)) for _ in range(18)]
    for i in range(n):
        t = i / (n - 1)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx, feet = size / 2, size * 0.9
        fade = int(255 * (1.0 if t < 0.7 else (1.0 - t) / 0.3))
        # rising golden rays
        for k in range(7):
            x = cx + (k - 3) * size * 0.11
            top = feet - size * (0.3 + 0.6 * t) * (0.7 + 0.3 * math.sin(k))
            w = max(2, size // 32)
            d.line((x, feet, x, top), fill=(255, 220, 90, int(fade * 0.8)), width=w)
            d.line((x, top + size * 0.08, x, top), fill=(255, 255, 220, fade), width=w)
        # ring at the feet
        ry = size * 0.08
        rx = size * (0.25 + 0.2 * t)
        d.ellipse((cx - rx, feet - ry, cx + rx, feet + ry), outline=(255, 230, 120, fade), width=max(2, size // 40))
        # sparkles drifting up
        for sx, sy, s in sparks:
            y = feet - (sy + t * 1.2) % 1.0 * size * 0.9
            x = sx * size
            r = size * 0.025 * s
            d.polygon([(x, y - r * 2), (x + r, y), (x, y + r * 2), (x - r, y)], fill=(255, 255, 200, fade))
        frames.append(glow(img, size // 22, 1.3))
    return frames, 80, False, [size // 2, int(size * 0.9)]


def heal(size, n):
    frames = []
    rnd = random.Random(5)
    crosses = [(rnd.uniform(0.2, 0.8), rnd.uniform(0.0, 1.0), rnd.uniform(0.5, 1.0)) for _ in range(9)]
    for i in range(n):
        t = i / (n - 1)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        fade = int(255 * (1.0 if t < 0.6 else (1.0 - t) / 0.4))
        for sx, sy, s in crosses:
            y = size * 0.9 - (sy + t) % 1.0 * size * 0.8
            x = sx * size
            r = size * 0.05 * s
            w = max(2, int(r * 0.6))
            d.line((x - r, y, x + r, y), fill=(190, 255, 190, fade), width=w)
            d.line((x, y - r, x, y + r), fill=(190, 255, 190, fade), width=w)
        cx, cy = size / 2, size * 0.55
        rr = size * 0.3 * (0.6 + 0.4 * math.sin(t * math.pi))
        d.ellipse((cx - rr, cy - rr * 0.5, cx + rr, cy + rr * 0.5), outline=(140, 255, 160, int(fade * 0.7)), width=max(2, size // 36))
        frames.append(glow(img, size // 22, 1.3))
    return frames, 80, False, [size // 2, int(size * 0.55)]


def bolt(size, n):
    """A magic bolt (Cold/Fire/Lightning bolt family): a crackling column dropping onto the target, 6 frames."""
    frames = []
    rnd = random.Random(11)
    for i in range(n):
        t = i / (n - 1)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx, feet = size / 2, size * 0.86
        fade = int(255 * (1.0 if t < 0.5 else (1.0 - t) / 0.5))
        top = feet - size * 0.95 * min(1.0, t * 2.2)
        # jagged core
        pts = [(cx, feet)]
        y = feet
        while y > top:
            y -= size * 0.09
            pts.append((cx + rnd.uniform(-1, 1) * size * 0.08, y))
        d.line(pts, fill=(140, 200, 255, fade), width=max(3, size // 14))
        d.line(pts, fill=(240, 250, 255, fade), width=max(1, size // 32))
        # impact flash at the feet once the bolt lands
        if t > 0.35:
            r = size * 0.18 * (1.0 + (t - 0.35))
            d.ellipse((cx - r, feet - r * 0.45, cx + r, feet + r * 0.45), fill=(170, 220, 255, int(fade * 0.7)))
        frames.append(glow(img, size // 18, 1.5))
    return frames, 60, False, [size // 2, int(size * 0.86)]


def arrow_shower(size, n):
    """Arrows raining on an area: 8 frames, each arrow falls and sticks, with a dust ring on landing."""
    frames = []
    rnd = random.Random(21)
    arrows = [(rnd.uniform(0.15, 0.85), rnd.uniform(0.3, 0.85), rnd.uniform(0.0, 0.5)) for _ in range(12)]
    for i in range(n):
        t = i / (n - 1)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        fade = int(255 * (1.0 if t < 0.75 else (1.0 - t) / 0.25))
        for ax, ay, delay in arrows:
            p = min(1.0, max(0.0, (t - delay) / 0.4))
            x, y_land = ax * size, ay * size
            y = y_land - (1.0 - p) * size * 0.9
            ln = size * 0.16
            d.line((x + ln * 0.25, y - ln, x, y), fill=(120, 80, 40, fade), width=max(2, size // 40))
            d.polygon([(x, y), (x - ln * 0.12, y - ln * 0.28), (x + ln * 0.16, y - ln * 0.22)], fill=(200, 200, 210, fade))
            d.polygon([(x + ln * 0.25, y - ln), (x + ln * 0.05, y - ln * 0.85), (x + ln * 0.35, y - ln * 0.8)], fill=(230, 80, 80, fade))
            if p >= 1.0 and t - delay < 0.6:
                r = size * 0.05
                d.ellipse((x - r, y - r * 0.4, x + r, y + r * 0.4), outline=(220, 200, 160, int(fade * 0.6)), width=max(1, size // 64))
        frames.append(glow(img, size // 40, 1.1))
    return frames, 70, False, [size // 2, int(size * 0.6)]


def magnum_break(size, n):
    """A fiery burst around the caster: an expanding ring of flame with embers, 6 frames."""
    frames = []
    rnd = random.Random(31)
    embers = [(rnd.uniform(0, 2 * math.pi), rnd.uniform(0.5, 1.0)) for _ in range(14)]
    for i in range(n):
        t = (i + 0.5) / n
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx, cy = size / 2, size * 0.62
        fade = int(255 * (1.0 if t < 0.55 else (1.0 - t) / 0.45))
        rx = size * 0.46 * (0.25 + 0.75 * t)
        ry = rx * 0.45
        lw = max(3, int(size / 12 * (1.0 - t * 0.6)))
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=(255, 140, 40, fade), width=lw)
        d.ellipse((cx - rx * 0.9, cy - ry * 0.9, cx + rx * 0.9, cy + ry * 0.9), outline=(255, 230, 120, int(fade * 0.8)), width=max(1, lw // 2))
        # flame tongues on the ring
        for k in range(10):
            a = k * math.pi / 5 + t * 1.5
            x, y = cx + rx * math.cos(a), cy + ry * math.sin(a)
            h = size * 0.2 * (1.0 - t * 0.4)
            d.polygon([(x - h * 0.3, y), (x + h * 0.3, y), (x, y - h)], fill=(255, 180, 60, fade))
        for a, s_ in embers:
            r = rx * (0.6 + 0.6 * t) * s_
            x, y = cx + r * math.cos(a), cy + r * math.sin(a) * 0.45 - t * size * 0.25
            e = size * 0.02
            d.ellipse((x - e, y - e, x + e, y + e), fill=(255, 220, 120, fade))
        frames.append(glow(img, size // 16, 1.5))
    return frames, 60, False, [size // 2, int(size * 0.62)]


EFFECTS = {"cast_circle": (cast_circle, 8), "hit_spark": (hit_spark, 6), "level_up": (level_up, 10), "heal": (heal, 8),
           "bolt": (bolt, 6), "arrow_shower": (arrow_shower, 8), "magnum_break": (magnum_break, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = {"version": 1, "scale": 0.5, "effects": {}}
    total = 0
    for name, (fn, n) in EFFECTS.items():
        frames, delay, loop, anchor = fn(args.size, n)
        sheet = Image.new("RGBA", (args.size * n, args.size), (0, 0, 0, 0))
        for i, fr in enumerate(frames):
            sheet.paste(fr, (i * args.size, 0), fr)
        p = os.path.join(args.out, name + ".png")
        sheet.save(p, optimize=True)
        total += os.path.getsize(p)
        meta["effects"][name] = {"file": name + ".png", "image": "res://assets/anim/fx/%s.png" % name, "frame_w": args.size,
                                 "frame_h": args.size, "frames": n, "delay_ms": delay, "loop": loop, "anchor": anchor}
    with open(os.path.join(args.out, "fx.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("EFFECTS written=%d total_kb=%d out=%s" % (len(EFFECTS), total // 1024, args.out))


if __name__ == "__main__":
    main()
