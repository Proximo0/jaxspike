"""Item icons in the chibi cel look: equipment rendered from the CC0 KayKit meshes through blender_render.py
(static job, one 3/4 view), consumables and materials drawn procedurally with Pillow. Writes
client/assets/icons/<name>.png (ICON px square, RGBA) and repoints the existing `item.<name>` slots in
client/assets/slots.json to them (values only, keys unchanged, file re-read right before the write).

    python client/tools/art/render_icons.py [--size 32] [--no-render] [--only NAME ...]
"""
import argparse
import json
import math
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_kind as rk  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(CLIENT, "assets", "icons")
SLOTS = os.path.join(CLIENT, "assets", "slots.json")
WORK = "D:/jaxspike-art/render/_icons"
SRC = "D:/jaxspike-art/src"
ADV = SRC + "/kaykit_adv/KayKit-Character-Pack-Adventures-1.0-main/addons/kaykit_character_pack_adventures/Characters/gltf/"
SKEL = SRC + "/kaykit_skel/KayKit-Character-Pack-Skeletons-1.0-main/addons/kaykit_character_pack_skeletons/Characters/gltf/"

# name -> (source glb, show regex list, yaw, elev, tint)
MESH_ICONS = {
    "sword": (ADV + "Knight.glb", ["^1H_Sword$"], 45, 12, None),
    "dagger": (ADV + "Rogue.glb", ["^Knife$"], 45, 12, None),
    "bow": (ADV + "Rogue.glb", ["^2H_Crossbow$"], 45, 30, [0.9, 0.75, 0.55]),
    "staff": (ADV + "Mage.glb", ["^2H_Staff$"], 45, 12, None),
    "mace": (SKEL + "Skeleton_Warrior.glb", ["^Skeleton_Warrior_"], 30, 20, None),  # replaced below if an axe-like mesh is better
    "axe": (ADV + "Barbarian.glb", ["^1H_Axe$"], 45, 12, None),
    "shield": (ADV + "Knight.glb", ["^Round_Shield$"], 20, 15, None),
    "hat": (ADV + "Mage.glb", ["^Mage_Hat$"], 30, 25, None),
    "garment": (ADV + "Knight.glb", ["^Knight_Cape$"], 30, 20, [1.0, 0.5, 0.5]),
    "armor": (ADV + "Knight.glb", ["^Knight_Body$"], 30, 20, None),
    "boots": (ADV + "Barbarian.glb", ["^Barbarian_LegLeft$", "^Barbarian_LegRight$"], 30, 25, [0.8, 0.6, 0.4]),
}
MESH_ICONS["mace"] = (ADV + "Mage.glb", ["^1H_Wand$"], 45, 12, [0.75, 0.75, 0.8])


def blender_icon(name, size, ss):
    src, show, yaw, elev, tint = MESH_ICONS[name]
    work = os.path.join(WORK, name)
    job = dict(rk.load_manifest()["defaults"])
    job.pop("shadow", None)
    job.update({"source": src, "out": work.replace("\\", "/"), "frame": [size, size], "show": show, "hide": ["^Icosphere"],
                "dirs": 1, "yaw": yaw, "elev_deg": elev, "zoom": 0.95, "margin_bottom": 4, "margin_top": 4,
                "floor": False, "base_lift": 0.0, "center": True, "outline_rel": 0.03, "specular": 0.3, "rim": 0.2,
                "supersample": ss, "actions": {"idle": {"frames": 1}}, "bone_scales": {}, "face": None, "hair": None,
                "radius_cap": 1.0})
    if tint:
        job["tint"] = tint
    rk.run_blender(rk.find_blender(""), job, work)
    big = Image.open(os.path.join(work, "idle_d0_f00.png")).convert("RGBA")
    # crop the drawn pixels on the supersampled render, fit them into the icon (pad 2 px at 1x), then shrink
    bbox = big.getbbox()
    full = size * ss
    pad = 2 * ss
    canvas = Image.new("RGBA", (full, full), (0, 0, 0, 0))
    if bbox:
        crop = big.crop(bbox)
        scale = min((full - 2 * pad) / crop.width, (full - 2 * pad) / crop.height)
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
        canvas.paste(crop, ((full - crop.width) // 2, (full - crop.height) // 2), crop)
    fr = rk.shrink(canvas, ss)
    return fr


# ------------------------------------------------------------------ procedural icons (cel: fill, shade, outline, gloss)
OUTLINE = (40, 30, 50, 255)


def cel(size, draw_fn):
    s = size * 4  # draw at 4x then shrink: smooth outline
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    draw_fn(d, s)
    return img.resize((size, size), Image.LANCZOS)


def potion(color):
    def fn(d, s):
        lw = s // 16
        # bottle
        d.rounded_rectangle((s * 0.28, s * 0.35, s * 0.72, s * 0.9), radius=s * 0.14, fill=color, outline=OUTLINE, width=lw)
        d.rectangle((s * 0.4, s * 0.15, s * 0.6, s * 0.4), fill=(230, 220, 200, 255), outline=OUTLINE, width=lw)
        d.rectangle((s * 0.37, s * 0.1, s * 0.63, s * 0.2), fill=(120, 80, 50, 255), outline=OUTLINE, width=lw)
        dark = tuple(max(0, c - 60) for c in color[:3]) + (255,)
        d.rounded_rectangle((s * 0.5, s * 0.45, s * 0.68, s * 0.86), radius=s * 0.1, fill=dark)
        d.ellipse((s * 0.34, s * 0.45, s * 0.42, s * 0.6), fill=(255, 255, 255, 200))
    return fn


def coin(d, s):
    lw = s // 16
    d.ellipse((s * 0.12, s * 0.2, s * 0.88, s * 0.85), fill=(200, 150, 40, 255), outline=OUTLINE, width=lw)
    d.ellipse((s * 0.12, s * 0.12, s * 0.88, s * 0.78), fill=(250, 210, 70, 255), outline=OUTLINE, width=lw)
    d.ellipse((s * 0.26, s * 0.24, s * 0.74, s * 0.66), outline=(200, 150, 40, 255), width=lw)
    d.ellipse((s * 0.3, s * 0.2, s * 0.42, s * 0.3), fill=(255, 255, 230, 220))


def meat(d, s):
    lw = s // 16
    d.ellipse((s * 0.2, s * 0.25, s * 0.85, s * 0.8), fill=(200, 90, 60, 255), outline=OUTLINE, width=lw)
    d.ellipse((s * 0.3, s * 0.3, s * 0.75, s * 0.65), fill=(230, 130, 90, 255))
    d.rounded_rectangle((s * 0.08, s * 0.55, s * 0.35, s * 0.75), radius=s * 0.08, fill=(240, 230, 210, 255), outline=OUTLINE, width=lw)
    d.ellipse((s * 0.4, s * 0.35, s * 0.5, s * 0.43), fill=(255, 220, 200, 200))


def herb(d, s):
    lw = s // 16
    d.line((s * 0.5, s * 0.9, s * 0.5, s * 0.3), fill=(60, 120, 50, 255), width=lw * 2)
    for k, (x, y, r) in enumerate([(0.32, 0.5, 0.16), (0.68, 0.45, 0.16), (0.5, 0.25, 0.17)]):
        d.ellipse((s * (x - r), s * (y - r * 0.7), s * (x + r), s * (y + r * 0.7)), fill=(110, 200, 90, 255), outline=OUTLINE, width=lw)
        d.line((s * (x - r * 0.6), s * y, s * (x + r * 0.6), s * y), fill=(60, 140, 60, 255), width=lw)


def ore(d, s):
    lw = s // 16
    pts = [(0.5, 0.12), (0.85, 0.4), (0.75, 0.85), (0.25, 0.85), (0.15, 0.4)]
    d.polygon([(s * x, s * y) for x, y in pts], fill=(120, 110, 140, 255), outline=OUTLINE, width=lw)
    d.polygon([(s * 0.5, s * 0.12), (s * 0.85, s * 0.4), (s * 0.5, s * 0.5)], fill=(170, 160, 190, 255))
    d.polygon([(s * 0.5, s * 0.5), (s * 0.75, s * 0.85), (s * 0.25, s * 0.85)], fill=(90, 80, 110, 255))
    d.ellipse((s * 0.55, s * 0.28, s * 0.65, s * 0.36), fill=(255, 255, 255, 220))


def fish(d, s):
    lw = s // 16
    d.ellipse((s * 0.12, s * 0.3, s * 0.72, s * 0.7), fill=(90, 160, 220, 255), outline=OUTLINE, width=lw)
    d.polygon([(s * 0.68, s * 0.5), (s * 0.92, s * 0.25), (s * 0.92, s * 0.75)], fill=(90, 160, 220, 255), outline=OUTLINE, width=lw)
    d.ellipse((s * 0.2, s * 0.42, s * 0.45, s * 0.62), fill=(140, 200, 240, 255))
    d.ellipse((s * 0.22, s * 0.4, s * 0.3, s * 0.48), fill=(30, 20, 40, 255))


def ring(d, s):
    lw = s // 16
    d.ellipse((s * 0.2, s * 0.3, s * 0.8, s * 0.9), outline=OUTLINE, width=lw * 3)
    d.ellipse((s * 0.2, s * 0.3, s * 0.8, s * 0.9), outline=(250, 210, 80, 255), width=lw)
    d.polygon([(s * 0.5, s * 0.1), (s * 0.66, s * 0.3), (s * 0.5, s * 0.45), (s * 0.34, s * 0.3)], fill=(120, 220, 240, 255), outline=OUTLINE, width=lw)
    d.ellipse((s * 0.44, s * 0.18, s * 0.5, s * 0.24), fill=(255, 255, 255, 230))


def card(d, s):
    lw = s // 16
    d.rounded_rectangle((s * 0.22, s * 0.1, s * 0.78, s * 0.9), radius=s * 0.06, fill=(80, 60, 140, 255), outline=OUTLINE, width=lw)
    d.rounded_rectangle((s * 0.3, s * 0.2, s * 0.7, s * 0.7), radius=s * 0.04, fill=(240, 220, 150, 255), outline=(50, 40, 90, 255), width=lw // 2)
    d.polygon([(s * 0.5, s * 0.28), (s * 0.62, s * 0.55), (s * 0.38, s * 0.55)], fill=(200, 80, 120, 255))
    d.rectangle((s * 0.33, s * 0.74, s * 0.67, s * 0.82), fill=(240, 220, 150, 255))


def etc(d, s):
    lw = s // 16
    # a small drawstring pouch: the generic "etc" loot
    d.ellipse((s * 0.18, s * 0.35, s * 0.82, s * 0.92), fill=(170, 120, 70, 255), outline=OUTLINE, width=lw)
    d.ellipse((s * 0.28, s * 0.42, s * 0.6, s * 0.72), fill=(200, 150, 90, 255))
    d.rounded_rectangle((s * 0.36, s * 0.2, s * 0.64, s * 0.42), radius=s * 0.06, fill=(150, 100, 60, 255), outline=OUTLINE, width=lw)
    d.line((s * 0.3, s * 0.33, s * 0.7, s * 0.33), fill=(240, 200, 90, 255), width=lw)


PROC_ICONS = {
    "etc": etc,
    "potion_red": potion((220, 60, 80, 255)), "potion_blue": potion((70, 110, 230, 255)), "gold_coin": coin,
    "meat": meat, "herb": herb, "ore": ore, "fish": fish, "ring": ring, "card": card,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--ss", type=int, default=4)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--only", action="append", default=[])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    done = {}
    names = args.only or (list(MESH_ICONS) + list(PROC_ICONS))
    for name in names:
        if name in MESH_ICONS and not args.no_render:
            img = blender_icon(name, args.size, args.ss)
        elif name in PROC_ICONS:
            img = cel(args.size, PROC_ICONS[name])
        else:
            continue
        p = os.path.join(OUT, name + ".png")
        img.save(p, optimize=True)
        done[name] = p
        print("icon=%s" % name)
    # repoint the item.* slots (values only); re-read right before writing
    with open(SLOTS, "r", encoding="utf-8") as fh:
        s = fh.read()
    d = json.loads(s)
    changed = 0
    for name in done:
        key = "item." + name
        d["slots"][key] = {"path": "res://assets/icons/%s.png" % name}  # existing keys repointed, new ones appended
        changed += 1
    for alias, name in (("item.1", "gold_coin"), ("item.2", "meat"), ("item.3", "sword")):
        if alias in d["slots"] and name in done:
            d["slots"][alias] = {"path": "res://assets/icons/%s.png" % name}
            changed += 1
    with open(SLOTS, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    # strip
    strip = Image.new("RGBA", (args.size * 2 * len(done), args.size * 2), (70, 130, 70, 255))
    for i, name in enumerate(done):
        im = Image.open(done[name]).convert("RGBA").resize((args.size * 2, args.size * 2), Image.NEAREST)
        strip.paste(im, (i * args.size * 2, 0), im)
    strip.save(os.path.join(CLIENT, "assets", "anim", "_review", "icons.png"))
    print("ICONS written=%d slots_changed=%d" % (len(done), changed))


if __name__ == "__main__":
    main()
