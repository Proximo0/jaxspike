"""Build the placeholder art sheets from the CC0 Kenney packs.

Usage: python client/assets/build_sheets.py --src <dir holding the unzipped kenney packs>
Copies the used pack files into client/assets/kenney/<pack>/ (with License.txt), writes
client/assets/sheets/{terrain.png, monsters.png, job_<n>.png/.json} and client/assets/slots.json.
Standard library + Pillow only.
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
T = 16
RES = "res://assets/"

PACK_FILES = {
    "tiny-dungeon": ["Tilemap/tilemap_packed.png"],
    "tiny-town": ["Tilemap/tilemap_packed.png"],
    "roguelike-rpg-pack": ["Spritesheet/roguelikeSheet_transparent.png"],
    "ui-pack-rpg-expansion": [
        "PNG/panel_blue.png", "PNG/panel_brown.png", "PNG/panelInset_blue.png",
        "PNG/buttonLong_blue.png", "PNG/buttonLong_blue_pressed.png", "PNG/buttonLong_beige.png",
        "PNG/barBack_horizontalMid.png", "PNG/barRed_horizontalMid.png", "PNG/barBlue_horizontalBlue.png",
        "PNG/barYellow_horizontalMid.png", "PNG/cursorHand_grey.png", "PNG/iconCross_grey.png",
    ],
}

TERRAIN = ["grass", "road", "floor", "temple", "water", "tree", "wall", "rock", "warp", "waybell", "station"]
JOBS = ["novice", "swordsman", "mage", "archer", "acolyte", "thief", "merchant"]
MONSTERS = ["rat", "orc", "blobbin", "emberblob", "vexblob", "moonbun", "wrigglet", "cocoonling", "buzzeye", "bramblet", "fiddlehop",
            "capling", "pilferbug", "cragwing", "peepit", "gallopeck", "snarlfang", "sandgulper", "grukk_brute", "rotling",
            "clackbone", "imphorn"]
DIRS = ["down", "up", "left", "right"]

TD = "tiny-dungeon"
TT = "tiny-town"
RL = "roguelike-rpg-pack"
UI = "ui-pack-rpg-expansion"

# job -> tiny-dungeon (col, row) of the character
JOB_CELLS = [(1, 7), (0, 8), (0, 7), (4, 9), (4, 8), (3, 9), (2, 7)]

# monster -> (source tile col,row in tiny-dungeon, recolour rgb or None)
MONSTER_CELLS = {
    "rat": ((4, 10), None), "orc": ((1, 9), (0.55, 0.95, 0.45)), "blobbin": ((0, 9), (1.0, 0.55, 0.75)),
    "emberblob": ((0, 9), (1.0, 0.65, 0.25)), "vexblob": ((0, 9), (0.55, 0.95, 0.35)),
    "moonbun": ((3, 10), (1.0, 1.0, 1.0)), "wrigglet": ((2, 10), (0.6, 1.0, 0.5)),
    "cocoonling": ((0, 9), (0.9, 0.85, 0.5)), "buzzeye": ((0, 10), (0.5, 0.7, 1.0)),
    "bramblet": ((3, 9), (0.55, 1.0, 0.55)), "fiddlehop": ((3, 10), (0.6, 0.9, 0.45)),
    "capling": ((0, 9), (0.8, 0.5, 1.0)), "pilferbug": ((2, 10), (0.55, 0.45, 0.65)),
    "cragwing": ((0, 10), (0.85, 0.5, 0.45)), "peepit": ((4, 10), (1.0, 0.9, 0.4)),
    "gallopeck": ((3, 10), (1.0, 0.85, 0.35)), "snarlfang": ((4, 10), (0.55, 0.55, 0.65)),
    "sandgulper": ((3, 10), (0.6, 0.45, 0.3)), "grukk_brute": ((3, 7), (1.0, 0.5, 0.45)),
    "rotling": ((1, 10), (0.55, 0.9, 0.6)), "clackbone": ((1, 10), (1.0, 1.0, 0.9)),
    "imphorn": ((2, 9), (0.55, 0.3, 0.6)),
}

# item name -> (pack, col, row) straight regions from the 16 px packed sheets
ITEM_CELLS = {
    "gold_coin": (TT, 9, 7), "meat": (TT, 10, 8), "sword": (TD, 10, 8), "potion_red": (TD, 7, 9),
    "potion_blue": (TD, 8, 9), "dagger": (TD, 7, 8), "bow": (TT, 10, 9), "staff": (TD, 9, 10),
    "mace": (TD, 9, 9), "axe": (TD, 10, 9), "shield": (TD, 5, 8), "armor": (TD, 6, 8), "hat": (TD, 3, 7),
    "boots": (TT, 7, 10), "garment": (TD, 6, 5), "ring": (TD, 8, 4), "card": (TD, 1, 5), "ore": (TD, 0, 2),
    "herb": (TT, 5, 1), "fish": (TT, 9, 10),
}
ENGINE_ITEMS = {"1": "gold_coin", "2": "meat", "3": "sword"}

UI_FILES = {
    "panel": "panel_blue.png", "panel_title": "panel_brown.png", "button": "buttonLong_blue.png",
    "button_pressed": "buttonLong_blue_pressed.png", "button_hover": "buttonLong_beige.png",
    "slot": "panelInset_blue.png", "hpbar_bg": "barBack_horizontalMid.png",
    "hpbar_fill": "barRed_horizontalMid.png", "spbar_fill": "barBlue_horizontalBlue.png",
    "expbar_fill": "barYellow_horizontalMid.png", "cursor": "cursorHand_grey.png",
    "icon_close": "iconCross_grey.png",
}


def cell(img, col, row, spacing=0):
    x = col * (T + spacing)
    y = row * (T + spacing)
    return img.crop((x, y, x + T, y + T))


def over(base, top):
    out = base.copy()
    out.alpha_composite(top)
    return out


def recolour(tile, rgb):
    """Luminance times colour, keeps alpha and the dark outline."""
    out = tile.copy()
    px = out.load()
    for y in range(tile.height):
        for x in range(tile.width):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            lum = min(1.0, (0.3 * r + 0.59 * g + 0.11 * b) / 255.0 * 1.25)
            px[x, y] = (int(rgb[0] * lum * 255), int(rgb[1] * lum * 255), int(rgb[2] * lum * 255), a)
    return out


def darken(tile, k):
    out = tile.copy()
    px = out.load()
    for y in range(tile.height):
        for x in range(tile.width):
            r, g, b, a = px[x, y]
            px[x, y] = (int(r * k), int(g * k), int(b * k), a)
    return out


def shifted(tile, dx, dy):
    out = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    out.paste(tile, (dx, dy))
    return out


def build_terrain(td, tt, rl, out):
    grass = cell(tt, 0, 0)
    tiles = [
        grass,
        cell(tt, 1, 2),
        cell(td, 0, 0),
        cell(tt, 1, 9),
        cell(rl, 0, 0, 1),
        over(grass, cell(tt, 4, 1)),
        cell(td, 4, 3),
        cell(tt, 7, 3),
        over(cell(td, 0, 0), cell(td, 8, 4)),
        over(grass, cell(td, 3, 8)),
        over(grass, cell(tt, 11, 6)),
    ]
    sheet = Image.new("RGBA", (T * len(tiles), T))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * T, 0))
    sheet.save(out / "terrain.png")


def build_jobs(td, out):
    for j, (c, r) in enumerate(JOB_CELLS):
        base = cell(td, c, r)
        facing = {
            "down": base,
            "up": darken(base, 0.6),
            "left": base.transpose(Image.FLIP_LEFT_RIGHT),
            "right": base,
        }
        att = {"down": (0, 2), "up": (0, -2), "left": (-2, 0), "right": (2, 0)}
        dead = base.rotate(90, expand=True)
        sheet = Image.new("RGBA", (4 * T, 4 * T))
        for row, d in enumerate(DIRS):
            f = facing[d]
            frames = [f, shifted(f, 0, -1), shifted(f, *att[d]), dead]
            for col, fr in enumerate(frames):
                sheet.paste(fr, (col * T, row * T))
        sheet.save(out / f"job_{j}.png")
        anims = {}
        for row, d in enumerate(DIRS):
            anims[f"idle_{d}"] = {"frames": [[0, row]], "fps": 4, "loop": True}
            anims[f"walk_{d}"] = {"frames": [[0, row], [1, row]], "fps": 6, "loop": True}
            anims[f"attack_{d}"] = {"frames": [[0, row], [2, row]], "fps": 8, "loop": False}
        anims["dead_down"] = {"frames": [[3, 0]], "fps": 1, "loop": False}
        data = {"image": f"{RES}sheets/job_{j}.png", "frame_w": T, "frame_h": T, "margin": 0, "spacing": 0,
                "animations": anims}
        (out / f"job_{j}.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def build_monsters(td, out):
    sheet = Image.new("RGBA", (T * len(MONSTERS), T))
    for i, name in enumerate(MONSTERS):
        (c, r), rgb = MONSTER_CELLS[name]
        t = cell(td, c, r)
        sheet.paste(recolour(t, rgb) if rgb else t, (i * T, 0))
    sheet.save(out / "monsters.png")


def build_slots():
    slots = {}
    for i in range(len(TERRAIN)):
        slots[f"tile.{i}"] = {"path": f"{RES}sheets/terrain.png", "region": [i * T, 0, T, T]}
    for j in range(len(JOBS)):
        for a in ("idle", "walk", "attack", "dead"):
            slots[f"job.{j}.{a}"] = {"sheet": f"{RES}sheets/job_{j}.json", "anim": a}
    for i, name in enumerate(MONSTERS):
        slots[f"monster.{name}"] = {"path": f"{RES}sheets/monsters.png", "region": [i * T, 0, T, T]}
    for name, (pack, c, r) in ITEM_CELLS.items():
        slots[f"item.{name}"] = {"path": f"{RES}kenney/{pack}/tilemap_packed.png", "region": [c * T, r * T, T, T]}
    for k, name in ENGINE_ITEMS.items():
        slots[f"item.{k}"] = dict(slots[f"item.{name}"])
    for k, f in UI_FILES.items():
        slots[f"ui.{k}"] = {"path": f"{RES}kenney/{UI}/{f}"}
    text = json.dumps({"version": 1, "slots": slots}, indent=1) + "\n"
    (HERE / "slots.json").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding the unzipped kenney packs")
    src = Path(ap.parse_args().src)
    for pack, files in PACK_FILES.items():
        dst = HERE / "kenney" / pack
        dst.mkdir(parents=True, exist_ok=True)
        lic = next(p for p in (src / pack).iterdir() if p.name.lower() == "license.txt")
        shutil.copyfile(lic, dst / "License.txt")
        for f in files:
            shutil.copyfile(src / pack / f, dst / Path(f).name)
    out = HERE / "sheets"
    out.mkdir(exist_ok=True)
    td = Image.open(HERE / "kenney" / TD / "tilemap_packed.png").convert("RGBA")
    tt = Image.open(HERE / "kenney" / TT / "tilemap_packed.png").convert("RGBA")
    rl = Image.open(HERE / "kenney" / RL / "roguelikeSheet_transparent.png").convert("RGBA")
    build_terrain(td, tt, rl, out)
    build_jobs(td, out)
    build_monsters(td, out)
    build_slots()


if __name__ == "__main__":
    main()
