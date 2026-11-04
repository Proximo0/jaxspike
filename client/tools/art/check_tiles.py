"""Check client/assets/tiles/tiles.json against its atlases (standard library + Pillow): every prop cell lies inside
props.png and has size, origin and base_px; every rule names a cell and valid terrain ids; the roofs atlas has
16 columns x (styles x feature rows) tiles; the extra-props rows are where the generator says. Ends with
`TILES cells=N rules=M problems=P`.

    python client/tools/art/check_tiles.py
"""
import json
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "tiles"))
TERRAIN_IDS = 13


def main():
    with open(os.path.join(TILES, "tiles.json"), "r", encoding="utf-8") as fh:
        d = json.load(fh)
    problems = []
    cell = int(d.get("cell_px", 64))
    atlas = Image.open(os.path.join(TILES, os.path.basename(d["props_atlas"].replace("res://assets/tiles/", ""))))
    cols, rows = atlas.width // cell, atlas.height // cell
    if rows != int(d.get("props_rows", rows)) or cols != int(d.get("props_columns", cols)):
        problems.append("props.png is %dx%d cells, tiles.json says %sx%s" % (cols, rows, d.get("props_columns"), d.get("props_rows")))
    cells = d["props"]["cells"]
    for name, c in cells.items():
        at, sz = c.get("atlas", [0, 0]), c.get("size", [1, 1])
        if at[0] < 0 or at[1] < 0 or at[0] + sz[0] > cols or at[1] + sz[1] > rows:
            problems.append("cell %s at %s size %s is outside the %dx%d atlas" % (name, at, sz, cols, rows))
        if "base_px" not in c or "origin" not in c:
            problems.append("cell %s lacks base_px/origin" % name)
        region = atlas.crop((at[0] * cell, at[1] * cell, (at[0] + sz[0]) * cell, (at[1] + sz[1]) * cell)).convert("RGBA")
        if region.getbbox() is None:
            problems.append("cell %s is empty in the atlas" % name)
    for key, names in d["props"].get("on_terrain", {}).items():
        for n in names:
            if n not in cells:
                problems.append("on_terrain %s names unknown cell %s" % (key, n))
    for r in d.get("rules", []):
        if r.get("prop") not in cells:
            problems.append("rule names unknown cell %s" % r.get("prop"))
        for k in ("on", "near"):
            if not (0 <= int(r.get(k, -1)) < TERRAIN_IDS):
                problems.append("rule %s: %s out of range" % (r.get("prop"), k))
        if int(r.get("every", 1)) < 1:
            problems.append("rule %s: every < 1" % r.get("prop"))
    roofs = d.get("roofs")
    if roofs:
        ra = Image.open(os.path.join(TILES, os.path.basename(roofs["atlas"].replace("res://assets/tiles/", ""))))
        styles = max(1, len(roofs.get("styles", ["tile_red"])))
        feats = max(1, len(roofs.get("features", {}).get("rows", ["plain"])))
        want = (int(roofs.get("columns", 16)) * cell, styles * feats * cell * int(roofs.get("size", [1, 2])[1]))
        if ra.size != want:
            problems.append("roofs.png is %s, want %s for %d styles x %d features" % (ra.size, want, styles, feats))
    for p in problems:
        print("PROBLEM " + p)
    print("TILES cells=%d rules=%d problems=%d" % (len(cells), len(d.get("rules", [])), len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
