"""Check rendered action sheets against the round2 contract 1.4 format without Godot (standard library + Pillow).

    python client/tools/art/check_sheets.py [--dir client/assets/anim] [--kind K ...] [--min-px 96]

For every kind dir with an act.json: frame size, anchor inside the frame, atk_frame 3 of 6 (60 %), every action
png present with width = frame_w * frames and height = frame_h * 5, every drawn direction of frame 0 has opaque
pixels, consecutive walk/attack/hurt/dead frames differ (real motion), the feet (lowest opaque row of idle frame 0)
sit near the anchor, and the png palette carries alpha. Ends with `SHEETS kinds=N problems=M`.
"""
import argparse
import json
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "anim"))
PLAYER_ACTIONS = ["idle", "walk", "sit", "pickup", "standby", "attack", "hurt", "dead", "cast"]
MONSTER_ACTIONS = ["idle", "walk", "attack", "hurt", "dead"]
NPC_ACTIONS = ["idle", "talk"]
MOTION = ["walk", "attack", "hurt", "dead", "talk"]
MIN_OPAQUE = 80


def frame(img, fw, fh, row, col):
    return img.crop((col * fw, row * fh, (col + 1) * fw, (row + 1) * fh))


def opaque_count(fr):
    return sum(1 for a in fr.getchannel("A").tobytes() if a > 128)


def lowest_opaque_row(fr):
    a = fr.getchannel("A")
    w, h = a.size
    data = a.tobytes()
    for y in range(h - 1, -1, -1):
        row = data[y * w:(y + 1) * w]
        if max(row) > 128:
            return y
    return -1


def check_kind(kdir, kind, min_px, problems):
    act_path = os.path.join(kdir, "act.json")
    with open(act_path, "r", encoding="utf-8") as fh:
        act = json.load(fh)
    fw, fh_ = int(act["frame_w"]), int(act["frame_h"])
    anchor = act.get("anchor", [0, 0])
    is_player = "sit" in act["actions"]
    want = PLAYER_ACTIONS if is_player else (NPC_ACTIONS if "talk" in act["actions"] else MONSTER_ACTIONS)
    if fw < min_px or fh_ < min_px:
        problems.append("%s: frame %dx%d under %d px" % (kind, fw, fh_, min_px))
    if not (0 <= anchor[0] < fw and 0 <= anchor[1] < fh_):
        problems.append("%s: anchor %s outside the frame" % (kind, anchor))
    if "attack" in want and (int(act.get("atk_frame", -1)) != 3 or int(act["actions"].get("attack", {}).get("frames", 0)) != 6):
        problems.append("%s: atk_frame must be 3 of 6" % kind)
    rows = int(act.get("rows", 0))
    if rows == 5:
        if act.get("mirror_rows") != {"5": 3, "6": 2, "7": 1}:
            problems.append("%s: 5 rows need mirror_rows 5..7 -> 3..1" % kind)
    elif rows != 8:
        problems.append("%s: rows must be 5 (mirrored) or 8 (all drawn), got %d" % (kind, rows))
    for a in want:
        e = act["actions"].get(a)
        if e is None:
            problems.append("%s: action %s missing" % (kind, a))
            continue
        png = os.path.join(kdir, e["file"])
        if e.get("image") != "res://assets/anim/%s/%s" % (kind, e["file"]):
            problems.append("%s.%s: image path %s" % (kind, a, e.get("image")))
        if not os.path.isfile(png):
            problems.append("%s.%s: png missing" % (kind, a))
            continue
        img = Image.open(png)
        n = int(e["frames"])
        if img.size != (fw * n, fh_ * rows):
            problems.append("%s.%s: png %s, want %s" % (kind, a, img.size, (fw * n, fh_ * rows)))
            continue
        if img.mode == "P" and not img.info.get("transparency"):
            problems.append("%s.%s: palette png without alpha" % (kind, a))
        img = img.convert("RGBA")
        for d in range(rows):
            if opaque_count(frame(img, fw, fh_, d, 0)) < MIN_OPAQUE:
                problems.append("%s.%s: direction %d frame 0 near empty" % (kind, a, d))
        if a in MOTION:
            first = frame(img, fw, fh_, 0, 0).tobytes()
            if all(frame(img, fw, fh_, 0, c).tobytes() == first for c in range(1, n)):
                problems.append("%s.%s: all frames identical" % (kind, a))
        if a == "idle":
            feet = lowest_opaque_row(frame(img, fw, fh_, 0, 0))
            if feet >= 0 and abs(feet - anchor[1]) > fh_ * 0.2:
                problems.append("%s: idle feet row %d far from anchor y %d" % (kind, feet, anchor[1]))
    return sum(os.path.getsize(os.path.join(kdir, f)) for f in os.listdir(kdir) if f.endswith(".png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--kind", action="append", default=[])
    ap.add_argument("--min-px", type=int, default=96)
    args = ap.parse_args()
    kinds = args.kind or sorted(d for d in os.listdir(args.dir)
                                if os.path.isfile(os.path.join(args.dir, d, "act.json")) and not d.startswith("npc"))
    problems = []
    total = 0
    for k in kinds:
        try:
            total += check_kind(os.path.join(args.dir, k), k, args.min_px, problems)
        except Exception as exc:  # noqa: BLE001
            problems.append("%s: %s" % (k, exc))
    for p in problems:
        print("PROBLEM " + p)
    print("SHEETS kinds=%d problems=%d total_kb=%d" % (len(kinds), len(problems), total // 1024))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
