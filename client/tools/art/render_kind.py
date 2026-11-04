"""Render one or more sprite kinds from the manifest into client/assets/anim/<kind>/ (round2 contract 1.4).

    python client/tools/art/render_kind.py --kind swordsman [--kind blobbin ...] [--all] [--jobs] [--monsters]
        [--out DIR] [--work DIR] [--no-render] [--blender EXE] [--ss 2]

Pipeline per kind (manifest `client/tools/art/kinds.json`, sources under D:/jaxspike-art/src):
  1. write a job json and run Blender headless with `blender_render.py` (toon ramp, outline, ortho camera,
     5 directions S SW W NW N, N frames per action) into the work dir (D:/jaxspike-art/render/<kind>/);
  2. compose <action>.png (rows = directions, columns = frames) at 1x from the supersampled renders, trim the
     alpha fringe, and write act.json (anchor = feet pixel, atk_frame at 60 % of the attack clip);
  3. add a row to the contact sheet (<work>/_contact/<kind>.png).
Pillow is required (venv). Blender 4.2+ (tested on 5.0.1). Nothing here touches slots.json or git.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT = os.path.normpath(os.path.join(HERE, "..", ".."))
MANIFEST = os.path.join(HERE, "kinds.json")
DEFAULT_OUT = os.path.join(CLIENT, "assets", "anim")
DEFAULT_WORK = "D:/jaxspike-art/render"
BLENDER_CANDIDATES = [
    os.environ.get("BLENDER", ""),
    "C:/Program Files/Blender Foundation/Blender 5.0/blender.exe",
    "D:/tools/blender/blender.exe",
    "blender",
]

PLAYER_ACTIONS = ["idle", "walk", "sit", "pickup", "standby", "attack", "hurt", "dead", "cast"]
MONSTER_ACTIONS = ["idle", "walk", "attack", "hurt", "dead"]
NPC_ACTIONS = ["idle", "talk"]


def actions_for(role):
    return {"job": PLAYER_ACTIONS, "monster": MONSTER_ACTIONS, "npc": NPC_ACTIONS}[role]
# action -> (frames, delay_ms, loop) as the placeholder generator and the client's ClipTiming expect
CLIP = {
    "idle": (4, 200, True), "walk": (8, 75, True), "sit": (4, 300, True), "pickup": (4, 100, False),
    "standby": (4, 150, True), "attack": (6, 100, False), "hurt": (4, 75, False), "dead": (5, 120, False),
    "cast": (6, 100, True), "talk": (4, 160, True),
}
ATK_FRAME = 3  # of 6 -> 60 %
# The sheets are rendered at 2x the size the client draws at a 32 px tile (crisp at zoom 2.0); act.json carries
# "scale" so the loader draws frame_w*scale x frame_h*scale world pixels with the anchor scaled the same way.
DRAW_SCALE = 0.5
DIRECTIONS = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]


def find_blender(explicit):
    for c in ([explicit] if explicit else []) + BLENDER_CANDIDATES:
        if c and (os.path.isfile(c) or c == "blender"):
            return c
    raise SystemExit("blender.exe not found; pass --blender or set BLENDER")


def load_manifest():
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_job(kind, spec, man, work, ss):
    src_root = man.get("source_root", "D:/jaxspike-art/src")
    actions = {}
    role = spec.get("role", "job")
    is_player = role == "job"
    names = actions_for(role)
    defaults = man["defaults"]
    for a in names:
        n, _delay, loop = CLIP[a]
        clip = spec["clips"].get(a)
        if clip is None:
            raise SystemExit("%s: no clip for action %s" % (kind, a))
        if isinstance(clip, str):
            clip = {"clip": clip}
        entry = {"clip": clip["clip"], "frames": n, "loop": loop, "hit": clip.get("hit", 0.5 if a == "attack" else None),
                 "span": clip.get("span", [0.0, 1.0])}
        if "yaw" in clip:
            entry["yaw"] = clip["yaw"]
        actions[a] = entry
    job = dict(defaults)
    if not is_player and "zoom_monster" in defaults:
        job["zoom"] = defaults["zoom_monster"]
    job.update({k: v for k, v in spec.items() if k not in ("clips", "source", "role", "license", "note")})
    job["source"] = os.path.join(src_root, spec["source"]).replace("\\", "/")
    if job.get("face"):
        face = dict(job["face"])
        face["image"] = os.path.join(man.get("faces_dir", "D:/jaxspike-art/faces"), face.get("style", "novice") + ".png").replace("\\", "/")
        job["face"] = face
    job["out"] = work.replace("\\", "/")
    job["frame"] = spec.get("frame", defaults["frame_job"] if role != "monster" else defaults["frame_monster"])
    job["role"] = role
    job["actions"] = actions
    job["supersample"] = ss
    return job


def expected_frames(job):
    out = []
    for a, spec in job["actions"].items():
        for d in range(int(job.get("dirs", 5))):
            for i in range(int(spec["frames"])):
                out.append("%s_d%d_f%02d.png" % (a, d, i))
    return out


def run_blender(blender, job, work, fresh=True, retries=6):
    """Runs Blender until every frame exists; a crashed run (the NVIDIA GL driver dies now and then under a
    shared GPU) is resumed from the frames already on disk."""
    os.makedirs(work, exist_ok=True)
    if fresh:
        for f in os.listdir(work):
            if f.endswith(".png") or f == "_meta.json":
                os.remove(os.path.join(work, f))
    jp = os.path.join(work, "_job.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(job, fh, indent=1)
    log = os.path.join(work, "_blender.log")
    t0 = time.time()
    want = expected_frames(job)
    for attempt in range(retries):
        missing = [f for f in want if not os.path.isfile(os.path.join(work, f))]
        if not missing and os.path.isfile(os.path.join(work, "_meta.json")):
            break
        cmd = [blender, "-b"]
        if os.environ.get("JX_GPU_BACKEND"):
            cmd += ["--gpu-backend", os.environ["JX_GPU_BACKEND"]]
        cmd += ["--python", os.path.join(HERE, "blender_render.py"), "--", jp]
        with open(log, "a" if attempt else "w", encoding="utf-8") as lf:
            lf.write("=== attempt %d, %d frames missing\n" % (attempt, len(missing)))
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if rc != 0:
            time.sleep(3)
    missing = [f for f in want if not os.path.isfile(os.path.join(work, f))]
    if missing or not os.path.isfile(os.path.join(work, "_meta.json")):
        raise SystemExit("blender failed after %d attempts, %d frames missing, see %s" % (retries, len(missing), log))
    return time.time() - t0


def shadow_layer(fw, fh, anchor, spec):
    """A soft dark ellipse under the feet (RO draws one under every sprite): baked into each frame so the client
    needs nothing. spec: {"w": fraction of frame_w, "h": fraction of w, "alpha": 0..255, "blur": px}."""
    from PIL import ImageDraw, ImageFilter
    w = fw * float(spec.get("w", 0.5))
    h = w * float(spec.get("h", 0.36))
    cx, cy = anchor[0], anchor[1] - h * 0.15
    img = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), fill=(20, 25, 40, int(spec.get("alpha", 95))))
    return img.filter(ImageFilter.GaussianBlur(float(spec.get("blur", 1.5))))


def shrink(img, ss):
    if ss == 1:
        return img
    w, h = img.size
    out = img.resize((w // ss, h // ss), Image.LANCZOS)
    # harden the alpha edge a little so the outline reads crisp at zoom 2 (no soft halo)
    r, g, b, a = out.split()
    a = a.point(lambda v: 0 if v < 40 else (255 if v > 200 else int((v - 40) * 255 / 160)))
    out = Image.merge("RGBA", (r, g, b, a))
    return out


def make_act(kind, fw, fh, anchor, is_player, source, folder=None, rows=5):
    """The act.json dict (round2 contract 1.4) for a kind: 5 drawn rows with 5..7 mirrored, or all 8 rows drawn
    (no mirror: the weapon stays in the right hand), atk_frame 3 of 6. `folder` is the dir name under assets/anim."""
    folder = folder or kind
    mirror = rows < 8
    act = {
        "kind": folder, "scale": DRAW_SCALE, "frame_w": fw, "frame_h": fh, "anchor": [int(anchor[0]), int(anchor[1])],
        "atk_frame": ATK_FRAME, "rows": rows, "directions": DIRECTIONS,
        "mirror_rows": {"5": 3, "6": 2, "7": 1} if mirror else {},
        "actions": {}, "source": source,
    }
    names = actions_for(is_player) if isinstance(is_player, str) else (PLAYER_ACTIONS if is_player else MONSTER_ACTIONS)
    for a in names:
        n, delay, loop = CLIP[a]
        entry = {"file": a + ".png", "image": "res://assets/anim/%s/%s.png" % (folder, a), "row0": 0, "frames": n,
                 "delay_ms": delay, "mirror": mirror, "loop": loop}
        if a == "attack":
            entry["atk_frame"] = ATK_FRAME
        act["actions"][a] = entry
    return act


def default_anchor(job):
    fw, fh = job["frame"]
    return [fw // 2, fh - int(job.get("margin_bottom", 12))]


def compose(kind, job, work, out_dir, ss, is_player, partial=False):
    fw, fh = job["frame"]
    meta_path = os.path.join(work, "_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        anchor = [int(round(meta["anchor"][0])), int(round(meta["anchor"][1]))]
    elif partial:
        anchor = default_anchor(job)
    else:
        raise SystemExit("%s: no _meta.json in %s (render first)" % (kind, work))
    folder = job.get("folder", kind)
    kdir = os.path.join(out_dir, folder)
    os.makedirs(kdir, exist_ok=True)
    rows = int(job.get("dirs", 5))
    act = make_act(kind, fw, fh, anchor, job.get("role", "job" if is_player else "monster"),
                   "client/tools/art render_kind.py (%s)" % job["source"], folder, rows)
    total_bytes = 0
    missing = 0
    shadow = shadow_layer(fw, fh, anchor, job["shadow"]) if job.get("shadow") else None
    for a in act["actions"]:
        n = CLIP[a][0]
        sheet = Image.new("RGBA", (fw * n, fh * rows), (0, 0, 0, 0))
        for d in range(rows):
            for i in range(n):
                p = os.path.join(work, "%s_d%d_f%02d.png" % (a, d, i))
                if not os.path.isfile(p):
                    if partial:
                        missing += 1
                        continue
                    raise SystemExit("%s: missing frame %s" % (kind, p))
                fr = shrink(Image.open(p).convert("RGBA"), ss)
                if shadow is not None and a != "dead":
                    fr = Image.alpha_composite(shadow, fr)
                sheet.paste(fr, (i * fw, d * fh), fr)
        # palette-quantise to keep the PNGs small while staying crisp (alpha kept in the palette)
        q = sheet.quantize(colors=255, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        png = os.path.join(kdir, a + ".png")
        q.save(png, optimize=True)
        total_bytes += os.path.getsize(png)
    with open(os.path.join(kdir, "act.json"), "w", encoding="utf-8") as af:
        json.dump(act, af, indent=1)
    if missing:
        print("kind=%s partial: %d frames missing (blank)" % (kind, missing))
    return total_bytes


def review(kind, out_dir, is_player, folder=None):
    """User-facing review sheet: every action's first frame in all 8 directions (5..7 mirrored as the client
    does), drawn at 2x with labels, into client/assets/anim/_review/<folder>.png (.gdignore'd)."""
    from PIL import ImageDraw
    folder = folder or kind
    kdir = os.path.join(out_dir, folder)
    with open(os.path.join(kdir, "act.json"), "r", encoding="utf-8") as fh:
        act = json.load(fh)
    fw, fh_ = act["frame_w"], act["frame_h"]
    names = actions_for(is_player) if isinstance(is_player, str) else (PLAYER_ACTIONS if is_player else MONSTER_ACTIONS)
    z = 2
    label_w = 70 * z
    img = Image.new("RGBA", (label_w + fw * z * 8, (fh_ * z + 4) * len(names) + 24), (70, 130, 70, 255))
    d = ImageDraw.Draw(img)
    for c, dn in enumerate(DIRECTIONS):
        d.text((label_w + c * fw * z + 4, 4), dn, fill=(255, 255, 255, 255))
    for r, a in enumerate(names):
        sheet = Image.open(os.path.join(kdir, a + ".png")).convert("RGBA")
        y0 = 24 + r * (fh_ * z + 4)
        d.text((6, y0 + fh_ * z // 2), a, fill=(255, 255, 255, 255))
        rows = int(act.get("rows", 5))
        for c in range(8):
            row = c if c < rows else {5: 3, 6: 2, 7: 1}[c]
            fr = sheet.crop((0, row * fh_, fw, (row + 1) * fh_))
            if c >= rows:
                fr = fr.transpose(Image.FLIP_LEFT_RIGHT)
            fr = fr.resize((fw * z, fh_ * z), Image.NEAREST)
            img.paste(fr, (label_w + c * fw * z, y0), fr)
    rdir = os.path.join(out_dir, "_review")
    os.makedirs(rdir, exist_ok=True)
    p = os.path.join(rdir, folder + ".png")
    img.save(p)
    return p


def contact(kind, out_dir, work_root, is_player, folder=None):
    """One row per action: direction 0..4 of frame 0, then every frame of direction 0."""
    kdir = os.path.join(out_dir, folder or kind)
    with open(os.path.join(kdir, "act.json"), "r", encoding="utf-8") as fh:
        act = json.load(fh)
    fw, fh_ = act["frame_w"], act["frame_h"]
    names = actions_for(is_player) if isinstance(is_player, str) else (PLAYER_ACTIONS if is_player else MONSTER_ACTIONS)
    cols = 5 + 8
    img = Image.new("RGBA", (fw * cols, fh_ * len(names)), (70, 130, 70, 255))
    for r, a in enumerate(names):
        sheet = Image.open(os.path.join(kdir, a + ".png")).convert("RGBA")
        n = act["actions"][a]["frames"]
        for d in range(min(5, int(act.get("rows", 5)))):
            fr = sheet.crop((0, d * fh_, fw, (d + 1) * fh_))
            img.paste(fr, (d * fw, r * fh_), fr)
        for i in range(n):
            fr = sheet.crop((i * fw, 0, (i + 1) * fw, fh_))
            img.paste(fr, ((5 + i) * fw, r * fh_), fr)
    cdir = os.path.join(work_root, "_contact")
    os.makedirs(cdir, exist_ok=True)
    p = os.path.join(cdir, kind + ".png")
    img.save(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", action="store_true")
    ap.add_argument("--monsters", action="store_true")
    ap.add_argument("--npcs", action="store_true")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--blender", default="")
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--no-render", action="store_true", help="compose only, from an earlier render")
    ap.add_argument("--resume", action="store_true", help="keep frames already rendered in the work dir")
    ap.add_argument("--partial", action="store_true", help="compose even when frames are missing (blank cells)")
    ap.add_argument("--act-only", default="", metavar="DIR",
                    help="write only act.json per kind into DIR/<kind>/ (no Blender, no PNGs); a dry run")
    ap.add_argument("--portrait", action="store_true",
                    help="render a 3/4 bust close-up per kind (256x256 RGBA, same look) to --portrait-dir")
    ap.add_argument("--portrait-dir", default="D:/jaxspike-art/portraits")
    args = ap.parse_args()
    if args.portrait:
        blender = find_blender(args.blender)
        man = load_manifest()
        kinds = list(args.kind)
        for k, spec in man["kinds"].items():
            role = spec.get("role", "job")
            if args.all or (args.jobs and role == "job") or (args.monsters and role == "monster"):
                if k not in kinds:
                    kinds.append(k)
        os.makedirs(args.portrait_dir, exist_ok=True)
        for kind in kinds:
            spec = man["kinds"][kind]
            work = os.path.join(args.work, "_portrait", kind)
            job = build_job(kind, spec, man, work, args.ss)
            idle = spec["clips"]["idle"]
            job["actions"] = {"idle": {"clip": idle["clip"] if isinstance(idle, dict) else idle, "frames": 1, "loop": True,
                                       "hit": None, "span": [0.0, 1.0]}}
            job.update({"frame": [256, 256], "dirs": 1, "yaw": float(spec.get("portrait_yaw", 35.0)),
                        "zoom": float(spec.get("portrait_zoom", 1.1)), "margin_bottom": float(spec.get("portrait_bottom", -80.0)),
                        "margin_top": 10, "elev_deg": float(spec.get("portrait_elev", 20.0)), "squash": 0.0})
            secs = run_blender(blender, job, work)
            fr = shrink(Image.open(os.path.join(work, "idle_d0_f00.png")).convert("RGBA"), args.ss)
            out = os.path.join(args.portrait_dir, job.get("folder", kind) + ".png")
            fr.save(out, optimize=True)
            print("portrait=%s render_s=%.0f file=%s" % (kind, secs, out))
        print("RENDER_PORTRAITS done kinds=%d dir=%s" % (len(kinds), args.portrait_dir))
        return
    man = load_manifest()
    kinds = list(args.kind)
    for k, spec in man["kinds"].items():
        role = spec.get("role", "job")
        if args.all or (args.jobs and role == "job") or (args.monsters and role == "monster") or (args.npcs and role == "npc"):
            if k not in kinds:
                kinds.append(k)
    if not kinds:
        ap.error("give --kind, --jobs, --monsters or --all")
    if args.act_only:
        for kind in kinds:
            spec = man["kinds"][kind]
            is_player = spec.get("role", "job")
            job = build_job(kind, spec, man, os.path.join(args.work, kind), args.ss)
            kdir = os.path.join(args.act_only, job.get("folder", kind))
            os.makedirs(kdir, exist_ok=True)
            act = make_act(kind, job["frame"][0], job["frame"][1], default_anchor(job), is_player,
                           "client/tools/art render_kind.py (%s)" % job["source"], job.get("folder", kind),
                           int(job.get("dirs", 5)))
            with open(os.path.join(kdir, "act.json"), "w", encoding="utf-8") as fh:
                json.dump(act, fh, indent=1)
            print("kind=%s act=%s frame=%dx%d actions=%d" % (kind, os.path.join(kdir, "act.json"), job["frame"][0],
                                                               job["frame"][1], len(act["actions"])))
        print("RENDER_KINDS act-only kinds=%d" % len(kinds))
        return
    blender = None if args.no_render else find_blender(args.blender)
    grand = 0
    for kind in kinds:
        spec = man["kinds"][kind]
        is_player = spec.get("role", "job")
        work = os.path.join(args.work, kind)
        job = build_job(kind, spec, man, work, args.ss)
        secs = 0.0
        if not args.no_render:
            secs = run_blender(blender, job, work, fresh=not args.resume)
        size = compose(kind, job, work, args.out, args.ss, is_player, partial=args.partial)
        grand += size
        cp = contact(kind, args.out, args.work, is_player, job.get("folder", kind))
        rp = review(kind, args.out, is_player, job.get("folder", kind))
        print("kind=%s render_s=%.0f png_kb=%d contact=%s review=%s" % (kind, secs, size // 1024, cp, rp))
    print("RENDER_KINDS done kinds=%d total_kb=%d" % (len(kinds), grand // 1024))


if __name__ == "__main__":
    main()
