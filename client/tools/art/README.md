# client/tools/art: the sprite pipeline (Blender pre-render in the anime chibi cel look)

Owner: team ro3-art. Other teams (tiles, cards, effects) use the entry points below; they do not edit these files
(ask ro3-art for additions). One Blender process per machine at a time (the RAM guard enforces it).

## Entry point 1: a sprite kind (characters, monsters) = one manifest entry + one command

```bash
venv/Scripts/python.exe client/tools/art/render_kind.py --kind <name>     # or --jobs, --monsters, --all
```
Reads `kinds.json`, runs Blender headless with `blender_render.py`, composes `client/assets/anim/<folder>/
<action>.png` (rows = directions S SW W NW N, columns = frames) + `act.json` (contract 1.4 + `"scale": 0.5`),
writes a contact sheet to `D:/jaxspike-art/render/_contact/<kind>.png` and a labelled 8-direction review sheet to
`client/assets/anim/_review/<folder>.png`. Flags: `--resume` (keep frames on disk after a crash), `--partial`,
`--no-render` (compose again), `--act-only DIR` (dry run), `--out`, `--work`, `--ss` (supersample, default 2).
Check the result: `check_sheets.py` (`SHEETS kinds=N problems=0`) and the Godot test `test_art_anim_sheets`.

Manifest fields (`kinds.json` -> `kinds.<name>`): `role` job|monster, `folder` (output dir), `source` (glb/gltf
under `source_root`), `license` (key in `licenses.json`), `show`/`hide` (regexes over mesh names), `clips`
(action -> clip name, or `{"clip", "hit" 0..1, "span" [a,b], "yaw"}`), `frame` [w,h], `zoom`, `floor`, `hover`,
`tint` [r,g,b] (multiply), `lift` [r,g,b] (add), `brightness`, `saturation`, `bone_scales` {bone: factor},
`face` {style (gen_faces.py), bone, target regex, size, up, lift, offset, aspect}, `hair` {style spiky|messy|long|
bob, color, cover, spike, spikes, seed, bangs, long, target, bone}, `squash`, `outline_rel`, `specular`, `rim`.
Defaults for everything are in `defaults`.

## Entry point 2: a raw Blender job (props, tiles, portraits, anything with a model)

```bash
"C:/Program Files/Blender Foundation/Blender 5.0/blender.exe" -b --python client/tools/art/blender_render.py -- job.json
```
`job.json` is the dict `render_kind.build_job()` writes (see `D:/jaxspike-art/render/<kind>/_job.json` for a real
one). The minimum:
```json
{"source": "D:/path/model.glb", "out": "D:/out/dir", "frame": [128, 128], "elev_deg": 35, "zoom": 1.0,
 "margin_bottom": 12, "margin_top": 6, "hide": ["^Icosphere"], "outline_rel": 0.016,
 "ramp": [[0.0, 0.55], [0.42, 0.8], [0.66, 1.0]], "specular": 0.22, "rim": 0.18, "saturation": 1.2, "brightness": 1.15,
 "dirs": 1, "yaw": 0, "supersample": 2, "samples": 8,
 "actions": {"idle": {"frames": 1}}}
```
- No rig / no `clip` in an action -> a static render of the model (one frame per direction). `dirs` 1..8 turns
  the model 45 degrees per direction (0 faces the camera); `yaw` rotates every direction (a 3/4 portrait: dirs 1,
  yaw 35). Output: `<out>/<action>_d<dir>_f<frame>.png` at `frame * supersample` px, plus `_meta.json` with the
  feet anchor. Downscale with `render_kind.shrink()` (LANCZOS + hardened alpha) for the committed PNG.
- Tiles: `elev_deg` 35 matches the character camera; for a flat top-down tile set `elev_deg` 90 and a square
  frame; `floor: true` puts the model's lowest vertex at the ground. Everything renders with a transparent film.
- Cards: `frame` [256, 256], `dirs` 1, `yaw` 30, `zoom` 1.6, `margin_bottom` 24 gives a 3/4 bust-like portrait;
  the toon look, tints, face decals and hair come from the same job fields as a kind (copy them from the kind's
  `_job.json` to match the sheet exactly).
- Effects (cast circle, hit spark, level-up) are 2D: they do not go through Blender, see `gen_effects.py` when it
  exists.

## Faces and hair
`gen_faces.py` writes the 12 decal styles to `D:/jaxspike-art/faces/<style>.png` (edit `FACES`, run it, re-render).
Hair is procedural in `blender_render.make_hair`: an ellipsoid over the head's bounding box (`offset` grows it),
cut below `cover` (and above `front_cut` in front, with a zigzag fringe when `bangs`), then styled (`spiky`:
`spikes` tufts of `spike` length, `messy`: jittered, `long`: back and sides pulled down by `long`, `bob`: plain),
coloured by `color`, parented to `bone`. No asset needed. Sheets draw all 8 directions (`dirs` 8) so a weapon stays
in the right hand; `dirs` 5 gives the smaller mirrored sheets.

## Effects
`gen_effects.py` writes `client/assets/anim/fx/<name>.png` (one row of frames) and `fx.json`; slots `fx.<name>`.

## Licences
Every source is CC0 (`licenses.json`, `client/assets/LICENSES.md`). Nothing from RO or any other game, ever.
Sources live on `D:/jaxspike-art/src`, never in git; only the rendered PNGs and act.json are committed.
