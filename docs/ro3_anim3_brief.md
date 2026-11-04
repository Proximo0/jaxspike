# ro3-anim3 brief (round 3, client animation): sit/pickup/dead clips, fade, flinch, sheets, anim fps

Team ro3-anim3 of `docs/round3_contract.md`. Contract in force: `docs/round2_contract.md` 1.2-1.4, `docs/anim_wire.md`.
Shared checkout with ten other teams: **no git**, edit only the files your row lists, re-read a shared file right
before every edit and keep edits anchored inside the named functions.

## Measured before (headless lab, 2026-11-02, box shared by 11 teams)
anim_walk_jump_p95 1.01 (<= 1.2), anim_walk_stall_pct 1.45 (< 5), anim_clock_drift_p95 0.19 (< 1),
anim_atk_sync_frames 0 (<= 1), **anim_fps_p95 22.95 (>= 55)**; `anim ms/frame: net 0.91 layer 9.53 metrics 2.41
frame 15.26`. Windowed in round 2: fps 66.6.

## Wire facts (docs/anim_wire.md)
`a` 0 idle, 1 walk, 2 sit, 3 pickup, 4 standby, 5 attack, 6 hurt, 7 dead, 8 cast; `d` face 0..7; `t` start tick;
`m` dur_ms (0 = loop or held); `s` speed_ms; `h` damage this tick (ro3-wire adds `h = 0` for a miss). ro3-wire makes
dead entities visible for 3 s with `a = 7`, then `gone`; adds `sit` and `pickup` inputs so `a = 2/3` really arrive.
Sheets: `client/assets/anim/<kind>/act.json` + `<action>.png`, 5 rows (0 S, 1 SW, 2 W, 3 NW, 4 N; 5..7 mirror
3..1), players idle walk sit pickup standby attack hurt dead cast; monsters idle walk attack hurt dead.

## Workers and files (disjoint)
| worker | files | delivers |
|---|---|---|
| gen | `client/tools/gen_placeholder_anim.py`, `client/assets/anim/**` (rebuilt), `client/assets/slots.json` anim entries only, `client/assets/anim/index.json`, `client/assets/LICENSES.md` anim rows, `client/tests/unit/test_art_anim_sheets.gd` (expectation counts only) | hurt frames no longer washed out (no white tint; a saturated red/orange flash, outlines stay dark), MVP sheets upgraded (Gildscarab, Scorchstripe: bigger, bolder, MVP-looking), NPC sheets `npc_shop` and `npc_waybell` (idle 4 + talk 4 frames, humanoid chibi), contact sheet with a hurt row, MVP row and NPC row |
| play | `client/game/anim/*` (new `dead_fade.gd`), `client/game/world/entity_layer.gd` ONLY `_update` and `_animate`, `client/game/world/entity_track.gd` ONLY `clip_at`, `client/tests/unit/test_anim_*.gd` | sit/pickup/dead clips from `a`, dead fade after the dead clip, configure skipped when unchanged, pooled clip_at dict, hurt flinch with no position change |
| feed | `client/tests/lab/lab_anim_feed.gd`, `client/tests/unit/test_anim_feed.gd` (new) | walkers that get hit mid-walk (hurt + `h`), a few sitters, pickers and a death every few seconds in the feed so the lab exercises the new clips; the feed stays deterministic and keeps 200 entities |

## Rules
- Typed GDScript, tabs, pure RefCounted for logic (`game/anim/*`), no wall clock in logic, numbers as constants.
- Run from `client/` in Git Bash with
  `GODOT=/c/Users/crp27/Downloads/Godot_4.7.2/Godot_v4.7.2-stable_win64_console.exe PATH="$PWD/tools/winshim:$PATH"`.
  `tools/run_tests.sh test_anim_clip` runs one file; `tools/lint.sh` ends `LINT problems=0`. Never more than one Godot
  process per worker; never leave one running. Never wrap tools in `flock`.
- The lab: `timeout 600 $GODOT --headless --fixed-fps 60 --path . res://tests/lab/lab.tscn 2>&1 | grep '^LAB'`.
- Python tools: standard library only. The generator: `python client/tools/gen_placeholder_anim.py [--only kind]`.
- Report: files changed, checks run with their summary lines, numbers before/after, renders looked at, and lines
  for files you do not own.
