# Performance: how to measure

The method that took Sandblood (a gore-heavy physics fighter) from 726 to about 250-340 draw calls and its worst-case
effects script time from 67 ms to 34 ms at high quality, without guessing. Numbers here are examples; each project
keeps its own table at the bottom of this file.

Target: 60 fps on a mid PC and on mid Android phones (Compatibility / GLES3).

## The harness: `tools/perf.sh` + `tests/perf/perf.tscn`

Build one per project once there is a level and a bot. Shape that worked:

`tools/perf.sh [--quality=0|1|2|all] [--seed=S] [--measure=FRAMES] [--heavy] [--hide=part1,part2,...]`

1. **A presented, deterministic scene.** One bot-vs-bot match (or a bot run of a level) with full presentation (level,
   crowd, effects, HUD) in a real window: xvfb on Linux, a tiny resolution (320x180) so the software rasteriser does
   not dominate. Fixed seed so runs compare.
2. **Fast-forward to a state, not a time.** Run at a high time scale until an observed condition holds (for example
   `decals >= 60 and tick >= 600`, or the match ends), with a tick fallback. Then measure N frames (300) at 1x.
3. **A scripted worst case (`--heavy`).** When the measured window starts, inject the most expensive events at once
   (every gore effect on both fighters, every particle burst). Budgets are set against this, not the average.
4. **Toggle parts off (`--hide=`).** Switch one part off during the measured window (crowd, HUD, shadows, effects,
   characters, props); whole scene minus that run is that part's cost in draw calls and triangles.
5. **One row per quality level** (`--quality=all`), printed in a quiet table.

## What to report, and what not to trust

| column | meaning |
| --- | --- |
| frame ms p50 / p95 / max | real frames: `Time.get_ticks_usec()` between frames in a lowest-priority node, after a warm-up |
| script ms per system | each heavy system's own script time, timed apart from the renderer (the number to trust for CPU) |
| phys ms | physics time per frame |
| draw calls | per frame, including shadow passes and the HUD |
| MultiMesh triangles | instances shown x mesh triangles, counted by the harness itself |
| peaks | the counts that grow with play: decals, particles, pooled objects |
| VRAM, nodes | `Performance` monitors, once at the end |

- **Software GL frame time says nothing about a phone or a GPU.** On the cloud box (llvmpipe) engine process time is
  the renderer. Read script ms, draw calls and triangles there; profile frame time on real hardware (the owner's PC
  via Remote Control).
- **Never average `Performance.TIME_PROCESS`.** It updates about once a second with the slowest frame; averaging it
  reported 36 ms for a real 5.2 ms.
- **Godot's primitive counter counts a MultiMesh's mesh once per draw, not once per instance.** Count MultiMesh
  triangles yourself.
- **Warm up.** The first run after `--import` was 4x slower (shader compiles). Rerun a gap twice before hunting it.
  With `--fixed-fps` and vsync off, game time runs faster than wall time: report both.
- **Per-script cost without the editor profiler:** in the measured window call `set_process(false)` on scripted nodes
  and call each `_process` from one timing node, summing by script path.
- CPU ms are noisy under a shared box: read ratios, run heavy jobs one at a time on one machine.

## Budgets

Write budgets per quality level from the worst-case table, and keep low Android-safe. Example shape (Sandblood,
software GL, so treat CPU ms as 3-5x a desktop):

| budget | low | medium | high |
| --- | --- | --- | --- |
| worst-case effects script ms per frame | <= 8 | <= 15 | <= 35 (desktop) |
| draw calls, whole scene | <= 250 | <= 320 | <= 400 |
| MultiMesh triangles incl. crowd | <= 100k | <= 300k | <= 1.6M |
| VRAM | <= 32 MB | <= 48 MB | <= 64 MB |

CI can gate draw calls, triangles and script ms (they are stable under xvfb); GPU frame time stays on real hardware.

## Quality setting

One `GfxQuality.level` (0 low, 1 medium, 2 high) chosen before the scene is built (low on `OS.has_feature("mobile")`,
high on desktop), one Options picker for it, and a `--quality=N` run argument that overrides it. Every expensive
system reads it: crowd density and mesh LOD, shadow splits and distance, MSAA, effect caps, simulation rates and
texture resolution, mesh detail of instanced particles.

## Rules for new effects (each one came from a measured defect)

- Pool and cap everything per quality; never let a count grow with match length.
- A MultiMesh draws its first `visible_instance_count` instances: sort and cut instead of hiding in a shader. Do not
  hide geometry by scaling to zero in the vertex shader; the vertices are still processed.
- Gate repeated work on a changed input (a pool repainting unchanged texels cost 25 ms); hoist per-texel trig into a
  table; budget per-frame work by cost (area, bytes), not by count.
- Only big shapes cast shadows; thin decorative and small limb parts do not, at least on low.
- Many small `draw_*` calls in a HUD add up: cache segment lists and draw them with one `draw_multiline`; hide idle
  banners with `visible = false`.
- Visibility ranges (HLOD) work on `MultiMeshInstance3D`; use decimated meshes or impostors at distance (a MultiMesh
  cannot use engine mesh LOD).
- Pre-warm every effect, material and pool once off-screen during the intro: Godot 4.4+ ubershaders precompile only
  what is loaded, so first use hitches (one was 596 ms).
- Re-run `tools/perf.sh --quality=all` after adding an effect and put the new row in this file.

## Soak

Before a release, run 50+ back-to-back bot matches on real hardware and track the worst frame per match. It finds
per-match hitches (a slow match start, a first-use shader compile) that a single run never shows.

## This project's numbers
<!-- One table per pass: before/after per quality level, the scripted worst case, and what was changed. -->
