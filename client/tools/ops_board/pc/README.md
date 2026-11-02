# Ops board kit

A reusable project dashboard for game projects: frame rate and memory from soak tests, match-start stutter, the
heaviest scene on low vs high graphics, script cost, every 3D model against its triangle budget, the GPU runner's jobs,
and a job list. First used for Sandblood (Gladiator), 2026-10-22. Live example: https://claude.ai/artifact/9bJDtKhvTkUAQo96zXp8Se

## What's here

- `template/index.html`: the board page. It has no project names in it; everything comes from the datasets below.
  Six tabs: Build (priorities, branches waiting to merge, assets not wired into the game, recent commits), Overview, Performance, Stutter, Assets, Pipeline.
- `make_ops_data.ps1`: builds the data files from local logs and model folders (soak log, hitch log, GitHub Actions
  runner logs, folders of .glb files). Run it with `Get-Help .\make_ops_data.ps1` to see the usage.
- `template/make_test_page.ps1`: wraps the page with a fake `dash` API and your data, so you can check it in a browser
  before publishing (`window.__errors` should be empty after clicking through every tab).
- `probes_godot/`: the Godot probe scripts that write the logs: `soak.gd` (matches back to back, real fps and memory),
  `hitch.gd` (frames over 25 ms and match build time), `build_split.gd` (what a match build spends its time on) and
  `prof.gd` (per-script cost by taking over each scripted node). They were written against Sandblood's `Match`,
  `MatchPresenter`, `Fighter` and `Flags` classes. In a new game, point them at its own match or level class.
  They ship as `*.gd.example` so Godot does not parse them (lint and `check_all` stay green in a new project):
  copy them to an untracked folder (Sandblood used `.prof/`), drop the `.example` suffix, and run them with
  `--scripted` so they never write player data.

## Starting a board for a new project

Ask Claude: "make an ops board for <project> from the ops board kit". It should:

1. Create a new Dashboard artifact (Artifact tool, `quickstart` with intent `other`, then publish with the Dashboard
   type and a title).
2. Run `make_ops_data.ps1` with the project's name, machine, GPU, engine and log paths.
3. Upload each JSON file as an asset of the new board, and write a `datasets/<id>` document for each one (ids below)
   plus `files/index.html` = `template/index.html` and `dash/meta` = the board title.
4. Check it with `make_test_page.ps1` before handing it over.

## Dataset ids and columns

The page reads these dataset ids. Missing ones show as empty cards, not errors.

| id | file | columns |
|---|---|---|
| `project` | project.json | name, machine, gpu, engine, soak_setup (one row) |
| `soak` | soak_fights.json | fight (match number), seed, game_seconds, fps, p95_frame_ms, worst_frame_ms, static_mem_mb, video_mem_mb, nodes, objects |
| `match_start` | match_start.json | fight, build_ms, first_frame_ms |
| `hitch_frames` | hitch_frames.json | fight, frame, ms, cause |
| `build_parts` | build_parts.json | part, ms, nodes, fix ("-" when none; others draw red) |
| `perf` | perf_heavy.json | tier (Low/High), measure, ms |
| `scripts` | script_cost.json | script, kind (frame/physics), ms |
| `models` | models.json | model, group, triangles, size_kb, budget |
| `runner_jobs` | runner_jobs.json | started (ISO UTC), result (Succeeded/Failed), failed_step, cause |
| `runner_perf` | runner_perf.json | run, tier, old_counter_ms, physics_ms, effects_ms, draw_calls, triangles, video_mem_mb |
| `jobs` | jobs.json | job, status (Done / In progress / Waiting / Blocked), note |
| `backlog` | backlog.json | priority (P0/P1/P2), item, owner, status, note (kept by hand or by Claude) |
| `builds` | builds.json | sha, date, subject, kind (Merge/Change): last 25 commits on main (`-RepoDir`) |
| `branches` | branches.json | branch, ahead, last_commit, subject: branches not merged yet (`-RepoDir`) |
| `asset_status` | asset_status.json | group, files_on_main, code_files_using, status: asset folders the game does or doesn't load (`-RepoDir`) |

`perf`, `scripts`, `build_parts` and `jobs` are kept by hand (or by Claude) because they come from one-off
investigations. The builder creates empty ones so the page always loads.

## Keeping it current

The datasets are saved files, so the board shows what was last uploaded. Refresh it after each soak, runner job or
model batch: rerun `make_ops_data.ps1`, upload the changed files, and point each dataset at its new file. For a
hands-off board, have the GPU workflow write its results (perf_*.json, soak and hitch logs) as workflow artifacts or
into the repo, and refresh from those.

## Measuring rules learned on Sandblood

- Time real frames (`Time.get_ticks_usec()` between frames). `Performance.TIME_PROCESS` reports the slowest frame of
  each second, not the average.
- Warm up once before measuring; the first run after an import pays shader compiles.
- Rerun a surprising number twice before hunting it.
- Track the worst frame per match, not only averages: match-start freezes hide in averages.
