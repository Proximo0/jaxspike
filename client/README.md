# Jaxspike

Godot 4.7.2 project started from the game starter kit: the tooling that proved itself on Skate Park and Sandblood,
there from minute one.

## What is in the box
- `project.godot`: Compatibility renderer, fixed 60 Hz physics, typed-GDScript warnings, ETC2/ASTC import for
  Android, `Flags` and `Controls` autoloads. `export_presets.cfg`: Windows Desktop and Android presets.
- `game/autoload/flags.gd`: user args after `--`, frame counter, state hash, `is_scripted_run()`.
- `game/debug/`: `FrameStamp` + `FrameCode` (F10 on-screen frame/hash code, decoded by `tools/frame_code.py`),
  `RenderToggles` (F4, F6-F9 for GPU glitch hunting), `Tuning` (template for live-tunable numbers).
- `tests/`: `runner.gd` (unit tests in `tests/unit/test_*.gd`, `extends TestCase`), `sim/` (sim suite + example).
- `tools/`: `run_tests.sh`, `lint.sh` (+ `lint.py`, `lint_parse.gd`), `run_sims.sh`, `check_all.sh`,
  `export_windows.sh`, `export_android.sh`, `frame_code.py`. Quiet by design: failures plus one summary line.
- `tools/agent_stats.py` + `tools/agent_dashboard/index.html`: the agent dashboard (agents, models, token pull per
  minute, cache hit rate, commits, CI). Set it up on day one: `docs/agent_dashboard.md`.
- `.github/`: `checks` workflow (unit, lint, sims matrix, commented bot shard block) and the `setup-godot` action.
- `.claude/`: settings (75-agent caps, SessionStart hook) and the hook that installs Godot and imports the project.
- `CLAUDE.md` and `docs/`: `playbook.md` (kickoff, fan-out, measure-fix loop), `agent_brief.md` (subagent rules and
  contracts), `lessons_learned.md`, `new_project_checklist.md`, `production_pipeline.md`, `performance.md`,
  `online.md`, `android.md`, `agent_dashboard.md`, `character_creation_lessons.md` (read before building any
  humanoid character).

## Daily use
    tools/run_tests.sh          # TESTS passed=N failed=0
    tools/lint.sh               # LINT problems=0
    tools/run_sims.sh           # SIMS passed=N failed=0 total=N
    tools/check_all.sh          # all of it in parallel, ends CHECK_ALL PASS (+ bot/lab if their scenes exist)
    tools/export_windows.sh     # build/<name>-windows.zip
    tools/export_android.sh     # build/<name>-android.apk (debug-signed)
    godot --path . -- --stamp   # run with the frame stamp visible

Tuning pattern: add a row to `Tuning.FIELDS` plus a `var`, read `Tuning.shared().<field>` in game code.
Add a sim: write `tests/sim/sim_<name>.gd` (extends `sim_base.gd`, `run()` returns `[[name, ok, detail]]`) and list it
in `SIM_FILES` in `sim_suite.gd`. Add a unit test: `tests/unit/test_<module>.gd`, methods named `test_*`.

Note: the first run of any tool imports the project (class cache); a raw `godot` render before that needs `godot --headless --path . --import` once.

Start with `docs/new_project_checklist.md`. When something goes wrong, add a row to `docs/lessons_learned.md`; at the
end of the project, copy the generic rows back into the starter kit (see the kit's README).
