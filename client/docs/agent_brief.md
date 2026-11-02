# Agent brief

Read `CLAUDE.md` first, then `docs/lessons_learned.md`. This file lists the rules for helper agents and the
contracts other modules already provide. Use the contracts; do not change them.

## Hard rules for helper agents
- Only create or edit the files your task lists (you own those files). Never edit anyone else's files,
  `project.godot`, autoloads, `tests/runner.gd`, or shared bases. If you need a change there, describe it in your
  final report.
- Never call any `mcp__hearthbot__` tool and never message the user. Report only through your final answer to the lead.
- Git is lead-only. Do not run git at all (no clean, stash, checkout or reset). Never delete files you did not create. The lead stages files by name (never `git commit -a`) and commits.
- Only `godot --import` takes the shared lock, and the `tools/*.sh` scripts already take it. Never wrap
  `tools/*.sh` or any other Godot run in `flock` (nesting deadlocks).
- Run only your own checks: `tools/run_tests.sh <test_name>`, `tools/lint.sh`, and `tools/run_sims.sh` if you touched
  sim-covered code. The box has few CPUs and other workers run Godot too: keep runs short, never leave one running.
- A `-s` SceneTree script must `load()` game classes inside a deferred call (`call_deferred`). Otherwise autoloads
  like `Flags` are not ready yet and compilation fails.
- Rendering check: `xvfb-run -a -s "-screen 0 1920x1080x24" godot --path . --rendering-driver opengl3 --resolution 1280x720 <scene or -s script> -- --scripted`.
  Put throwaway render scripts in `/tmp/<your task>/`, not in the project, and look at the PNG you produce.
- Avoid pure `#00FF00` in debug colours (it is the colour of a classic GPU glitch).
- Kill only your own Godot PIDs (`kill <pid>`); never `pkill godot` while other workers run. At most 2 engine runs
  of yours at once.
- A failure in a file you did not touch is probably another worker's change: report it, do not "fix" it.
- Long bot or sweep runs: watch the log size and error lines, not just the exit code.
- Finish with a report in this format:
  1. Files written or changed (absolute paths).
  2. Checks run, each with its result line (`TESTS passed=..`, `LINT problems=..`, `SIMS passed=..`).
  3. What you saw in your render, if you rendered.
  4. Anything others must know, including lines the lead must add to files you do not own.

## Units and axes
<!-- Fill in: units, axes, physics rate, collision layers. Example (Sandblood): metres/kg/s, Y up, -Z forward,
     60 Hz Jolt, layers 1 arena, 2 bodies, 3 weapons. -->

## Contracts
<!-- One section per module other workers depend on: class, readable fields, signals, methods, data keys, taken
     input keys. When a build step has its own brief (`docs/<step>_brief.md`, see `docs/playbook.md`), point at it
     here instead of copying it. Workers code against the contract at the same time; they never change it. -->

## Debug layer already provided
- `Flags` (`game/autoload/flags.gd`): `Flags.args` (user args after `--`), `frame`, `commands`, `is_scripted_run()`,
  `dev_menu()`, `seed_value()`, `register_hash(callable)`, `state_hash()`.
- `FrameStamp` (F10 or `--stamp`): frame, state hash and a 112-bit code strip on screen; decode a screenshot with
  `python3 tools/frame_code.py decode <png>`.
- `RenderToggles` (F4 hide, F6 particles, F7 MSAA, F8 shadows, F9 3D text): for GPU glitch hunting.
- `Tuning.shared()` (`game/debug/tuning.gd`): live-tunable numbers. Read `Tuning.shared().<field>` each tick.
