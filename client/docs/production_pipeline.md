# Production pipeline (lean loop)

Proven on Sandblood (a physics fighter, 2026-10): AAA-leaning results at a much lower token draw than earlier
projects. Use this after the debug layer and bot exist (see `new_project_checklist.md`); the
kickoff, fan-out and measure-fix steps it builds on are in `playbook.md`.

## Roles
- **Lead (one session):** talks to the owner, writes short task prompts, owns git, verifies and pushes. Never does
  worker-sized coding itself and never reads worker transcripts, only their final reports.
- **Workers (background agents):** one task each, explicit file ownership, the hard rules in `agent_brief.md`.
  Model per task: Sonnet by default (FX, UI, tools, sweeps, data); Opus only for hard core logic (AI behaviour,
  locomotion, physics); Haiku for docs and simple data.

## One loop round
1. **Measure:** a bot sweep (every weapon/character/level vs a baseline, small n, both sides) collecting win rate,
   draws, fight length, self-hits, stuck time, log errors.
2. **Pick 1-2 worst defects:** crashes and errors > stuck or no engagement > broken behaviour > balance outliers.
3. **Reproduce, trace, fix the mechanism** in the smallest set of files; before/after numbers from the same tool.
4. **Verify on a clean clone** (lint, tests, sims), commit by name with the attribution trailers, push.
5. Ask the worker for a ranked "next 3 defects" list; that list seeds the next round.

## Keeping token draw low
- Prompts are short and point at files and docs instead of retyping them; workers read `CLAUDE.md`,
  `lessons_learned.md` and the relevant `docs/*.md` themselves.
- Disjoint file ownership lets 2-3 workers run in parallel with no merge work. Shared files (`tuning.gd`, the
  fighter/player script): small anchored edits, re-read before each edit, and tell the other worker.
- Workers keep tool output terse (tail logs, no file dumps) and return a short structured report.
- The lead checks renders by looking at 1-2 images, not all of them.
- Builds only at milestones the owner can feel, not every round. Updates to the owner only for results, blockers
  or questions; progress goes on the checklist.
- After a container restart, relaunch from the WIP in the tree with the previous findings in the prompt.

## Owner-facing extras
- Test bypasses ship as launchers next to the exe (for example "unlock all", "test shop"), never in release paths.
- Secrets are injected at export from a git-ignored `config/*credentials*` file (`docs/online.md`).
- Art made on the owner's machine (Houdini via Remote Control) comes back as a zip the owner drags into the thread.

## Scaling out
- **Time real frames.** Never average a once-per-second monitor such as `Performance.TIME_PROCESS`. Tick between
  frames, report p50/p95/max after a warm-up (the first run includes shader compiles), and profile on a real GPU
  (the owner's PC via Remote Control): the cloud box renders in software GL. Method: `docs/performance.md`.
- **More agents than cores.** A 4-core cloud box caps Godot runs. Run extra agents as remote cloud agents, each on
  its own machine and branch (`ramp/<topic>`); the lead merges after the gate. Keep local workers to about 8,
  because every worker runs the engine. Stop a finished worker once its work is integrated.
- **Soak before release.** Run 50+ back-to-back bot fights on real hardware. They find per-match hitches (a long
  match start, a first-use shader compile). Pre-warm materials and pools during the intro.
- **Dashboard for the owner.** A Dashboard artifact fed by uploaded JSON (workers, commits, checks, perf, balance),
  refreshed at each merge, keeps the owner informed without long replies. The agent dashboard
  (`docs/agent_dashboard.md`) is the day-one version.

## CI on the owner's PC (one self-hosted runner)
Proven on VoxelSpace (UE 5.8, 2026-10-27): one Windows laptop runner served four parallel workers in an afternoon.
- **Fast loop:** a dispatch input (`fast=true`) that builds and runs one test filter only. Iterate with it and run the
  full suite only before merging.
- **Optional gates decide themselves:** slow checks (plugin export into a blank project, client target) run only when
  the diff against main touches their sources; `true`/`false` still force them.
- **Queue discipline:** stack branches that depend on each other, run one capture for the stack, and cancel
  superseded queued runs. The runner serialises everything, so each saved run is 6-20 minutes back.
- **Unattended means no prompts:** git and other tools must fail fast (no credential prompts), and every step has a
  timeout, so one stuck step cannot hold the only runner.
- **Owner PC setup, once:** antivirus exclusions for the runner and build-cache folders (the biggest win), and the
  laptop's OEM performance mode while plugged in. Modern Standby laptops have no High performance plan, so
  `powercfg` cannot add one; use the vendor app (for example MSI Center) instead.
- **Merges:** with the owner's standing permission the lead merges green PRs itself and posts the link; a PR the
  platform refuses (no runner check, other thread's branch) goes to the owner as a clickable link at once.

## Houdini to Godot
1. Build scripts make their hip/export/preview folders first, take `--only=<ids>` and `--no-preview`, and share
   helper libs (`LIB`, `read_glb`).
2. Export glTF; force materials opaque on import (the exporter writes `alphaMode BLEND`) and assert `alphaMode` in
   the exporter test.
3. Named empties need object-mode export (objpath + objects, null OBJs); mirror with a negative-scale Transform
   plus a Reverse SOP.
4. Check armor and garment fit radially (distance to the part axis minus radius >= 0), not by bounding boxes.
5. PolyReduce keeps vertex-colour masks; render impostor cards with flat light so masks stay tintable.
6. Previews: modest emission or a dark background, so emissive parts do not blow out.
7. Run heavy jobs one at a time on the laptop; write JSON without a BOM; do not commit `.import` churn from a
   local Godot version bump.
