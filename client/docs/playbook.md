# Playbook: kickoff, fan-out, measure-fix loop

How a lead Claude session builds a game with many cheap parallel subagents. It made Skate Park (a kid-friendly
skateboarding game) and Sandblood (an M-rated physics fighter), both Godot 4.7.2. Build guide:
https://claude.ai/artifact/WtmrkVRtizP8oJBRwhxdWW. The rules for workers are in `docs/agent_brief.md`; the
round-by-round production loop is in `docs/production_pipeline.md`; mistakes and their rules are in
`docs/lessons_learned.md`.

## 1. Kickoff (day one)

1. **Read the lessons.** `docs/lessons_learned.md`, then this file and `docs/new_project_checklist.md`. Follow them
   from minute one.
2. **Agree the vision and the feel targets with the owner**, in one round, recommending defaults: genre, engine
   (Godot 4 + GDScript, Compatibility renderer), rating and tone, v1 scope, art plan (placeholders behind fixed slots
   if the owner makes art), platforms, and 3 to 6 measurable feel targets with numbers. Write them to project memory
   the same turn and into `Tuning` defaults.
3. **Scaffold.** From the kit repo: `tools/new_project.sh <dest> "<Project Name>"`. Create a GitHub repo (public =
   free Actions minutes; secrets never go in git anyway) and push. Confirm that the `checks` workflow and
   `.claude/hooks/session-start.sh` both run.
4. **Build the debug layer before any content.** The kit gives `Flags`, the frame stamp, render toggles and `Tuning`.
   Add, with tests: a tuning panel for every gameplay number; a per-tick debug frame plus `snapshot()`/`restore()`;
   a flight recorder with replay; an overlay; a lab that measures the agreed feel targets. Add a play-bot as soon as
   a level exists.
5. **Spike the asset pipeline end to end** (import, rig, one pose, one render) and test-download one file from each
   asset source. For humanoids, read `docs/character_creation_lessons.md` first.
6. **Set up memory, the build guide and the dashboard.** Project memory: decisions, the owner's working preferences
   (agent count, models, how builds are delivered), lessons. The project's own build guide doc. The agent dashboard
   (`docs/agent_dashboard.md`).
7. **Plan in contracts** (section 2), then fan out.
8. **Ship early.** After each round, give the owner a playable build and record their feedback in memory.

## 2. Fan-out

### Before
1. **Write a contract brief** in `docs/<step>_brief.md`: the goal; the shared API (signatures, data dictionaries and
   their keys, file formats); input keys already taken; units and axes; who owns which file. Workers code against it
   at the same time and never change it.
2. **Land the core first.** If several features depend on one API (a per-tick debug frame, snapshot/restore, an input
   struct), one worker lands it first; everything built on it fans out afterwards.
3. **Assign file ownership.** Each worker gets an explicit file list and edits nothing else. Shared integration files
   (`main.gd`, `project.godot`, autoloads, `tuning.gd`) belong to the lead: workers report the lines to add. If two
   workers must touch one big file, give them different functions and tell both to make small anchored edits and
   re-read before each edit. Better: split the file.
4. **Pick the cheapest capable model:** Haiku for UI panels, simple tools, data files and docs; Sonnet for systems,
   features, test harnesses, sweeps and investigations (the default); Opus only for the hardest core logic
   (AI behaviour, locomotion, physics refinement).

### Standard preamble for every worker prompt
Point at `CLAUDE.md` and `docs/agent_brief.md` instead of retyping them, and add:
- Never call `mcp__hearthbot__` tools or message the user. Report only to the lead.
- Only edit the files listed. Do not run git at all.
- Never wrap tools in `flock`. Keep engine runs short, at most 2 at once, kill only your own PIDs, never leave one running.
- Renders: `-s` scripts `load()` game classes inside a deferred call. Look at every image you make and fix what is wrong.
- Final message: files changed, checks run with their summary lines, numbers before and after, lines for
  integration files, and a ranked "next 3 defects" list.

### Launching
- Launch background agents directly from the lead. Subagents do not get the Agent tool, so a "sub-orchestrator"
  cannot fan out; nested fan-out needs the Workflow tool, which caps concurrency at about CPUs-2 on small boxes.
- A 4-core cloud box caps Godot runs: keep local workers to about 8, and run more as remote cloud agents, each on its
  own machine and branch, merged by the lead after the gate. Scale checks on the GitHub Actions matrix.
- Keep the owner's progress checklist current, mapping workers to steps.

### After each hand-back
1. Read the report critically and look at the renders yourself (1-2 images, not all).
2. Copy the finished files into a clean clone, integrate the reported lines, run `tools/check_all.sh` there, and gate
   the commit on the summary lines (`failed=0`, `problems=0`), not on a subshell exit code.
3. Stage files by name (never `git commit -a`), commit with the attribution trailers, push, and confirm CI.
4. Forward new findings to any running worker they affect instead of starting a duplicate; stop a finished worker.
5. Refresh the dashboard. Add a lessons row the same day if something went wrong.
6. After a container restart, relaunch workers on their WIP with the previous findings in the prompt.

## 3. Measure, fix, verify loop

Use it when the owner reports a feel, physics, visual or performance problem, or asks to refine until it is right.

1. **Reproduce first.** Pick the tool that can see the problem:
   - a snag sweep for edges and corners (`tools/snag_sweep.sh` once the game has one);
   - the play-bot (`tests/bot/bot.tscn`, sharded) for real routes and matches;
   - the lab (`tests/lab/lab.tscn`) for feel numbers;
   - the flight recorder plus a replay tool for a run the owner recorded;
   - goldens and screenshot pixel sampling for visual glitches (the frame stamp, F10, ties a screenshot to a frame);
   - the perf harness on a scripted worst case for frame time (`docs/performance.md`).
   If nothing reproduces it, ship diagnostic toggles and ask the owner one precise question.
2. **Find the mechanism.** A tick-by-tick trace around the event: position, velocity, floor and wall normals,
   collider, state, events. Name the code line that causes it. Check git history for when it first appeared.
3. **Fix the mechanism, not the symptom.** Keep each fix minimal. Check a lab WARN against the owner's stated design
   wishes before tuning it away. Re-measure the baseline on the current tree before tuning: numbers go stale after any
   AI or physics change.
4. **Lock it in.** Add a test or sim that fails before the fix and passes after.
5. **Verify everything.** `tools/check_all.sh` (unit, sims, lint, plus bot and lab when they exist), compare against
   the saved baseline with the same tool, push, and let the Actions matrix confirm it.
6. **Loop.** Bot, lab and sweep findings become the next round's brief: crashes and errors, then stuck or no
   engagement, then broken behaviour, then balance outliers. Repeat until the bot runs clean, the sweep is near
   zero, the lab meets its targets and the sims are green.
7. **Ship.** Export the build, give it to the owner with the before/after numbers in two or three lines, and record
   the result and any new lesson.

## 4. After the project: feed the kit

At the end of a project (or a big milestone), harvest its `docs/lessons_learned.md` into the starter kit repo:
generic rows go into the kit's `docs/lessons_learned.md` under the matching heading (dedupe against what is there);
stories that only make sense for that game go into the kit's `docs/lessons/<game>.md`. Tools and docs that proved
themselves and are not game-specific go into the kit too, with `Jaxspike` / `Jaxspike` where the
name appears, and the kit's own checks (`tools/run_tests.sh`, `tools/lint.sh`) must pass before committing.

## Pacing the weekly budget (production weeks)

- A steady week spends about 1/7 of the weekly limit per day (about 4-5% per five-hour window).
- Every session checks usage at the start. Ahead of pace: routine work on Sonnet or Haiku at low effort, no ultracode.
  Behind pace: fan out more cloud threads on the ops board's priority list.
- Model split: lead/coordinator on the strongest model at normal effort; cloud workers Sonnet by default, the strongest
  model only for core logic, net code and hard merges; helpers on the GPU PC Sonnet or Haiku.
- Scale with cloud threads and the free GitHub runners, not with more local workers (a 4-core box saturates at 3-4
  engine runs). Fresh sessions per task or day keep messages cheap; the lessons file and the ops board carry context.
