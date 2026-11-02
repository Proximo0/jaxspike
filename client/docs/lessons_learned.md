# Lessons learned (read before starting work)

Generic lessons from Skate Park (kid-friendly skateboarding) and Sandblood (M-rated physics fighter), both Godot 4.7.2
built by a lead Claude with many subagents. Build guide: https://claude.ai/artifact/WtmrkVRtizP8oJBRwhxdWW.
The full game-specific stories behind some rows are in the starter kit repo, `docs/lessons/skate_park.md` and
`docs/lessons/sandblood.md`.

Add a row under the matching heading whenever something goes wrong: what happened (one line, with numbers when you
have them) and the rule that prevents it. At the end of the project, copy the generic rows back into the starter kit
(`docs/playbook.md`, section 4).

## Workflow and agents

| What happened | Rule |
| --- | --- |
| Subagents messaged the user directly | Never call `mcp__hearthbot__` tools. Report only to the lead. |
| A global `flock` around every Godot run deadlocked agents | Only `godot --import` takes `/tmp/<project>_import.lock`. Never wrap `tools/*.sh` in flock. |
| Sub-orchestrator agents had no Agent tool | Nested fan-out needs the Workflow tool, or the lead starts every worker. |
| 4 cloud cores capped how many checks and agents could run Godot at once | Use `tools/check_all.sh` locally and the GitHub Actions matrix (`checks`) for scale. Keep local workers to about 8 (every worker runs the engine); run more as remote cloud agents, each on its own machine and branch, merged by the lead after the gate. |
| `git commit -a` pushed another worker's half-finished file | The lead stages files by name while workers are editing. |
| Untracked new files from a finished worker vanished from the shared tree while other workers were still running | Commit a finished worker's files promptly, or back them up first. Workers never run git at all (no clean, stash, checkout or reset) and never delete files they did not create. |
| Unit tests passed in the working tree but failed in CI, and mid-flight workers left the tree red while CI linted the pushed tree | Other workers' uncommitted edits hide failures. The lead copies finished files into a clean clone (`/tmp/<project>_build`) or a `git worktree` of HEAD, runs lint, tests and sims there, commits by name and pushes from it. |
| A worker typed a parameter as a contract class that was not committed yet; CI lint failed on the pushed tree | Code against an uncommitted contract class with duck typing (`Node3D`), and lint a clean clone before pushing. |
| The lead pushed a red tree: `(lint; tests; sims) && git commit` only checks the last command of the subshell | Gate commits on the summary lines (`failed=0`, `problems=0`), not on a subshell exit code. |
| A worker reported "sims and bot skipped as instructed" when the brief asked for them | The lead reruns `check_all` and the bot before every commit of physics-adjacent files. |
| Container restarts killed running workers three times | Workers leave WIP in the tree; relaunch on that WIP with the prior findings in the prompt, never from scratch. |
| A worker cleared its own hung Godot with `pkill godot` and killed another worker's render | Kill only your own PIDs (`kill <pid>`); never `pkill godot` while other workers run. |
| A worker reported a shared test failing in a file it did not touch; the cause was a concurrent worker (a tuning row made a filter match two groups; a data file length change broke a data test) | Before fixing a failure outside your files, check whether another worker's recent change caused it, and tell that worker. |
| A finished background worker sent duplicate hand-back reports | Stop a finished worker (TaskStop) once its work is integrated. |
| Two commit hashes were misquoted in reports | Report a commit hash only after the push has returned it. |
| The owner had to read long replies to see project state | Keep a Dashboard artifact fed by uploaded JSON (agents, commits, checks, perf, balance), refreshed at each merge (`docs/agent_dashboard.md`). |
| Remote Control on the owner's PC refused a guarded action (e.g. git push) approved only through the lead | Actions its permission check guards need the owner's own words in that session; ask them to type the approval there. |
| Two lead sessions pushed main at the same time after a handover (VoxelSpace) | One lead at a time. The new lead checks `git log origin/main` for foreign pushes before every merge, and the old session is closed only after the new lead's first green run. |
| A remote session on the owner's PC refused to push or to run the game in the CI runner's folder on the lead's say-so | Owner approval comes from the owner's own message. Ask once, name the whole step (push the shots branch, run captures there), and let that cover it. |
| The cloud session could not read files from the owner's PC while a remote session was running | Move outputs through git (a throwaway orphan branch such as `shots`), not through folder access. |
| More workers did not speed up engine work: one PC runs one engine process, and every build, capture and engine test queues there | Split each step into an engine-free core part (fan out wide, verified on cloud CI in seconds) and a narrow engine part (1-2 workers) batched into one PC run. Cap parallel workers at about 4 per lead, because the lead's review is the next queue. |
| The orchestrator ran on the largest model for days and used most of the owner's token budget | Run the lead on a mid-tier model and workers on a small one from day one; use the largest model only for short contract or design passes the owner asks for. |
| The remote-control safety layer refused lead-requested source edits and checkouts in the CI runner's checkout | Ship diagnostics as a branch and run it through the CI workflow; the remote session only runs capture and read-only scripts. |
| A commit was pushed without its attribution trailers, and force-push is denied, so it stays that way | Check the message before every push (`git log -1 --format=%B`, look for the trailer lines). |
| A self-hosted CI run sat 20 min in a git step with no output and held the only runner until the PC restarted | Unattended git (CI steps, runner services, agents) runs with `GIT_TERMINAL_PROMPT=0`, `GCM_INTERACTIVE=never` and `-c credential.helper=`, so a credential prompt fails at once instead of waiting forever. Every CI step that touches the network gets `timeout-minutes`. |
| A new CI option was "proven" by a run that never reached its new code: the fast-mode proof skipped the diff step, and the first real run hung in it | A proof run must go through the code it proves. Run each new branch of a workflow (here `auto`, `true`, `fast`) once before calling it done. |
| One self-hosted runner had four branches queued, each with its own 6-20 min run | Give the PC CI a fast mode (build + one test filter, no optional gates). Run slow optional gates (plugin export, client target) only when the diff touches their sources. Stack branches that depend on each other, and cancel a queued run that a stacked run already covers. |

| Unattended overnight agent work stalled for 45 min and nobody noticed: two loops each waited for the other's "go", `claude -p` killed its background subagents 10 min after its last turn, and the watcher (a background job in a chat-driven session) died between turns | Run unattended work as `claude -p` in a restart loop, one window per lane, prompt re-read from a file each run so the lead can steer by editing it. Lanes share scarce resources (engine, GPU) through a lock directory with a stale timeout and never wait on each other; set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 so workers finish; a standalone supervisor process restarts dead loops and kills runs silent for 40 min; the remote lead still checks liveness hourly (newest push < 45 min). |
| A loop told only to "work the queue" ran one agent for hours | Fan-out must be explicit in the prompt: worker count, model, one worktree plus one wip branch per worker, workers never push, the lane verifies and merges. |
| Agent build work on the CI machine pushed the CI perf gate 26-33% over budget | Run agent loops at Idle priority (children inherit it) so CI timing gets the CPU; one engine process at a time. |
| A merge broke the sellable-plugin export: a new source folder was in neither the ship nor the private list | Every new module folder is classified (ship or private) in the same commit, and the export check runs before every push; a shipped folder may never include a private one. |
| Branch work verified on g++ and clang hid 30+ MSVC errors; the first MSVC error stopped the build and hid the rest | Before a merge, build the branch on every compiler CI uses; one compiler is not a proxy for another. |
| The Windows core gate printed PASS on a build that did not compile (exit code 1 accepted, stale test binary found) | A build gate accepts exit 0 only and deletes or timestamps the test binary; prove a gate fix by injecting a compile error and seeing FAIL. |
| A worker's new test, written to match its own hypothesis, still passed with the fix switched off | Every new test is shown red without the fix, then green with it; the lead reruns the red build, not just the green one. |
| A chat-driven watcher on the owner's PC died between turns, so stalled lanes went unnoticed | Liveness is checked from outside: the cloud lead checks hourly that the newest PC push is under 45 min old, and restarts the lanes if not. |
| A blacked-out-planet bug took hours of guessing until env-var debug traces (why a body is resident, probes of big bodies) named the missed call in minutes | Build cheap env-var-gated traces for each subsystem's "why" (why resident, why culled, why rejected) before the second guess. |
## Dashboards

The full story is in `docs/lessons/ops_board.md` (VoxelSpace ops board: 20 commits and 4 debug runs before it saw every agent, then a second round about wrong values).

| What happened | Rule |
| --- | --- |
| Three debug runs passed while the board still missed most agents | The first layer of every dashboard debug run diffs the real sources (threads, sessions, workers, `gh run list`, `git log origin/main`, routines) against what the board shows. Renderer tests come second. |
| "Every session pushes its status" in CLAUDE.md: 2 of 5 sessions did, and the lead never pushed its ~75 workers | Pushes that depend on an agent remembering are missed. Automate them (hook, scheduled routine, or the script the lead already runs), and show which sources have not reported. |
| Researcher rows said "running" 32 h after the researcher was paused; a failed CI run showed as in progress | Every row carries a time and expires. "Running" with no news for 2 h shows as gone quiet. A snapshot is refreshed on a schedule, not only at merges. |
| The same fact in two shapes (`task`/`title`, `at`/`updated`, `session_`/`cse_` ids) gave blank columns, duplicate cards and false "stale" labels | One field name per fact, and one normaliser shared by every card, KPI and table. |
| Three polish rounds and a sale kit shipped before the board could see the lead's workers | Function first: polish only after the coverage check is green. |
| The page polled a connector tool that artifacts are blocked from, which cost a full round and a permission popup | Make one real call to any tool a page depends on, from the page's own context, before building on it. |
| A day after "fixed", the lead's card showed $171.91 and 31% context (real: $412.57, 60%) and "0 workers running" while 18 ran; the coverage check passed because every card existed | Check values, not presence: for each number on the board, fetch it from its primary source and compare. A card that exists can still be wrong. |
| Cost and context were typed by the session they describe, once, then re-sent unchanged with a fresh timestamp | Never let the observed party report its own measurements. Read them from the system that measures them (here `get_session`), and show when they were measured. |
| The board thread could call `get_session` all along; only the page could not. Three requests to the lead to push its workers were ignored | If a page cannot reach a source but an agent can, relay it on a schedule. Asking another agent twice for the same push means automate it now. |
| The lead's own status said "18 workers" while the board had 0 worker docs | Cross-check counts across two sources and show the disagreement on the page instead of trusting either. |

## Testing and measurement

| What happened | Rule |
| --- | --- |
| The lab's per-snap apply budget (2 ms) failed as soon as nine agent sessions and four workers shared the PC: 1.4 ms idle became 2-5 ms at HEAD and in the working tree alike, so a real change was blamed for load noise | Before blaming a change for a lab FAIL, run the same lab on a clean worktree of HEAD under the same load; set CPU-time budgets with headroom for a loaded box (here 5 ms against a 120 ms tick) and keep the tight number as a WARN. |
| Feel bugs were found by the player, not by us | Build the debug layer and the bot before content: tunables, debug frame, snapshot/replay, recorder, overlay, lab, bot. |
| A fix looked plausible but the bug predated the changed code | Check git history for when a bug first appeared. |
| A lab WARN almost undid a design choice the player wanted | Check a WARN against the player's stated wishes before tuning it away. |
| Evaluators reported bugs that were capture artefacts (a bot teleport wiped open state; a UI element caught mid-fade) | Before fixing an evaluator finding, reproduce it in the live scene; bots wait for open state to settle before teleporting. |
| Balance work started from stale numbers after an AI change | Re-measure the baseline on the current tree before tuning; numbers go stale after any AI or physics change. |
| A bot run hung 15 min and wrote 334 MB: another worker's half-saved shared script failed to parse, so every tick errored | Watch a long bot run's log size or error lines, not just its exit; re-run balance after a shared script stops changing. |
| A rate-per-second sim check failed on a 4 s bot fight: the opening clash and a finisher burst inflated the mean | Divide rates by a floor duration (e.g. 8 s) so short runs do not inflate them; bursts belong to a max/s check. |
| Mid-run restore of a physics match diverged (0.2 m after 1 s) though idle restore and fresh starts were exact | Record runs as seed + inputs from tick 0 and replay from the start; treat mid-run restore as approximate. |
| A bot tool reported a false "crash" on big outputs | Under `set -o pipefail`, `echo "$out" \| grep -q` fails when grep exits early (SIGPIPE). Use a here-string: `grep -q pat <<<"$out"`. |
| A perf tool averaged `Performance.TIME_PROCESS` per frame and reported 36 ms vs a real 5.2 ms: that monitor updates about once a second with the slowest frame, and the first run included shader compiles | Time real frames (ticks between frames, p50/p95/max) after a warm-up; never average a once-per-second monitor. Profile on a real GPU: the cloud box renders in software GL (`docs/performance.md`). |
| Per-frame timing was noisy and misleading: a first run after `--import` was 4x slower, a gap shrank on rerun, and with `--fixed-fps` and vsync off game time ran faster than wall time | Warm up before measuring and rerun a gap twice before hunting it. Report game time and wall time. Time frames with `Time.get_ticks_usec()` in a lowest-priority node (mean, p95, max) and track the worst frame per match. |
| Per-script cost was unknown without the editor profiler | In the measured window call `set_process(false)` on scripted nodes and call each `_process` from one timing node, summing by script path. |
| A texture pool re-stamped 30x a second at an unchanged radius (25 ms in the worst case), and a per-texel loop called three sines | Time sections in a scripted worst case (`perf.sh --heavy`) before guessing; gate repeated work on a changed input, hoist per-texel trig into a table. |
| A soak of 50+ back-to-back bot matches on real hardware found per-match hitches (63 ms at each match start, one 596 ms first-use shader compile) | Run a long soak on real hardware before release; pre-warm materials, shaders and pools during the intro. |
| Heavy GPU/CPU jobs run in parallel on one laptop distorted each other's results | Run heavy jobs one at a time on one machine. |
| Net sims and tests opened fixed ports, so two suites on one machine failed with "Could not listen on port", and two matchmaking runs found each other's peers | Sims take ports from `net_port(slot)` in `tests/sim/sim_base.gd` (a 64-port block per OS process id plus a slot). Never write a literal port in a test. |
| `check_all.sh` logged to a shared `/tmp/check`, so two runs overwrote each other's logs, and it ended on a table with no single verdict | Each run logs to its own `mktemp` dir; the last line is `CHECK_ALL PASS` or `CHECK_ALL FAIL`, and the exit code matches. |
| Cloud sessions could not download CI logs or artifacts (blob host blocked), so results were invisible | Emit the key result lines as check-run annotations (`::notice`, `::error`) and read them with `gh api .../check-runs/<id>/annotations`. Keep a digest script that works from annotations alone. |
| Screenshots were black three times for three different reasons: no GPU on the runner service, a failed material, wrong exposure | Grade every capture automatically (mean, black %, verdict) and fail on BLACK. Split the cause in one round with console overrides: exposure off, unlit view mode. |
| A once-a-second debug hash of the whole world ran every frame once frames got longer than a second: 0.5 fps | Bound debug work per frame, not by wall clock. Cache expensive state hashes and invalidate them on mutation. Read the engine's own hitch dumps before guessing (here: HUD self time 1-1.8 s, GPU 7% busy). |
| Collision became pawn-radius-only and a test that waited for terrain collision with no pawn in the world hung | When a system becomes demand-driven, grep the tests for waits on its output and give each one a demand source (a probe pawn). |
| A capture run with the perf bands skipped graded a 0.5 fps build as ok | Keep a hard fps floor even when bands are off. |
| The CI runner's checkout belongs to the runner service; a desktop session tripped git ownership checks and could collide with a running job | Read it with `git -c safe.directory=...`, never pull or clean it, and launch the engine only after the runner job has exited. |
| A 'find the surface' test helper scanned down from the base radius and landed on a cave floor under a hill | Scan from the true upper bound (base radius + sky margin), never from the nominal radius. |
| A CI perf gate failed 4 runs in a row (2.5-3.7x the golden) while the same commit passed interactively on the same PC | A CI runner installed as a Windows service runs as a background process: Windows 11 puts it on E-cores with EcoQoS throttling. Start perf tests at High priority with EcoQoS off, and rerun the same commit interactively before suspecting the code. |
| Screenshot runs measured a 45-90 ms hitch that was the editor compiling a missing shader permutation, not the game | After every camera cut a capture waits in a warm phase: at least N frames, then M quiet frames (no pending shader or asset compiles, no render-state churn), with a timeout, and logs the wait. Measure only after it. |
| A hitch was chased for a day through five wrong hypotheses without engine debug symbols | Log module + RVA for each frame of the stack at the suspicious call (GetModuleHandleExW FROM_ADDRESS) and map each RVA to the nearest `dumpbin /exports` entry: one run named the caller. |
| Showcase captures pinned the universe clock for repeatable lighting, which also froze the field tick: batteries never charged and lamps never lit in any screenshot | Pin the presentation (sun position, sky) separately from the simulation clock; the sim keeps advancing under a pinned shot. A capture waits on the observed state it is meant to show (lamps lit), with a time fallback. |

## Physics and gameplay

| What happened | Rule |
| --- | --- |
| Wall ends were accepted as floor (`floor_max_angle` 80°), so a fake launch cut speed to 15% | Measure snags with a sweep tool (`tools/snag_sweep.sh`), then fix the traced mechanism. |
| Ramps had knife-edge side walls under 0.3 m | Piece rule: no collision edge under the character's radius, side walls run below ground, no coplanar floors. |
| A z-fighting floor pad | Lift the visual mesh, never the collider. |
| Same-seed physics runs split late in about 40% of fresh runs; a second match in one process always diverged | Jolt's contact order is thread-dependent: sort contacts before first-per-key debounce, and run each same-seed repeat in a fresh process. |
| A dynamic prop with a one-sided layer/mask became an immovable block that pinned a character for 40 s | Give dynamic props symmetric masks, or no layer at all; never one-sided. |
| An AI attack never hurt: it aimed open-loop and moved slower than the damage threshold | Steer AI actions on the measured result (closed loop) and check speeds against the damage thresholds. |
| A win rule had no fallback for its zero case, so every time-out of one match type was a draw | Every judge or scoring rule needs a fallback when its main measure is level. |
| A brief's threshold could never fire because it was normalised against the wrong range | Normalise between the real limits (the death line and full) and check a new threshold against the existing ones. |
| Active-ragdoll characters looked tiptoe and floating: the body was pushed while legs were posed from a sine phase, so PD-lagged legs trailed a stride behind (planted foot pitch 48°, slip 1.9 m/s) | Plan feet in world space (planted feet hold, land where the foot really is), move feet with a PD force whose reaction goes on the pelvis, and measure gait with a probe script before tuning. |
| Tucking legs on a timer while standing made a roll hop, and turning about the pelvis vaulted it into a handstand | Dip on planted feet first, tuck only once low, and let the support hold only part of the weight while tucked. |

## Rendering and art

| What happened | Rule |
| --- | --- |
| A GPU glitch was blamed on a guess | Sample screenshot pixels and reproduce first. If it can't be reproduced, ship toggles. |
| A `-s` render script couldn't see autoloads | `load()` game classes inside a deferred call. |
| Shader compile failed: `MODELVIEW_MATRIX` used in a fragment shader; a disc rendered culled | `MODELVIEW_MATRIX` is vertex-only (use `VIEW_MATRIX * MODEL_MATRIX` in fragment); Godot front faces wind clockwise. |
| Hand-built meshes were inside out but looked fine in previews (previews draw both sides) | Check face direction numerically (normals against the outward direction) on any hand-built mesh. |
| A toon shader blew lit surfaces out to flat colour: the light term ignored that `LIGHT_COLOR` already includes PI, and a white rim was added on lit faces | Normalise light so lit albedo ≈ albedo; check every shared-shader change on level, character and UI renders. |
| Thin decorative meshes cast hard shadow streaks; a cool sky ambient made warm ground read green in shadow | Only big shapes cast shadows; use a warm ambient on warm ground and sample shadow hues before blaming shaders. |
| A shader-hidden crowd twin (`VERTEX *= 0`) still cost its vertices, and Godot's primitive counter ignores MultiMesh instances, so the crowd looked cheap | Cut instances with `visible_instance_count`; count MultiMesh triangles yourself and time script cost separately. |
| Distant crowds and props cost too much | Visibility ranges (HLOD) work on `MultiMeshInstance3D`; use decimated meshes or impostors at distance. |
| Match-start hitches from first-use shader compiles (Godot 4.4+ ubershaders precompile only what is loaded at load time) | Instance every effect resource once off-screen at match start; keep GPU perf on real hardware (xvfb skews GPU frame times) and CI on draw calls, primitives and script time. |
| A humanoid character was rebuilt ten times (code lofts, Blender, Houdini, polish passes) and still read as a placeholder | Never sculpt humanoids in code. Start from a creator-made or licensed model on a standard humanoid skeleton; follow `docs/character_creation_lessons.md`. |
| The CI runner runs as a service with no GPU: offscreen captures were black at 0.2 fps | Take screenshots in the owner's desktop session (Remote Control or a desktop runner), never from a service-account runner. |
| UE manual exposure still applied the physical camera (ISO 100, 1/60 s, f/4, about EV 10), so a 10 lux sun rendered pitch black | In manual exposure turn off Apply Physical Camera Exposure, then tune the bias against a grey-card capture. |
| A generated material rebuilt in place kept stale nodes from an older build; an unwired texture-array sample made the engine use the default material | Delete and recreate generated materials, assert every sampler input is wired after building, and fail the asset step on any `Failed to compile Material` log line. |
| The showcase camera used the gameplay spawn and ended inside a block, with the aim outline across the frame | Showcase views get dedicated cameras (overlook, sunlit orbit) and hide gameplay debug drawing via a cvar. |
| The first look at the planet from orbit showed the north mirroring the south exactly, and terrain smeared into stripes around the equator | On a spherical world, sample every surface term (height, heat, moisture, latitude) by direction, using 3D noise at the point on the datum sphere, never by world (x, z). Look at generated worlds from far away early. |
| In space a white "second sun" floated beside the star | The sky atmosphere draws its sun disc along the light direction from the camera, which drifts away from the star as you leave the planet. Fade the disc (or the atmosphere) out with altitude. |
| A procedural tile kept its old glowing look after its recipe changed | A "bake only what is missing" mode never re-bakes a changed recipe. Bake tools need a `--force <asset>` option; pass it for every changed source and look at the tile before pushing. |

## Houdini and assets

| What happened | Rule |
| --- | --- |
| The cloud proxy blocked many asset sites, discovered late | On day one, test-download one file from each asset source; if blocked, fetch through the owner's machine into an asset branch. |
| An asset's polish was thrown away by the game's loader | Before building an asset the game will use, read the importer or loader code and check which data survives it. |
| Two instances of one glb shared a cached `Mesh`; per-instance override materials errored at teardown | Duplicate an instanced glb's mesh before setting surface override materials. |
| Houdini's glTF ROP exported every material as `alphaMode BLEND` | Force imported art opaque on import (or set it in the ROP) and assert `alphaMode` in the exporter test. |
| Houdini build scripts failed on missing folders (`hipFile.save` included) and drifted through copy-pasted helpers | Create hip/export/preview folders first; share helper libs (`LIB`, `read_glb`) instead of copying; every build script takes `--only=<ids>` and `--no-preview`. |
| Emissive parts blew out to white in previews | Use modest emission or a dark preview background. |
| Armor looked fitted by bounding-box overlap but clipped the body | Check fit radially: distance to the part axis minus radius >= 0. |
| Named attachment empties were lost in glTF export, and mirrored parts came out inside-out | Export in object mode (objpath + objects) with null OBJs for named empties; mirror with a negative-scale Transform plus a Reverse SOP. |
| Impostor cards and reduced meshes lost their tint masks | PolyReduce keeps vertex-colour masks; render impostor cards with flat light so masks stay tintable. |
| PowerShell 5 wrote JSON with a UTF-8 BOM that parsers rejected | Write JSON without a BOM (`[IO.File]::WriteAllText` with `UTF8Encoding($false)`). |
| A local Godot version bump rewrote every `.import` file | Never commit that churn; revert the `.import` files before staging. |

## Online

| What happened | Rule |
| --- | --- |
| Closing an ENet peer from inside its own `poll()` (a disconnect signal handler) crashed with signal 11 | Never close a MultiplayerPeer from its own signal; defer the close to the next idle frame. |
| ENet's packet throttle silently dropped about half the unreliable snapshots after a frame hitch | Switch the ENet throttle off on connect (deceleration 0) for snapshot streams. |
| A reliable events packet could exceed a relay's 1164-byte limit and was silently dropped | Split network packets by bytes under the relay limit, not by record count. |
| A net soak recorded the host pose one tick off the snapshot's stamp, tripling every measured error | Record the reference state at the exact tick the snapshot is stamped with; sanity-check a new harness on a clean link first. |
| An online-service plugin's create-user login path fired no login callback, so the runtime never learned the local user | Log in a second time after creating the device user; mock a plugin's native singleton to run its real GDScript without the binaries. |
| Secrets (online service client keys) must ship in builds but not in a public repo | Keep them in a git-ignored `config/*credentials*` file injected by the export script; never write them to memory or docs (`docs/online.md`). |

## Android/touch

| What happened | Rule |
| --- | --- |
| `dl.google.com` is blocked in the cloud sandbox, so the Android SDK could not be installed the usual way | `tools/export_android.sh` assembles JDK 17 and the SDK tools from distro packages under `$HOME`; on a real machine set `ANDROID_SDK_ROOT`. |
| The Android export refused to run | Keep `textures/vram_compression/import_etc2_astc=true` in `project.godot` (the kit has it). |
| An APK has no folder beside it, so a credentials file copied next to the build never reached the phone | Put the git-ignored credentials file in the Android preset's `include_filter`; the export script warns when it is missing. |
| A rendering bug survived many guesses because every probe read game-thread state | Write one probe that reads what the renderer actually published (proxy, draw masks, ranges) before guessing again; plugin ids built from names can collide silently. |
| A diagnostic probe traced the previous view | Run probes a few frames after the camera settles and log the camera they used. |
| Gatherer bots jammed outside town gates (M4 soak: 69 of 127 on pz, mining never happened) | A distance field that routes through a walkable town contradicts "never step onto pz": compute per-goal fields with pz blocked unless the goal is inside town (stations); check the soak's "where are the bots" line before the kill or gather count. |
| Two bots descending different distance fields stood face to face forever (slots never moved in 60 ticks) | Treat tiles holding a living player as blocked in the descent so the loser sidesteps; a single shared field (dist_town) never had the problem, so test every new field policy with two kinds of bots in one town. |
