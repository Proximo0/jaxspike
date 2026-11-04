# Kit harvest: what to send back to game-starter-kit

Source: Jaxspike rounds 1 to 3 (2026-11-02 to 2026-11-03). Target: https://github.com/Proximo0/game-starter-kit (not pushed from
here; the lead opens the PR). Checked against the kit's current `docs/lessons_learned.md`, `docs/lessons/ops_board.md`
and `tools/`: nothing below is already there.

## 1. Rows for `docs/lessons_learned.md`

Add under a new section "## Fan-out and shared checkouts" (the kit has Dashboards, Online and Android sections, none for
running many sessions). Same `what happened | rule` format.

| What happened | Rule |
| --- | --- |
| Nine sessions started in their own worktrees never got their task: the desktop app showed a "Trust repository" button in each new session and nobody clicked | Trust is per folder. Warn the user before a worktree fan-out that every new session needs one click, or run the sessions on the shared checkout with disjoint file ownership. |
| Twelve teams shared one checkout and one PC | Shared checkout rules: teams touch only the paths the contract table gives them, never run git (the lead commits files by name), and edit hot-spot files with small anchored edits after a re-read. Cap each team's parallel test processes. |
| A lab feel metric (`apply_ms_p95`, target 2.0, measured about 1.4) and a 20 ms-tick timing test failed while other sessions loaded the CPU | A timing or feel number measured on a loaded PC measures the load. Rerun a failing timing test alone before calling it a regression; gate feel metrics on a quiet machine or CI, and write the load next to the number. |
| scrypt password checks took 74 ms against a 12 ms tick budget; JSON snapshot encoding ran beside the tick | Anything longer than the tick budget goes to `run_in_executor`, but the executor shares the GIL: only calls that release it (hashlib, jitted JAX) overlap with the loop. Measure tick overruns under a load bench. |
| Nine branches merged into lead-owned shared files (`state.py`, `step.py`, `config.py`) | Merge order is part of the contract (foundations first). The lead wires shared files from each branch's reported lines, one branch at a time, with the suite and a soak after each. |
| Skills and stats each kept an SP value and the HUD and cost check disagreed | One owner per quantity (HP, SP, money, exp). The contract names the owning block; others read it, and a test asserts no second copy exists. |
| The branch-based ops board had nothing to show once the work moved to a branch-less shared checkout | Show progress by ownership area: files changed under each team's paths since the contract commit, newest file time, new or changed tests that collect. Parse the paths from the contract table. |
| The engine CI job failed twice on the round-2 contract commit: a server test needed `after[1] > before[1]` within 5 snaps at a 50 ms tick, and the first test's jit compile on a 2-core runner starved the tick loop (`tick 7 overrun`). A second test used `h["s"]` before the server thread had set it | A socket or tick test waits on a condition with a generous budget (the pipelined loop adds a tick of input latency, so 20 snaps, 120 s for the server to be ready), never on a fixed few snaps. The fix is the longer wait with the reason in a comment; never skip or xfail to turn CI green. After the wait change main went green (engine 689 passed, client 204 on dbe5049). |
| A `concurrency: cancel-in-progress` group cancelled the run of a commit pushed right after another (1d306aa), so one commit had no CI result | Read the run list per sha before trusting a green main; a cancelled run is not a pass. Re-dispatch with `gh workflow run checks --ref main` if the head sha has no finished run. |
| The ops collector read the team table from a hard-coded round-2 contract and a team added after the contract (ro3-names) had no row | Parse the team table from the current round's contract, keep a small `EXTRA_TEAMS` map for teams added mid-round, and bump `--baseline` to the new contract commit each round. |
| GPU soak numbers (377 s for 64 worlds x 1000 ticks, kills 2.2 and deaths 1.1 per bot per 1000 ticks) come from WSL runs the collector cannot repeat | Put facts measured elsewhere in `tools/ops/manual_tests.json` with their source named, so the board shows them without implying the collector measured them. |
| Board rendering was reported done from data alone in round 2 | Open the published board in a browser and look at a screenshot after every page or dataset change; a green refresh line does not prove the page draws. |

## 2. Dashboards section addition

| What happened | Rule |
| --- | --- |
| Asset-backed datasets (`/_blob/<id>`) need the new url written to each dataset document after every upload; an upload alone changes nothing on the board | Make the refresh script's last line say which files changed, and make the republish a fixed three-step recipe: upload each changed JSON (one call per text file), one `batch` that updates each dataset's `source.url` and `updated`, then the page only when it changed. |

## 3. Tool changes

| Jaxspike file | Kit change |
| --- | --- |
| `tools/ops/refresh_ops.py` (`c_teams`, `parse_teams`, `--baseline`, `--no-collect`) | Replace the kit's branch-only collector with an optional team view: owned-path patterns parsed from a contract table (`| **team** | owns | delivers |`), changed files against a baseline commit via read-only `git diff --numstat` plus untracked files, newest mtime, and `pytest --collect-only -q` of each team's test files. Adds a `teams` dataset and the `teams_active` / `teams_stale` / `tests_collected` summary fields. Generic: only `CONTRACT`, `TEAM_TEST_KEYS` and the baseline default are project data. |
| `tools/ops/index.html` | The team-card section and two KPI tiles (`teams_active`, `teams_stale`) as the branchless variant of the branch cards. |
| `tools/ops/manual_tests.json` and the `EXTRA_TEAMS` map in `refresh_ops.py` | Facts measured outside the collector (GPU soak) shown with their source, and teams added mid-round. |
| `.github/workflows/export-windows.yml` | Manual-dispatch Windows export on Ubuntu: `setup-godot` action, `--import`, `tools/export_windows.sh`, `upload-artifact` of the zip. The kit has `checks.yml` only. |
| `client/tools/winshim/flock` and the `GODOT=... PATH="$PWD/tools/winshim:$PATH"` recipe in the client CLAUDE.md | A Git Bash shim for the missing `flock` on Windows, so the kit's tools run on a Windows PC. |
| `client/tools/export_windows_pc.sh` | Windows-PC export with templates on a chosen drive and a dated zip, next to the Linux `export_windows.sh`. |

## 4. How to send it

Copy sections 1 and 2 into the kit's `docs/lessons_learned.md`, the files in section 3 into `tools/` and
`.github/workflows/`, and open one PR from a branch of game-starter-kit. Do not push from the Jaxspike checkout.
