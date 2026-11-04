# Lessons learned (engine and process)

Add a row the day something goes wrong, in the format `what happened | rule`. The client has its own file at
`client/docs/lessons_learned.md`; its "Dashboards" section also applies to the ops board (`docs/ops.md`). The
first rows are seeded from `docs/handoff.md`, "Lessons from this build".

## Engine (JAX)

| What happened | Rule |
| --- | --- |
| The soak (`bench/soak_check.py`) found two real bugs the unit tests missed: monsters and players respawning onto occupied tiles, and HP going negative | Invariant soak beats unit tests for integration. Run it after every integration. |
| `jax.random.categorical` over all tiles for each spawn built a spawns x tiles table per world and exhausted GPU memory at batch 16 | Avoid per-entity tables under vmap. Use one ranking over tiles (argsort) or a cumulative count. Batched scatters into full-map arrays and searchsorted with many queries are suspect too. |
| Sampling spawn tiles with replacement put monsters and players on the same tile (60 of 64 worlds) | Spawns must be distinct: draw all spawn tiles from one permutation. |
| The failure JAX reported came from a different op than the one named | JAX errors are asynchronous. Test one piece at a time, under a memory cap. |
| Uncapped batch runs nearly crashed the user's machine | Cap GPU memory on every run: `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` (80%, user-approved). Never probe batch sizes above what has been verified under the cap. |
| Two entities waited on each other's tile forever (the gate jam) under a "pick the best neighbour" rule with first-index tie-breaks | Ties deadlock. Exclude tiles the entity should never want, and check where entities *are* after a soak. |
| The dead-player overlap only showed at 512 worlds x 1000 ticks (2 hits in 512,000 world-ticks) | Scale finds rare bugs. Run the biggest soak the cap allows after any change to movement, respawn or occupancy. |
| Balance changes broke tests that hard-coded old values (combat kill test, loot roll table) | Tests encode numbers. Prefer config constants in tests and update the hard-coded ones when balance changes. |
| Balance tuned from guesses drifted | Balance from the soak, not from guesses: measure kills, deaths and items per 1000 ticks, change one group of numbers, remeasure. |
| MAP=512 batch 512 failed on a 1.27 GiB `reduce_min` (the batched `dist_town.min` in spawn) and earlier on a 4 GiB `jit_scatter` | Per-world MAP*MAP tables (claim tables, occupancy maps, batched spawn reduces) set the batch ceiling. Prefer the sort-based approach in `movement.py` or `lax.map` over the batch for such tables. |

## Process

| What happened | Rule |
| --- | --- |
| Eight parallel Sonnet workers, each owning its own files and tests, integrated with no conflicts | Worker briefs with file ownership work. Shared contracts (state schema, phase order, spatial API) land first, and the lead writes them. |
| The lead reported visuals it could not see | Look before you claim. The lead cannot see the game window: say so and ask the user to check visuals. |
| Heavy parallel runs (dozens of agents or processes) overloaded a 24-core PC with 8 GB of GPU memory shared with other apps | Start with the GPU cap above and raise concurrency only with the user's explicit OK. Run heavy GPU jobs one at a time. |
| The previous build kept everything local because GitHub minutes were unavailable; this build runs CI on a public repo | Re-check infrastructure assumptions at each kickoff; the project memory records the current one. |
| Nine team sessions were started on branches named `claude/ro-*-<hash>` by the app while the contract said `ro/*` | A board that matches only the documented branch pattern misses the real branches. Collect with both patterns and show what was found, not what was planned. |

## Round 2 (shared checkout, twelve teams)

| What happened | Rule |
| --- | --- |
| Nine sessions started in their own worktrees never received their task: Claude Desktop showed a "Trust repository" button in each new session, and none was clicked | Trust is per folder. Tell the user up front that every worktree session needs one click, or start the sessions on the shared checkout (round 2 did, with disjoint file ownership, no git for teams, the lead committing by name). |
| The client lab's `apply_ms_p95` (target 2.0, measured ~1.4) and an engine timing test (`assert after[1] > before[1]`, 20 ms ticks) failed when run next to other sessions | A feel metric or timing test measured on a PC loaded by a dozen sessions measures the load. Record the load with the number, rerun a failing timing test alone before calling it a regression, and gate feel metrics on a quiet machine or CI. |
| scrypt password verify took 74 ms mean (p95 103 ms) against a 12 ms budget (10% of `TICK_MS`); snapshot JSON encoding competed with the tick on the same process | Work longer than the tick budget never runs on the event loop: use `loop.run_in_executor`. The executor shares the GIL: only calls that release it (`hashlib.scrypt`, jitted JAX) overlap with the loop, pure-Python encoding does not. Measure with `bench/server_load.py` overruns, not by reasoning. |
| Nine branches merged in the order stats, skills, items, craft, party; each report listed lines for the lead-owned `state.py`, `step.py`, `config.py` and `empty_inputs` | Merge order is part of the contract: foundations first, and the lead wires shared files from the reports one branch at a time, running the suite and a soak after each merge. In a shared checkout name the hot-spot files up front (`game/monsters.py`, `net/protocol.py`, `game/view.py`, `game/data/*`) and restrict teams to small anchored edits there. |
| Skills and stats each kept their own SP, so the HUD and the cost check disagreed (fixed in 2940d66 and b244f14: `stats.sp` is the one SP pool) | One owner per quantity. The contract names the block that owns HP, SP, ryn and exp; other blocks read it, never copy it, and a test asserts no second pool exists. |
| A 12-team round needs a board with no branches: the Round 1 board showed branch cards, and branch-less teams edit the same checkout | Show progress by ownership area: files changed under each team's paths since the contract commit, newest file time, and the new or changed tests that collect. Parse the owned paths from the contract table so the board follows the contract. |

## Round 3 (twelve teams, defect round)

| What happened | Rule |
| --- | --- |
| The engine CI job failed twice on the round-2 contract commit: a server test needed `after[1] > before[1]` within 5 snaps at a 50 ms tick, and the first test's jit compile on a 2-core runner starved the tick loop (`tick 7 overrun`). A second test used `h["s"]` before the server thread had set it | A socket or tick test waits on a condition with a generous budget (the pipelined loop adds a tick of input latency, so 20 snaps, 120 s for the server to be ready), never on a fixed few snaps. The fix is the longer wait with the reason in a comment; never skip or xfail to turn CI green. After the wait change main went green (engine 689 passed, client 204 on dbe5049). |
| A `concurrency: cancel-in-progress` group cancelled the run of a commit pushed right after another (1d306aa), so one commit had no CI result | Read the run list per sha before trusting a green main; a cancelled run is not a pass. Re-dispatch with `gh workflow run checks --ref main` if the head sha has no finished run. |
| The ops collector read the team table from a hard-coded round-2 contract and a team added after the contract (ro3-names) had no row | Parse the team table from the current round's contract, keep a small `EXTRA_TEAMS` map for teams added mid-round, and bump `--baseline` to the new contract commit each round. |
| GPU soak numbers (377 s for 64 worlds x 1000 ticks, kills 2.2 and deaths 1.1 per bot per 1000 ticks) come from WSL runs the collector cannot repeat | Put facts measured elsewhere in `tools/ops/manual_tests.json` with their source named, so the board shows them without implying the collector measured them. |
| Board rendering was reported done from data alone in round 2 | Open the published board in a browser and look at a screenshot after every page or dataset change; a green refresh line does not prove the page draws. |
