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
