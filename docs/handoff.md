# Handoff: JAX tile MMO

For the next lead. Read this, then docs/brief.md (module contracts and phase order).

## State
- Project: `C:\Users\crp27\jaxspike`. Python venv `venv` (CPU jax 0.12.0.1, pytest, pygame not available). GPU runs use WSL Ubuntu with `~/jaxgpu` (JAX CUDA).
- Modules: `game/` (config, state, spatial, map_gen, movement, monsters, combat, loot, step), `bots/` (policies, runner), `game/persistence.py`, `bench/` (soak_check with `--batch/--chunks/--ticks/--seed`, bench_step), `tests/` (62 passing incl. test_spatial.py), `play.py` (tkinter playable window, CPU).
- Verified: 62 tests pass on CPU. A 1000-tick soak on CPU with zero invariant violations. GPU soak results by batch size are in the "GPU batch" section below.
- Not verified: the play.py visuals (launched for 20 s without a traceback, but never seen on screen by a lead). Balance is tuned from bot soaks, not from human playtesting.

## Dead players are not physical (2026-11-02)
A 512-world GPU soak found two living players on one tile (2 of 512 worlds). Cause: `combat._respawn_players` ignored dead players when checking the spawn tile while `movement` and `monsters._occupancy` treated dead bodies as solid, so a respawn could land on a dead player who was stuck waiting (its own spawn blocked). Decision: dead players never occupy a tile anywhere (Tibia: you are gone until you reappear at the temple). Making corpses solid instead would allow a permanent deadlock when two dead players stand on each other's spawn. The soak invariant is now "no two *living* players share a tile".

## Spatial grid capacity
`tests/test_spatial.py` packs 300 players into a 16x16 town (4 cells): with K_CELL=8 only 32 of them are visible to the grid (268 dropped, up to 79 in one cell). Correctness holds (queries still find a target), but a town where players stand shoulder to shoulder needs K_CELL of 16-64; the cost is 9*K_CELL candidates per query. Decide this when towns exist.

## Lessons from this build
1. **Invariant soak beats unit tests for integration.** The soak (bench/soak_check.py) found two real bugs the unit tests missed: monsters and players respawning onto occupied tiles, and HP going negative. Run it after every integration.
2. **Avoid per-entity tables under vmap.** `jax.random.categorical` over all tiles for each spawn built a table of spawns x tiles per world, which exhausted GPU memory at batch 16. Use one ranking over tiles (argsort) or a cumulative count. Batched scatters into full-map arrays and searchsorted with many queries are suspect too.
3. **Spawns must be distinct.** Sampling with replacement put monsters and players on the same tile at spawn (60 of 64 worlds). Draw all spawn tiles from one permutation.
4. **JAX errors are asynchronous.** The failure you see may come from a different op than the one named. Test one piece at a time, under a memory cap.
5. **Cap GPU memory on every run:** `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` (the user raised the cap from 40% to 80%). The user's machine nearly crashed on uncapped batch runs. Never probe batch sizes above what has been verified without the cap.
12. **Scale finds rare bugs.** The dead-player overlap only showed at 512 worlds x 1000 ticks (2 hits in 512,000 world-ticks). Run the biggest soak the cap allows after any change to movement, respawn, or occupancy.
6. **Worker briefs with file ownership worked.** Eight parallel Sonnet workers, each owning its own files and tests, integrated with no conflicts. Shared contracts (state schema, phase order, spatial API) must land first, and the lead must write them.
7. **Tests encode numbers.** When balance changes, update the tests that hard-code old values (combat kill test, loot roll table). Prefer config constants in tests.
8. **Balance from the soak, not from guesses.** Measure kills, deaths, and items per 1000 ticks, change one group of numbers, remeasure.
9. **Look before you claim.** The lead can't see the game window. Say so, and ask the user to check visuals.
10. **Resource limits.** This machine has 24 cores and 8 GB of GPU memory shared with other apps. Heavy parallel runs (dozens of agents or processes) can overload the machine. Start with the caps in lesson 5, and raise them only with the user's explicit OK.
11. **GitHub minutes are unavailable.** Keep everything local. Don't push or use cloud sessions for this project.

## GPU batch (WSL, 80% cap, 1000-tick soak, 128 players + 512 monsters per world)
| batch | result | time | peak GPU mem |
|---|---|---|---|
| 512 | clean (after the dead-player fix; before it, 2 worlds overlapped) | 61 s | not sampled |
| 1024 | clean | 111 s | 5470 MiB of 8151 (67%) |
| 2048 | fails cleanly: RESOURCE_EXHAUSTED on a 4.37 GiB allocation inside jit_scan, memory freed | - | - |
1024 is the verified ceiling under the cap. Going higher means shrinking per-world memory (the scan carry is the whole batched world), not raising the cap.

## Balance (bot soak, 8 worlds x 1000 ticks, 1024 bots)
Targets chosen by the lead: 0.3-0.6 deaths and 5-10 kills per bot per 1000 ticks. Full sweep (20 runs) is in the balance worker's table in the session log; summary:
- Old (2,7) damage, cooldown 6: 7701 kills, 1582 deaths (7.5 / 1.55 per bot).
- New (1,3) damage, cooldown 8: 8402 kills, 512 deaths on seed 0; 8064 / 502 on seed 1; 8106 / 463 on seed 2 (about 8 / 0.5 per bot). Items held stayed within 6%.
- Damage alone only hit the target at (1,2), which makes orcs trivial; adding cooldown 8 let orcs keep 3 damage. Raising player HP also worked but was rejected as inflation.
- Loot thresholds now live in config (LOOT_*_MAX) and the loot test reads them.
- Caveat: no healing exists, so every bot eventually dies; deaths scale almost linearly with incoming damage. Healing (eat meat, regen) is a mechanic to add before tuning further.

## Open work, in priority order
1. Visual check of play.py by the user, then adjust graphics.
2. Playtest balance with a human player and tune; add a healing mechanic first.
3. Decide K_CELL for towns (see "Spatial grid capacity").
4. To batch beyond 1024 on this GPU, reduce per-world state (e.g. MAX_MONSTERS/MAX_DROPS closer to population) rather than raise the cap.

## Rules for the next lead
- GPU memory cap is 80% (user-approved): `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Never run above it.
- Sub-agents are Sonnet. Up to 75 if the machine keeps working; back off when CPU, RAM, or GPU load gets high (24 cores, shared GPU).
- Run `venv\Scripts\python.exe -m pytest tests -q` and the soak after every change.
- Keep scratch files in `bench/` and clean them up.
