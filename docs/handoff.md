# Handoff: JAX tile MMO

For the next lead. Read this, then docs/brief.md (module contracts and phase order).

## State
- Project: `C:\Users\crp27\jaxspike`. Python venv `venv` (CPU jax 0.12.0.1, pytest, pygame not available). GPU runs use WSL Ubuntu with `~/jaxgpu` (JAX CUDA).
- Modules: `game/` (config, state, spatial, world_layout, map_gen, movement, monsters, combat, loot, survival, step), `bots/` (policies, runner), `game/persistence.py`, `bench/` (soak_check with `--batch/--chunks/--ticks/--seed`, bench_step), `net/` (characters, pathing, protocol, server, client), `tests/` (136 passing), `play.py` (tkinter fullscreen network client, CPU).
- Run: server `venv\Scripts\python.exe -m net.server`, client `venv\Scripts\python.exe play.py --name you` (add `--host`/`--port` for a remote server). Load bench: `venv\Scripts\python.exe bench/server_load.py --embedded --clients 50 --seconds 10`.
- Verified: 136 tests pass on CPU. 1000-tick CPU soaks on seeds 0 and 1 with zero invariant violations on the 5-town map with hunting bots. GPU soak results by batch size are in the "GPU batch" section below. The user played the window (walked town to town, killed monsters) and called the world solid.
- Balance on the 5-town map is tuned with hunting bots (see "Balance").
- Direction (2026-11-02, from the user): a public Tibia-style MMO, hybrid architecture (JAX tick + Python server layer). Spec and plan for milestone 1: `docs/superpowers/specs/2026-11-02-world-survival-design.md`, `docs/superpowers/plans/2026-11-02-world-survival.md`. Milestones: (1) world + survival [done], (2) networked server + client [done], (3) items/equipment/spells, (4) skilling, (5) pygame-ce renderer with Kenney tiles.
- Local git repo (no remote): history of milestone 1 is in `git log`.

## World + survival (milestone 1, 2026-11-02)
- `MAP = 512`; `world["terrain"]` (ids `T_*`) and `world["pz"]` (protection zone); `walls` derived. Layout in `game/world_layout.py` (5 towns, 8 roads, 12x12 temples, in map fractions: 2048 is a one-constant change).
- Players spawn on their home temple (slot mod 5); monsters spawn off-PZ, rats within `RAT_RING` of a town, orcs beyond (about 160 rats / 350 orcs per world, an average over an 8-world batch). Monsters cannot enter towns or hit a player standing in one.
- `survival` phase: +1 HP every 4 ticks, meat heals 40 on `inputs["eat"]`; bots eat below half HP.

## Milestone 2: networked server (2026-11-02)
Spec `docs/superpowers/specs/2026-11-02-networked-server-design.md`, plan `docs/superpowers/plans/2026-11-02-networked-server.md`, module contracts in `docs/brief.md` ("Networking (net/)").
- One JAX world, 512 slots. WebSocket + JSON, name-only login (`[a-z0-9_]{3,16}`, lower-cased, one session per name). Every tick the server sends each client a viewport snapshot (61x33 window around the player); terrain only when the window moved.
- What persists: per character `characters/<name>.npz` with pos, spawn, hp, inv, kills, deaths, town, plus `characters/roster.json` (`{name: town}`). Saved on logout, every `SAVE_EVERY_TICKS` (2400), and on shutdown. A missing file makes a fresh character on the next round-robin home town's temple; a corrupt file refuses the login with an error.
- Slot order on login: an inactive slot with `slot % NT == town`, then a bot slot in that stripe, then any inactive, then any bot, else "server full". On logout the slot is re-seeded as a bot on its town's temple (full HP, empty inventory, counters 0).
- Click-to-walk is server-side A* (`net/pathing.py`, radius 32; out of map, too far or unreachable gives notice "too far" and keeps the connection; wrong-typed messages get error "malformed message" + close, a first message that is not a valid login gets "expected login" + close); the path emits one move per tick, a blocked move retries and the path drops after `PATH_STUCK_TICKS` ticks without progress. Any `input` cancels it.
- Death does not log out: `notice: "you died"`, respawn at the temple as in milestone 1.
- Not done: passwords/accounts, chat, multiple worlds, binary protocol, persisting the world (drops, monster timers) across restarts. The temple-spawn griefing item under Open work is still open.
- Scale notes: snapshot cost is per connected client; if the tick overruns the first moves are terrain as packed bytes and snapshot building in a thread pool (protocol shape unchanged).
- Load bench (`bench/server_load.py`, CPU, embedded, random inputs every 0.5 s, 10 s; bytes = compact-JSON size of each received snap, with the terrain window counted only when its object changed, i.e. when the server actually sent one; an estimate of wire payload): 50 clients: all connected, 8.40 snaps/client/s (min 8.40, nominal 8.33), ~37.9 KB/client/s mean (max 49.0 KB/s), 0 overruns in 87 ticks. 200 clients: all connected, 8.40 mean (min 8.30), ~49.2 KB/client/s mean (max 64.1 KB/s), 0 overruns in 96 ticks. Terrain is resent on every step, so moving clients dominate bandwidth (packed terrain is the first lever). Server and clients share one process here, so this is a floor.

## Distance fields and hunting bots (2026-11-02, later)
The concept that keeps goal-directed behaviour in JAX: **precomputed distance fields instead of pathfinding.**
- `map_gen.gate_aprons(terrain)` -> bool `[NT, MAP, MAP]`: walkable non-town tiles one step outside each town's gates. `map_gen.distance_fields(terrain)` -> int16 `[NT, MAP, MAP]`: walking distance from every tile to each town's aprons, by `d = min(d, 1 + min(neighbours))` relaxation in a `lax.while_loop` (unreachable = `DIST_UNREACHABLE`). Stored as `world["dist_town"]`; `make_world_batch` fills it with `lax.map` (sequential over worlds, so the loop's temporaries are never batched). About 1 s per world on CPU.
- "Step toward X" is then: look at the four neighbours, pick the smallest value; "away" picks the largest. Constant cost, vmappable, no search. The same field gives "monsters never spawn within `GATE_APRON_TILES` of a gate" and "rats live within `RAT_RING` walking distance of a gate" for free.
- Bot policy (`bots/policies.py`, stateless, derived from the world each tick): below `HUNT_RETREAT_HP` walk home (smallest neighbour) ignoring monsters; on pz below max HP rest (stand still, regen + eat); healthy in town walk to the gate; outside and closer than `HUNT_RADIUS // 2` to the gate walk away; beyond `2 * HUNT_RADIUS` walk back; in between keep a wander heading that is a pure function of (slot, tick // HEADING_TICKS). Chase and attack as before when a monster is within 6.
- Two traps found on the way, both fixed and pinned by tests: (1) indexing `dist_town[home]` per player materialises `[MAX_PLAYERS, MAP, MAP]` (268 MB per world); index `field[home, x, y]` directly. (2) "away = largest neighbour" from the apron tile tied between the gate behind and the road ahead and chose the gate (first index), so the apron bot and the gate bot waited on each other forever: 40% of all bots were queued in towns. Away-steps now never enter pz. Lesson: after a policy change, print where the bots *are*, not only the kill count.
- Soak after the change: 2.2 kills per bot per 1000 ticks with 512 monsters, 3.3-4.0 with 1024 (now the default `POP_MONSTERS`). Rats stay ~80% alive: encounter rate, not combat, bounds kills. Bot chase range 8 instead of 6 made no difference.

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
13. **Ties deadlock.** Any "pick the best neighbour" rule with first-index tie-breaks can make two entities wait on each other's tile forever (the gate jam). Exclude tiles the entity should never want, and check where entities *are* after a soak.
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
1024 was the verified ceiling under the cap on the 256x256 map.

MAP = 512 (5-town map), same cap, 1024 monsters per world, hunting bots, `dist_town` carried per world (2.6 MB each):
| batch | result | time |
|---|---|---|
| 64 / 128 / 256 | clean | 31 s / 34 s / 50 s |
| 512 | fails cleanly: RESOURCE_EXHAUSTED on a 1.27 GiB allocation in `reduce_min` (the `dist_town.min(axis=0)` in spawn, batched) | - |
(Before distance fields, with 512 monsters: 21 / 26 / 36 s, and 512 failed on a 4 GiB `jit_scatter`.) 256 is the ceiling at MAP=512. The per-world MAP*MAP tables (`monsters._chase` claim table, `combat._respawn_players` occupancy, the batched spawn reduce) are what grow; moving them to the sort-based approach in `movement.py`, or spawning with `lax.map` like the fields, would raise it. For one live world none of this matters.

## Balance (bot soak, 8 worlds x 1000 ticks, 1024 bots, 5-town map, hunting bots)
Targets chosen by the lead: 0.3-0.6 deaths and 5-10 kills per bot per 1000 ticks. Current: `MONSTER_DAMAGE = (3, 9)`, cooldown 8, 1024 monsters per world.
| damage | kills / bot | deaths / bot | items / bot |
|---|---|---|---|
| (1,3) | 3.97 | 0.02 | 1.32 |
| (2,6) | 3.66 | 0.23 | 1.14 |
| (3,9) seeds 0/1/2 | 3.28 / 3.41 / 3.43 | 0.42 / 0.54 / 0.50 | ~1.0 |
| (4,12) | 2.94 | 0.61 | 0.86 |
- Final GPU 256-world soak on this code: 3.23 kills and 0.45 deaths per bot, invariants held. Only ~66 of the 1024 monsters per world are rats (gate-distance < `RAT_RING * MAP` covers a small band around 10 gates); the rest are orcs. Raising `RAT_RING` makes more of the hunting band rat territory.
- Deaths are in target. Kills (3.3-3.4) are below the 5-10 target that was set on the old open map; on this map they are bounded by how often a bot meets a monster (rats stay ~80% alive), not by combat. Levers if more kills are wanted: more monsters near gates, a bot that walks toward the nearest monster it can see further away (needs a bigger grid block than 3x3 cells), or monsters that wander.
- Deaths come from orcs (9 damage) chasing a bot that retreats at 60 HP; rats alone cannot kill a bot. That is the intended feel: rats are safe, orcs are not.
- Old-map history: (2,7)/cooldown 6 gave 7.5 kills and 1.55 deaths per bot with no healing; (1,3)/cooldown 8 gave 8 / 0.5. With regen, eating and retreat, (1,3) gives 0.02 deaths.
- Loot thresholds live in config (LOOT_*_MAX) and the loot test reads them.

## Open work, in priority order
1. Milestone 3: items/equipment/spells (spec first). Builds on the network layer: new item ids and equipment slots extend `you.inv` and the character files.
   - Spec item: A living player idling on another player's temple spawn tile blocks that respawn indefinitely (griefing vector on a public server): consider letting living players overlap on temple tiles or respawning on the nearest free temple tile.
   - Milestone 2 leftovers: passwords, chat, world persistence, binary protocol / packed terrain if the tick overruns at scale.
2. Milestones 4-5 (skilling, pygame-ce renderer with Kenney tiles). Concepts agreed for staying in JAX: tables indexed by id for items/recipes/monster kinds; per-player xp arrays for skills; counters on entities for timers; distance fields for every "go to X".
3. Decide K_CELL for towns (see "Spatial grid capacity"); temples now hold up to ~26 players per 8x8 cell.
4. Tiled import for hand-authored towns; MAP = 2048 for the first playable release (distance fields: 42 MB per world, computed once).
5. Done since the last handoff: hunting bots, monster tile-sharing on revive (fixed, soak invariant added), gate-apron spawns (excluded), balance re-tuned.

## Rules for the next lead
- GPU memory cap is 80% (user-approved): `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Never run above it.
- Sub-agents are Sonnet. Up to 75 if the machine keeps working; back off when CPU, RAM, or GPU load gets high (24 cores, shared GPU).
- Run `venv\Scripts\python.exe -m pytest tests -q` and the soak after every change.
- Keep scratch files in `bench/` and clean them up.
