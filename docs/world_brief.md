# RO world content: worker contracts (branch `ro/world`)

Sub-orchestrator-owned. Read `CLAUDE.md`, `docs/ro_contract.md` and the module contracts in `docs/brief.md` first.
The layout is already written: `game/world_layout.py` (RO section at the bottom) and `game/data/terrain.py`.
Read both before anything else. Nobody edits `game/config.py`, `game/state.py`, `game/step.py`,
`game/movement.py`, `game/monsters.py`, `bots/runner.py`, `bots/policies.py`: report the lines instead.

## Shared facts
- Python: `C:\Users\crp27\jaxspike\venv\Scripts\python.exe` (absolute path; this worktree has no venv).
  Run from the worktree root. Tests: `-m pytest tests/<your file> -q`; every test file under 30 s on CPU.
- Terrain ids 0..7 in `game/config.py`; 8 `T_WARP`, 9 `T_WAYBELL`, 10 `T_STATION`, 11 `T_CAVE` in
  `game/data/terrain.py`. None of 8..11 blocks (`map_gen.blocking` is unchanged). Only `T_WARP` teleports;
  every warp destination is a `T_STATION` tile, which never teleports.
- Layout API (`game/world_layout.py`): `TOWN_NAMES`, `TOWN_ROLES`, `FIELD_BANDS` (lo, hi fraction of MAP of
  walking gate distance, kind names), `DUNGEONS`, `DUNGEON_ROADS`, `dungeon_boxes() -> int32 [ND, 4]`,
  `dungeon_centers()`, `dungeon_pads() -> int32 [ND, 4, 2]` (entrance warp, inside station, exit warp,
  outside station), `WARPS`, `warp_tiles()`, `waybell_tiles() -> [NT, 2]`, `town_station_tiles() -> [NT, 2]`,
  `dungeon_index(pos) -> int32 [N]` (-1 outside), tile constants `DUNGEON_WALL_TILES`, `DUNGEON_PILLAR_*`,
  `WARP_PAD_TILES`, `WARP_GAP_TILES`, `N_WARPS = 16`.
- `map_gen.warp_table(terrain=None) -> (src int32 [N_WARPS, 2], dst int32 [N_WARPS, 2])`, padded with -1 rows.
  Already written.
- Pure JAX rules: no Python `if` on traced values, no loops over entities, fixed shapes, masks over
  `active`, randomness only through `next_key(world)`, no coordinate literal outside `world_layout.py`
  (tests may use literal tiles). `jax.jit` at least once per test file; vmap-compatible.
- Existing functions keep their behaviour and `tests/test_map.py` stays green: the RO map is a **new**
  function `make_terrain_ro`, not a change to `make_terrain`.

## Worker B: carving (`game/map_gen.py` additions, `tests/test_ro_map.py`)
Owns: `game/map_gen.py` (append only; do not change existing functions except `protection_zone`, see
below), `tests/test_ro_map.py`. May adjust the numeric values of `DUNGEON_PILLAR_*`, `WARP_PAD_TILES`,
`WARP_GAP_TILES` in `world_layout.py` if the geometry needs it (nothing else there).
1. `make_terrain_ro(key) -> int8 [MAP, MAP]`: `make_terrain(key)`, then in this order:
   a. dungeon roads (`DUNGEON_ROADS`): from the dungeon's entrance warp tile (`dungeon_pads()[k][0]`) to
      the town centre, same L-shape and `replaceable` rule as the town roads (so it opens a gate in the
      town wall and bridges water);
   b. an entrance plaza of `T_ROAD` covering the outside pad tiles (entrance warp and outside station,
      plus one tile of margin) so both are walkable and connected to the road;
   c. each dungeon: box from `dungeon_boxes()`; interior `T_CAVE`; ring `DUNGEON_WALL_TILES` thick of
      `T_ROCK` (overwrites everything, roads included: the only way in is the warp); rock pillars inside,
      `DUNGEON_PILLAR_SIZE` square every `DUNGEON_PILLAR_PERIOD` tiles at phase `DUNGEON_PILLAR_PHASE`
      from the centre, never on the inside pad tiles (inside station, exit warp, and their 4-neighbours);
   d. special tiles last: `T_WARP` on each warp src, `T_STATION` on each warp dst, `T_WAYBELL` on
      `waybell_tiles()`, `T_STATION` on `town_station_tiles()`.
   Towns, temples and the rock border stay exactly as in `make_terrain`.
2. `protection_zone(terrain)`: add `T_WAYBELL` and `T_STATION` tiles **inside a town box** to the pz set
   (dungeon-side stations and warps stay outside pz). Old terrains have no such tiles, so old tests hold.
3. `distance_fields_ro(terrain) -> int16 [NT, MAP, MAP]`: like `distance_fields`, but the relaxation also
   relaxes across warps: for each real warp row `d[src] = min(d[src], d[dst] + 1)` (a fixed gather over
   `N_WARPS`, with -1 rows masked by `jnp.where`, then `.at[].min`). This makes dungeon tiles reachable
   (through the exit warp) so bots can walk home. `distance_fields` itself is unchanged.
4. `tests/test_ro_map.py` (seeds 0..2 where cheap): every warp src is `T_WARP` and every dst is
   `T_STATION`; waybell and town stations are the right ids and pz; dungeon ring all `T_ROCK`, interior
   only `T_CAVE`/`T_ROCK`/`T_STATION`/`T_WARP`; interior connected by BFS from the inside station
   (including the exit warp tile); no walkable path from outside the ring to inside without warps
   (BFS from a temple never reaches a `T_CAVE` tile); every dungeon pad tile is walkable; town tiles equal
   between `make_terrain` and `make_terrain_ro` except the waybell, station and the new gates (compare
   inside town boxes after masking those); town 0 has at least 4 gates; `distance_fields_ro` equals
   `distance_fields` outside the dungeons' reach and is finite on every `T_CAVE` tile; the inside station
   has distance `d[outside station] + WARP_GAP_TILES + 1` (walk to the exit warp, warp, land); jit + vmap
   of `make_terrain_ro` over 2 keys.
Report: pytest summary line, and the exact lines for `bots/runner.py`:
`terrain = map_gen.make_terrain_ro(k_walls)` and `fields = jax.lax.map(map_gen.distance_fields_ro, ...)`.

## Worker C: warps phase (`game/ro_warps.py`, `tests/test_ro_warps.py`)
Owns those two files only.
- `apply_warps(world) -> world`: for every active, living player whose tile matches a row of
  `map_gen.warp_table(world["terrain"])` (`src`), move it to that row's `dst`. Match with a
  `[MAX_PLAYERS, N_WARPS]` equality over the fixed table and `jnp.argmax`/`any` (never a MAP x MAP scatter).
  A destination tile that holds a living player (post-move positions) or an active monster is not
  entered: the player stays on the warp tile and retries next tick. Two players bound for one
  destination (possible only if two warps share a dst): the lowest slot wins, the other stays. Use the
  sort-based occupancy style of `game/movement.py` (`_occupied` is importable; or write your own 20-line
  version), not a `[MAP, MAP]` table. Monsters never warp; dead or inactive players never warp.
  Owns `players.pos` only.
- Phase placement to report to the lead: `step.py`, right after `movement.move_players` and before
  `monsters.monster_ai`: `world = ro_warps.apply_warps(world)`.
- Tests: build worlds with `make_empty_world(key)` and set `terrain` from `map_gen.make_terrain_ro` or a
  hand-made terrain (the table does not read terrain, so any terrain works); a player on a warp src lands
  on its dst; a player next to a warp does not move; a monster on a warp src does not move; a dead player
  on a warp src does not move; a player warping onto a dst occupied by a living player or an active
  monster stays; a padded -1 row never matches (a player at any tile never goes to (-1, -1), and no player
  at a clipped tile matches a -1 row); lowest slot wins when two players target one dst (build a 2-row
  table by monkeypatching `warp_table` or by passing an optional `table=(src, dst)` keyword argument that
  defaults to `map_gen.warp_table`); `jax.jit(apply_warps)` and `jax.vmap` over 2 worlds give the same
  result as eager. Keep the file under 30 s.
Report: pytest summary line and the `step.py` line.

## Worker D: monster kinds and spawn bands (`game/data/monsters.py`, `game/ro_spawns.py`, tests)
Owns `game/data/monsters.py`, `game/ro_spawns.py`, `tests/test_ro_monsters.py`, `tests/test_ro_spawns.py`.
- `game/data/monsters.py`: plain tuples, one row per kind, 20 kinds, **ids fixed**: 0 Blobbin, 1 Grukk Brute
  (these two keep the legacy rat/orc slots so `MONSTER_KIND_RAT`/`MONSTER_KIND_ORC` worlds still work),
  then in rising level: Wrigglet, Moonbun, Emberblob, Vexblob, Bramblet, Fiddlehop, Capling, Pilferbug, Rotling, Snarlfang,
  Clackbone, Gallopeck, Sandgulper, Clackbone Guard, Grumblebear, Tombwrap, Gildscarab (MVP), Scorchstripe (MVP).
  Columns (parallel tuples plus a `MONSTERS` tuple of namedtuples): `name`, `hp`, `atk_min`, `atk_max`,
  `def_`, `element` (index into `ELEMENTS = ("Neutral","Water","Earth","Fire","Wind","Poison","Holy",
  "Shadow","Ghost","Undead")`), `level`, `base_exp`, `job_exp`, `aggro` (bool), `move_ticks` (ticks per
  step, 1 = every tick), `size` (0 small, 1 medium, 2 large), `mvp` (bool). Classic pre-renewal flavour
  scaled to this game: Blobbin hp 40 atk 1-3 (= today's rat), Grukk Brute hp 120 atk 6-9 (= today's orc);
  Scorchstripe and Gildscarab around hp 3000-5000, atk 60-120. Export `KIND_BY_NAME: dict`, `N_KINDS = 20`,
  `MONSTER_HP_TUPLE`, `MONSTER_DAMAGE_TUPLE` (atk_max per kind) as drop-in replacements for config's
  `MONSTER_HP`/`MONSTER_DAMAGE`, and jnp arrays `HP`, `ATK_MIN`, `ATK_MAX`, `DEF`, `ELEMENT`, `LEVEL`,
  `BASE_EXP`, `JOB_EXP`, `AGGRO`, `MOVE_TICKS`, `SIZE`, `MVP` (all `[N_KINDS]`, int32 / bool).
- `game/ro_spawns.py`:
  - `band_of(world, pos) -> int32 [n]`: 0..len(FIELD_BANDS)-1 from `world["dist_town"].min(axis=0)` at
    `pos` compared with `FIELD_BANDS` fractions times MAP, `DIST_UNREACHABLE` counts as the last band;
    index `field[:, x, y]` per position, never gather a whole field per entity.
  - `kinds_for(world, pos, key) -> int32 [n]`: uniform choice among the band's kinds for field tiles;
    for tiles inside a dungeon (`world_layout.dungeon_index(pos) >= 0`) a uniform choice among that
    dungeon's kinds; the lowest-index position inside each dungeon becomes that dungeon's MVP
    (`in_k & (cumsum(in_k) == 1)`). Band kind tables are fixed `[N_BANDS, MAX_KINDS_PER_BAND]` int32
    arrays padded by repeating the last kind, built at import from the names via `KIND_BY_NAME`.
  - `choose_kinds(world, key, n) -> int32 [n]`: `kinds_for(world, world["monsters"]["pos"][:n], key)`.
  - `spawn_plan(terrain, dist_town) -> int32 [MAP, MAP]`: the band id per tile (field bands 0..2,
    then 3 + dungeon index for dungeon tiles, -1 on blocked or pz tiles); a debug/minimap helper, jit-able.
- Tests: table shape and id order; `KIND_BY_NAME["Blobbin"] == 0`; every `FIELD_BANDS` and `DUNGEONS`
  kind name exists; on a world built from `map_gen.make_terrain_ro` + `map_gen.distance_fields_ro`
  (one world, seed 0; build it once at module level) positions near a gate get band-0 kinds, far ones
  band 2, every `T_CAVE` position gets that dungeon's kinds, exactly one MVP per dungeon with positions
  inside, and `kinds_for` under `jax.jit` matches eager. Under 30 s.
Report: pytest summary; the wiring line for `bots/runner.spawn_population`:
`kind = ro_spawns.kinds_for(world, mpos, k_kind)` (with `key` split once more) and the config lines
`MONSTER_HP = game.data.monsters.MONSTER_HP_TUPLE` / `MONSTER_DAMAGE = ...MONSTER_DAMAGE_TUPLE`.

## Worker E: scale bench and plan (`bench/world_scale.py`, `docs/world_scale.md`)
Owns those two files only. Read `docs/handoff.md` ("GPU batch", "Distance fields") first.
- `bench/world_scale.py`: times `bots.runner.make_world_batch(seed, batch)` on CPU for batch 1 and 4
  (block_until_ready, report first call with compile and a second call), times `map_gen.make_terrain`,
  `map_gen.distance_fields` and `map_gen.distance_fields_ro` separately on one terrain (if
  `distance_fields_ro` does not exist yet, skip it and say so), counts the relaxation iterations (run the
  relaxation by hand with a Python loop over `_shift_min`, or measure `d.max()` over reachable tiles,
  which equals the iteration count), and prints the sizes in MB of every world array (`dist_town`
  `[NT, MAP, MAP]` int16, `terrain`, `walls`, `pz`) per world, plus `psutil`/`resource`-free peak RSS via
  `tracemalloc` is not meaningful for jax: report `jax` live buffer bytes from
  `jax.live_arrays()` after each build. Argument `--map` is **not** possible (MAP is a constant):
  estimate 2048 by scaling (area x16, iterations x4 => about x64 for the relaxation, x16 for everything
  else) and say that it is an estimate.
- `docs/world_scale.md`: the measured numbers in a table; the 2048 estimate (memory per world:
  `dist_town` 5 x 2048 x 2048 x 2 B = 42 MB, terrain/walls/pz 4 + 4 + 4 MB, players/monsters tiny); the
  GPU batch ceiling implied by the 80 % cap (6.5 GB) at 512 and at 2048 with and without the per-world
  `MAP x MAP` tables named in the handoff; and a plan for 2048 in ranked order: (1) compute the fields
  per town with `lax.map` over the town axis inside the per-world `lax.map` (peak temporaries drop from
  NT x MAP x MAP int32 to 1 x MAP x MAP int32), (2) keep int16 storage but relax in int32 chunks, (3)
  cache fields per terrain seed on disk (`numpy.savez`, 42 MB per world) because the fields depend only
  on the terrain, (4) a coarse 4x4-cell field for far-away steering and the fine field only inside a
  band around towns. Say which one you would do first and why (numbers).
Report: the two files and the headline numbers.

## Hand-back format (every worker)
Files touched; pytest summary line (`N passed`); any deviation from this brief with the reason; the
lines the lead must add to lead-owned files (exact text).
