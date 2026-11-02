# World + Survival Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 5-town procedural map at `MAP = 512` with safe towns, zoned monsters, home-town respawns, passive HP regen and eating meat, on the existing JAX tick.

**Architecture:** The world gains a `terrain` layer and a `pz` (protection zone) mask; `walls` stays as the derived blocking mask so the existing phases keep working. A scale-free layout spec (map fractions) drives a pure-JAX generator. A new `survival` phase runs after loot. The runner spawns players on their home temple and monsters off-PZ by zone. Protection-zone rules live in `monsters` (no entry) and `combat` (no hit into PZ).

**Tech Stack:** Python 3.15, JAX 0.12 (CPU venv; CUDA in WSL `~/jaxgpu`), numpy, pytest, tkinter.

**Spec:** `docs/superpowers/specs/2026-11-02-world-survival-design.md`

**No git:** the folder is not a repository. Where a plan would commit, run the full suite instead: `venv\Scripts\python.exe -m pytest tests -q` must stay green at the end of every task.

## Global Constraints

- Pure JAX in `game/` and `bots/`: no Python `if` on traced values, no Python loops over entities or tiles; fixed loops over the static layout (5 towns, 8 roads) are allowed.
- Every phase and generator must run under `jax.jit` and `jax.vmap`.
- `config.MAP = 512`. No coordinate literal anywhere except `game/world_layout.py`. Changing `MAP` to 2048 must need no code edit.
- Terrain ids (config): `T_GRASS=0, T_ROAD=1, T_FLOOR=2, T_TEMPLE=3, T_WATER=4, T_TREE=5, T_WALL=6, T_ROCK=7`. Blocking set: `{T_WATER, T_TREE, T_WALL, T_ROCK}`.
- Survival defaults (config): `REGEN_HP = 1`, `REGEN_TICKS = 4`, `MEAT_HEAL = 40`.
- Layout (world_layout): `TOWNS` 5 entries `(cx, cy, r)` in map fractions as in the spec, `ROADS` 8 pairs, `TEMPLE_TILES = 12`, `RAT_RING = 0.08`; town radius floor `TEMPLE_TILES // 2 + 4` tiles.
- `K_CELL`, `MAX_PLAYERS`, `MAX_MONSTERS`, `MAX_DROPS`, `POP_PLAYERS`, `POP_MONSTERS` unchanged.
- Ownership: `survival` writes `players.hp` (combat only subtracts) and `players.inv` (loot only adds); recorded in `docs/brief.md`.
- Run the CPU soak after integration; GPU runs only in WSL with `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.

## Review Focus

1. A road leg running along (not across) a town's wall ring would erase that whole side and make the town open; expected: every wall-ring side keeps most of its WALL tiles. Pinned in Task 2 (`test_wall_rings_mostly_intact`).
2. `MAX_PLAYERS` raised above the temple capacity (`len(TOWNS) * TEMPLE_TILES**2 = 720`) would give two players one spawn tile; expected: import fails loudly. Pinned in Task 2 (`test_temple_capacity_covers_max_players`) and an `assert` in `world_layout.py`.
3. Eating with full HP: the spec says the meat is still consumed; a player who mashes E loses food. Pinned in Task 3 (`test_eat_at_full_hp_consumes_meat`) so the behaviour is deliberate and visible.
4. A player standing in a gate (ROAD inside the town box) must count as safe, and one on the ROAD just outside must not. Pinned in Task 2 (`test_pz_covers_gate_not_outside_road`) and Task 4 (`test_monster_cannot_hit_player_on_pz`).
5. Saving and loading a world with the new fields must keep `pz` as bool and `terrain` as int8. Pinned in Task 1 (`test_persistence_roundtrip_exact` extended).

---

### Task 1: Schema, config, inputs and brief (lead)

**Files:**
- Modify: `game/config.py`, `game/state.py`, `docs/brief.md`
- Test: `tests/test_runner.py` (persistence test), `tests/test_state.py` (new)

**Interfaces:**
- Produces: `world["terrain"]: int8 [MAP, MAP]`, `world["pz"]: bool [MAP, MAP]`; `empty_inputs()` returns `{"move", "attack", "eat": bool [MAX_PLAYERS]}`; config names `T_*`, `REGEN_HP`, `REGEN_TICKS`, `MEAT_HEAL`, `MAP = 512`.

- [ ] **Step 1: Write the failing tests** in `tests/test_state.py`:

```python
def test_empty_world_has_terrain_and_pz():
    w = make_empty_world(jax.random.PRNGKey(0))
    assert w["terrain"].shape == (MAP, MAP) and w["terrain"].dtype == jnp.int8
    assert w["pz"].shape == (MAP, MAP) and w["pz"].dtype == jnp.bool_
    assert int(w["terrain"].max()) == T_GRASS and not bool(w["pz"].any())

def test_empty_inputs_has_eat():
    i = empty_inputs()
    assert i["eat"].shape == (MAX_PLAYERS,) and i["eat"].dtype == jnp.bool_ and not bool(i["eat"].any())
```
  And in `tests/test_runner.py::test_persistence_roundtrip_exact` add before saving:
  `w["terrain"] = w["terrain"].at[3, 4].set(T_WATER)` and `w["pz"] = w["pz"].at[5, 6].set(True)`.

- [ ] **Step 2: Run** `venv\Scripts\python.exe -m pytest tests/test_state.py tests/test_runner.py -q` — expected: FAIL (KeyError `terrain`, ImportError `T_GRASS`).

- [ ] **Step 3: Edit `game/config.py`**: `MAP = 512`; add the `T_*` ids and `REGEN_HP`, `REGEN_TICKS`, `MEAT_HEAL` with the values in Global Constraints; one comment line on the blocking set.

- [ ] **Step 4: Edit `game/state.py`**: add `"terrain": jnp.full((MAP, MAP), T_GRASS, jnp.int8)` and `"pz": jnp.zeros((MAP, MAP), jnp.bool_)` to `make_empty_world`; add `"eat": jnp.zeros((MAX_PLAYERS,), jnp.bool_)` to `empty_inputs`; document both fields and the input in the module docstring.

- [ ] **Step 5: Run** the two test files — expected: PASS. Then run the full suite — expected: 64 passed (62 + 2). `test_map.py` still passes against the old `make_walls` at MAP=512 (its literals are inside the map).

- [ ] **Step 6: Edit `docs/brief.md`**: add `terrain`/`pz` to the shared API line for `state.py`; add phase 5 `survival.survival(world, inputs)` before the tick counter; add the module section "game/survival.py (worker: survival)" with ownership of `players.hp` (+) and `players.inv` (−); add the layout rule "no coordinate literal outside game/world_layout.py"; add "game/world_layout.py" to the lead-owned list.

### Task 2: Layout spec and terrain generation (worker: map)

**Files:**
- Create: `game/world_layout.py`
- Modify: `game/map_gen.py`
- Test: `tests/test_map.py` (rewrite)

**Interfaces:**
- Consumes: config `MAP`, `T_*`, `MAX_PLAYERS`.
- Produces, in `game/world_layout.py`:
  - constants `TOWNS`, `ROADS`, `TEMPLE_TILES`, `RAT_RING`, `WATER_THRESHOLD = 0.78`, `FOREST_THRESHOLD = 0.62`, `TREE_DENSITY = 0.6`, `NOISE_CELLS = 32` (noise grid is `MAP // NOISE_CELLS` per side), `ROAD_HALF_WIDTH = 1`.
  - `town_boxes() -> np.ndarray int32 [NT, 4]` as `(x0, y0, x1, y1)` inclusive bounds of each wall ring (plain numpy, computed from `MAP`).
  - `temple_tiles() -> jnp int32 [NT, TEMPLE_TILES * TEMPLE_TILES, 2]` row-major tiles of each temple square.
  - `dist_to_nearest_town(pos: int32 [N, 2]) -> int32 [N]` Chebyshev distance from each position to the nearest town box (0 inside). JAX, fixed loop over towns.
  - module-level `assert len(TOWNS) * TEMPLE_TILES ** 2 >= MAX_PLAYERS`.
- Produces, in `game/map_gen.py`:
  - `make_terrain(key) -> int8 [MAP, MAP]`
  - `blocking(terrain) -> int8 [MAP, MAP]` (1 where terrain in the blocking set)
  - `protection_zone(terrain) -> bool [MAP, MAP]` = FLOOR | TEMPLE | (ROAD inside any town box)
  - `make_walls(key)` = `blocking(make_terrain(key))` (kept for existing call sites)
  - `walkable(walls, pos)` unchanged.

- [ ] **Step 1: Replace `tests/test_map.py`** with tests (numpy on `np.asarray(make_terrain(PRNGKey(s)))`, `s` in 0..2 where cheap):

```python
def test_border_is_rock(): all four edges == T_ROCK
def test_towns_are_floor_with_wall_ring_and_temple():
    for each box in town_boxes(): interior tiles are in {T_FLOOR, T_TEMPLE, T_ROAD};
    the four ring sides contain only {T_WALL, T_ROAD}; the TEMPLE_TILES square centred in the box is all T_TEMPLE
def test_no_water_or_trees_in_towns(): no T_WATER/T_TREE inside any box
def test_each_town_has_two_gates(): per town, count of T_ROAD on the ring >= 2 * (2 * ROAD_HALF_WIDTH + 1)
def test_wall_rings_mostly_intact(): per town per side, fraction of T_WALL >= 0.5
def test_temples_connected():
    BFS over blocking(terrain) == 0 from temple_tiles()[0][0]; every temple's first tile is reached
def test_pz_covers_gate_not_outside_road():
    pz = protection_zone(terrain); every T_ROAD tile inside a box is pz; every T_ROAD tile with
    dist_to_nearest_town > 0 is not pz; all T_FLOOR/T_TEMPLE are pz; no T_GRASS is pz
def test_wilderness_has_water_and_trees(): 0.02 < mean(terrain == T_WATER) < 0.25 and 0.03 < mean(terrain == T_TREE) < 0.30 on the tiles outside all boxes
def test_towns_fixed_across_keys(): terrain(k7) and terrain(k8) agree on every tile inside a box; differ somewhere outside
def test_deterministic(): terrain(k7) == terrain(k7)
def test_jit_and_vmap(): jax.jit(make_terrain) equals eager; vmap over 3 keys has shape (3, MAP, MAP)
def test_dist_to_nearest_town(): a temple tile -> 0; the tile one past a box's x1 -> 1; a corner far away -> matches numpy max(|dx|,|dy|) to the nearest box
def test_temple_capacity_covers_max_players(): len(TOWNS) * TEMPLE_TILES**2 >= MAX_PLAYERS
def test_walkable_walls_and_oob(): keep the existing test, using make_walls
```

- [ ] **Step 2: Run** `venv\Scripts\python.exe -m pytest tests/test_map.py -q` — expected: FAIL (ImportError `world_layout`, `make_terrain`).

- [ ] **Step 3: Write `game/world_layout.py`** with the constants and the three functions above. `town_boxes` uses `r = max(round(rf * MAP), TEMPLE_TILES // 2 + 4)` and `c = round(cf * MAP)`; box = `(c - r, c - r, c + r, c + r)`. `temple_tiles` centres a `TEMPLE_TILES` square on each box's centre.

- [ ] **Step 4: Write `make_terrain` in `game/map_gen.py`** following the spec's five steps, as per-tile masks over `xs, ys = jnp.meshgrid(jnp.arange(MAP), jnp.arange(MAP), indexing="ij")`:
  1. `T_GRASS` everywhere, `T_ROCK` on the border.
  2. Two noise fields: `jax.random.uniform(k, (MAP // NOISE_CELLS,) * 2)` → `jax.image.resize(..., (MAP, MAP), "bilinear")`; water where field1 > `WATER_THRESHOLD`; tree where field2 > `FOREST_THRESHOLD` and `jax.random.bernoulli(k3, TREE_DENSITY, (MAP, MAP))`.
  3. For each town box: interior → `T_FLOOR`; ring (tiles on the box edge) → `T_WALL`; temple square → `T_TEMPLE`.
  4. For each road `(a, b)`: L path from centre a to centre b, horizontal leg at `y = ay` for `x` between `ax` and `bx`, vertical leg at `x = bx` for `y` between `ay` and `by`, widened by `ROAD_HALF_WIDTH`; tiles on the path whose terrain is in `{T_GRASS, T_WATER, T_TREE, T_WALL}` become `T_ROAD`.
  Add `blocking`, `protection_zone` (uses `town_boxes()` as static numpy bounds) and rewrite `make_walls`. Remove `WALL_DENSITY`, `SPAWN_LO`, `SPAWN_HI`.

- [ ] **Step 5: Run** `tests/test_map.py` — expected: PASS, under 30 s. If `test_temples_connected` fails, the usual cause is a leg whose corner lands on a wall corner tile; widen the corner by stamping the vertical leg from `min(ay, by) - ROAD_HALF_WIDTH`.

- [ ] **Step 6: Run the full suite** — expected: all green (the runner still calls `make_walls`).

### Task 3: Survival phase and bot eating (worker: survival)

**Files:**
- Create: `game/survival.py`
- Modify: `game/step.py` (add phase 5), `bots/policies.py`
- Test: `tests/test_survival.py` (new), `tests/test_bots.py` (add two tests)

**Interfaces:**
- Consumes: config `REGEN_HP`, `REGEN_TICKS`, `MEAT_HEAL`, `PLAYER_MAX_HP`, `ITEM_MEAT`, `ITEM_NONE`; `inputs["eat"]`; `world["tick"]`.
- Produces: `survival(world, inputs) -> world`; `bot_inputs` returns `{"move", "attack", "eat"}`.

- [ ] **Step 1: Write `tests/test_survival.py`** (build with `make_empty_world`, activate player 0, set hp and inv directly):

```python
def test_regen_on_multiple_of_regen_ticks(): tick = REGEN_TICKS, hp 100 -> 100 + REGEN_HP
def test_no_regen_off_schedule(): tick = REGEN_TICKS + 1, hp 100 -> 100
def test_regen_capped_at_max(): hp PLAYER_MAX_HP -> PLAYER_MAX_HP
def test_dead_and_inactive_do_not_regen(): dead player and inactive slot keep hp
def test_eat_removes_one_meat_and_heals(): inv [MEAT, MEAT, NONE...], hp 50, eat -> hp 50 + MEAT_HEAL, inv has exactly one MEAT left in slot 1, slot 0 is NONE
def test_eat_heals_capped(): hp PLAYER_MAX_HP - 10 -> PLAYER_MAX_HP
def test_eat_at_full_hp_consumes_meat(): hp max, one meat, eat -> hp max, inv empty
def test_eat_without_meat_noop(): inv all NONE, hp 50 -> 50
def test_dead_player_cannot_eat(): dead, meat, eat -> inv unchanged
def test_jit_and_vmap(): jax.jit(survival) matches eager; vmap over 2 worlds gives shape [2, MAX_PLAYERS]
```

- [ ] **Step 2: Run** `venv\Scripts\python.exe -m pytest tests/test_survival.py -q` — expected: FAIL (no module `game.survival`).

- [ ] **Step 3: Write `game/survival.py`** with `survival(world, inputs)`: `alive = active & ~dead`; `regen = alive & (world["tick"] % REGEN_TICKS == 0)`; find the lowest meat slot with `jnp.argmax(inv == ITEM_MEAT, axis=1)` guarded by `jnp.any(inv == ITEM_MEAT, axis=1)`; `eats = alive & inputs["eat"] & has_meat`; clear that slot where `eats`; `hp = jnp.minimum(hp + regen * REGEN_HP + eats * MEAT_HEAL, PLAYER_MAX_HP)`. Return the world with `players.hp` and `players.inv` replaced.

- [ ] **Step 4: Run** `tests/test_survival.py` — expected: PASS.

- [ ] **Step 5: Add phase 5 to `game/step.py`**: `world = survival.survival(world, inputs)` after `loot.loot(world)`, before the tick increment; update the docstring list.

- [ ] **Step 6: Add to `tests/test_bots.py`**:

```python
def test_bot_eats_below_half_hp(): hp PLAYER_MAX_HP // 2 - 1 with a meat -> eat True; hp PLAYER_MAX_HP // 2 -> False
def test_bot_eat_needs_meat_and_life(): no meat -> False; dead with meat -> False
```
  Run them — expected: FAIL (KeyError `eat`).

- [ ] **Step 7: Edit `bots/policies.py`**: `has_meat = jnp.any(pl["inv"] == ITEM_MEAT, axis=1)`; `eat = live & (pl["hp"] < PLAYER_MAX_HP // 2) & has_meat`; return it in the dict; update the docstring.

- [ ] **Step 8: Run the full suite** — expected: green, 62 + 2 (Task 1) + 12 (this task) + the Task 2 count.

### Task 4: Spawns, zones and protection-zone rules (worker: zones)

**Files:**
- Modify: `bots/runner.py`, `game/monsters.py`, `game/combat.py`
- Test: `tests/test_runner.py`, `tests/test_monsters.py`, `tests/test_combat.py`

**Interfaces:**
- Consumes: `world_layout.temple_tiles()`, `world_layout.dist_to_nearest_town(pos)`, `world_layout.RAT_RING`, `map_gen.make_terrain`, `map_gen.blocking`, `map_gen.protection_zone`, `world["pz"]`, config `MONSTER_KIND_RAT`, `MONSTER_KIND_ORC`.
- Produces: `runner.make_world_batch(seed, batch)` fills `terrain`, `walls`, `pz`; `runner.spawn_population(world, key)` as before but with home-town spawns and zoned kinds.

- [ ] **Step 1: Add tests in `tests/test_runner.py`** (use `make_world_batch(0, 1)` and unbatch with `tree_map(lambda x: x[0])`):

```python
def test_players_spawn_on_home_temple():
    for active player i: spawn == temple_tiles()[i % len(TOWNS)][i // len(TOWNS)] and pos == spawn; all spawns distinct
def test_monsters_spawn_off_pz_on_walkable(): pos of active monsters: walls == 0 and pz False
def test_monster_kind_follows_zone():
    d = dist_to_nearest_town(pos); kind == RAT where d < RAT_RING * MAP else ORC; both kinds present
def test_world_batch_has_terrain_walls_pz(): shapes (B, MAP, MAP); walls == blocking(terrain); pz == protection_zone(terrain)
```
  Keep `test_spawn_population_counts`, but build its world with `make_terrain` + `blocking` + `protection_zone` instead of hand-set walls.

- [ ] **Step 2: Run** `tests/test_runner.py` — expected: FAIL (`pz` all false, players not on temples).

- [ ] **Step 3: Edit `bots/runner.py`**: rename `_distinct_walkable(walls, key, n)` to `_distinct_tiles(allowed, key, n)` taking a bool `[MAP, MAP]` mask; in `spawn_population`: player spawns from `temple_tiles()` as `tiles[i % NT, i // NT]` for `i < POP_PLAYERS` (build with `jnp.arange`), monsters from `_distinct_tiles((walls == 0) & ~pz, kk, POP_MONSTERS)`, kind = `jnp.where(dist_to_nearest_town(mpos) < RAT_RING * MAP, MONSTER_KIND_RAT, MONSTER_KIND_ORC)`; in `make_world_batch.one`: `terrain = make_terrain(k_walls)`, set `terrain`, `walls = blocking(terrain)`, `pz = protection_zone(terrain)`.

- [ ] **Step 4: Run** `tests/test_runner.py` — expected: PASS.

- [ ] **Step 5: Add tests** in `tests/test_monsters.py` (its `build` helper gains a `pz=[(x, y), ...]` parameter that sets `world["pz"]`):

```python
def test_monster_does_not_enter_pz(): player (20,20), monster (24,20), pz tile (23,20) -> monster stays at (24,20) (the only step toward the player is into the PZ; dy == 0 so there is no fallback axis)
def test_monster_steps_around_pz_tile(): player (20,20), monster (24,21), pz (23,21) -> monster steps to (24,20) (same shape as the wall test)
```
  and in `tests/test_combat.py`:

```python
def test_monster_cannot_hit_player_on_pz(): player at (20,20) with pz[20,20] True, orc adjacent, cd 0 -> player hp unchanged, monster attack_cd unchanged (no hit happened)
def test_player_on_pz_can_still_attack(): same setup with attack input -> monster hp reduced by PLAYER_ATTACK
```

- [ ] **Step 6: Run** them — expected: FAIL.

- [ ] **Step 7: Edit `game/monsters.py::_occupancy`**: `occ = (world["walls"] != 0) | world["pz"]` as the base. Edit `game/combat.py`: after computing `mhit`, `mhit = mhit & ~world["pz"][p["pos"][ptgt, 0], p["pos"][ptgt, 1]]`.

- [ ] **Step 8: Run the full suite** — expected: green.

### Task 5: play.py terrain, HUD and eat key (worker: play, after Tasks 1-4)

**Files:**
- Modify: `play.py`

**Interfaces:**
- Consumes: `world["terrain"]`, `world["pz"]`, config `T_*`, `inputs["eat"]` from `bot_inputs`.

- [ ] **Step 1: Add a `TERRAIN_FILL` dict** `{T_GRASS: None (keep draw_grass), T_ROAD: "#b89b6a", T_FLOOR: "#8a7a66", T_TEMPLE: "#e6d28a", T_WATER: "#2f6fb3", T_TREE: draw dark-green circle "#1f4d1a" on grass, T_WALL: draw_wall, T_ROCK: "#3a3a3a"}`; in `draw`, branch on `terrain[x, y]` instead of `walls[x, y]`.

- [ ] **Step 2: HUD**: append `town {HUMAN % len(TOWNS)}` and `SAFE` when `pz[hx, hy]`; add `E eat` to the controls line.

- [ ] **Step 3: Eat key**: bind `<e>`/`<KeyRelease-e>` to set `self.eating`; in `run_tick` copy `inputs["eat"]` to numpy, set `eat[HUMAN] = self.eating`, pass it in the inputs dict.

- [ ] **Step 4: Verify** with the headless check used before: instantiate `Game`, `run_tick()` three times, `draw()`; expected: no traceback, printed frame time. Smoke-launch for 20 s: no traceback. Tell the user the window has not been seen.

### Task 6: Integration, soak invariants, GPU ceiling, handoff (lead)

**Files:**
- Modify: `bench/soak_check.py`, `docs/handoff.md`, `docs/brief.md` (phase list check)

- [ ] **Step 1: Add soak invariants** in `check_invariants`: "active monster on pz tile"; "player spawn not on temple" (spawn tile's terrain == T_TEMPLE); keep the others.

- [ ] **Step 2: Run** `venv\Scripts\python.exe bench/soak_check.py` and `--seed 1` — expected: `all invariants held`; record kills, deaths, items at tick 1000 and the rat/orc counts (add a one-line print of kind counts at start).

- [ ] **Step 3: GPU ceiling** in WSL under the cap: `--batch 64`, then 128, 256, 512, stop at the first `RESOURCE_EXHAUSTED`; `nvidia-smi` before and after each. Expected: clean through at least 128.

- [ ] **Step 4: Full suite** — expected: green; note the final count.

- [ ] **Step 5: Update `docs/handoff.md`**: new state line (fields, phases, map), GPU table for MAP=512, balance numbers for the new map (not tuned), open work (Tiled import, K_CELL for towns, balance after regen, milestone 2).
