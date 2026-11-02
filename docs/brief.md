# Tile MMO rules engine: contract brief

## Goal
A Tibia-style single-floor tile world in JAX. Players move, attack monsters, die and respawn. Monsters chase, attack, die, drop loot, and respawn. Bots play the game in batches. Everything is jit-compiled and vmappable over worlds.

## Shared API (lead owns, do not edit)
- `game/config.py`: constants. Use these names. Do not hardcode numbers.
- `game/state.py`: world schema (see docstring), `make_empty_world(key)` (world also holds `terrain` int8 and `pz` bool, both `[MAP, MAP]`), `next_key(world) -> (sub_key, world)`, `empty_inputs()`.
- `game/spatial.py`: `build_grid(pos, active)`, `block_candidates(table, query_pos)`, `nearest_in_block(table, query_pos, targets_pos) -> (idx, dist)`. Only nearest within 3x3 cell block (about CELL=8 tiles). Use `BIG` as the "none" distance.
- `game/world_layout.py`: layout spec in map fractions; the only file allowed to contain coordinate literals.
- `game/step.py`: the tick. Phase order is fixed below.

## Phase order (one tick)
1. `movement.move_players(world, inputs) -> world`
2. `monsters.monster_ai(world) -> world`
3. `combat.combat(world, inputs) -> world`
4. `loot.loot(world) -> world`
5. `survival.survival(world, inputs) -> world`
6. tick counter +1 (lead, in step.py)

`monsters.died` is cleared by step.py at the start of the tick. Combat sets it on the tick a monster dies. Loot reads it.

## Rules every module must follow
- Pure JAX. No Python `if` on traced values, no Python loops over entities, no printing, no global mutation. Use `jnp.where`, `.at[].set/add(mode="drop")`, `jax.lax` as needed.
- Fixed shapes. Inactive slots are ignored: mask with `active`. Never assume a slot is in use.
- Every phase must be `jax.jit`-compatible and `jax.vmap`-compatible. Test with `jax.jit` at least once.
- Randomness only through `next_key(world)`. Never use `np.random` or Python `random`.
- Positions are int32 tile coordinates in `[0, MAP)`. Clip with `jnp.clip`.
- No coordinate literal outside `game/world_layout.py`; derive positions from `MAP` and the layout spec.
- Walls: `walls[x, y] == 1` is blocked. Index as `walls[pos[:,0], pos[:,1]]`.
- Deterministic tie-breaks: when two entities want the same tile, the lowest slot index wins. Use scatter-min or argmin, not race-prone writes.
- Only update the fields your phase owns (listed per module below). Return a world with the same keys and shapes.
- Keep it readable: short functions, a one-line docstring per public function, comments only where the logic is not obvious.

## Module ownership (one worker per module)

### `game/map_gen.py` (worker: map)
- `make_walls(key) -> int8 [MAP, MAP]`: random walls, border ring of walls, about 15% interior density, with a clear 8x8 square at tiles 16..23 on both axes (inclusive) so spawns are always walkable. Keep it simple.
- `walkable(walls, pos) -> bool [N]`: in bounds and not a wall.
- Tests: `tests/test_map.py`.

### `game/movement.py` (worker: movement)
- `move_players(world, inputs)`: for each active, living player, apply `DIR_DELTAS[inputs["move"]]` if the target is walkable and not occupied by another active player or active monster. Conflicts between players wanting the same tile: lowest slot wins. Players never move through monsters.
- Owns: `players.pos`.
- Tests: `tests/test_movement.py`.

### `game/monsters.py` (worker: monsters)
- `monster_ai(world)`:
  1. Respawn: dead monsters (`active == False`, `respawn_in > 0`) count down. At 0 they reset to `active=True`, `hp=MONSTER_HP[kind]`, `pos=spawn`, `attack_cd=0`.
  2. Chase: each active monster finds the nearest living active player with `nearest_in_block` (player grid built with `build_grid(players.pos, players.active & ~players.dead)`). If that player is within `MONSTER_AGGRO_RANGE` (manhattan), step one tile toward them (prefer the longer axis, fallback to the other axis if blocked). Otherwise wander: stay put (no random walk in v1).
  3. A monster does not step onto a walkable-blocked tile, an active player tile, or an active monster tile. Lowest monster index wins collisions.
- Owns: `monsters.pos`, `monsters.respawn_in`, `monsters.active`, `monsters.hp`, `monsters.attack_cd` (respawn resets only).
- Tests: `tests/test_monsters.py`.

### `game/combat.py` (worker: combat)
- `combat(world, inputs)`:
  1. Respawn players: `dead` players reset to `pos=spawn`, `hp=PLAYER_MAX_HP`, `dead=False`, `attack_cd=0`. This runs first.
  2. Player attacks: an active, living player with `inputs["attack"]` and `attack_cd == 0` hits the nearest active monster at manhattan distance exactly 1 (use a monster grid). Damage `PLAYER_ATTACK`. Sets `attack_cd = PLAYER_ATTACK_COOLDOWN`, otherwise `attack_cd` decrements toward 0. Credit `kills` to the player whose hit reduces a monster to 0 HP (lowest player index wins if several).
  3. Monster attacks: an active monster with `attack_cd == 0` adjacent (manhattan 1) to a living player hits the nearest such player for `MONSTER_DAMAGE[kind]`. Sets `attack_cd = MONSTER_ATTACK_COOLDOWN`, otherwise decrement.
  4. Monster deaths: monsters with `hp <= 0` and active become `active=False`, `respawn_in = MONSTER_RESPAWN_TICKS`, `died=True`.
  5. Player deaths: players with `hp <= 0` and not yet dead become `dead=True`, `deaths += 1`, `active` stays True (dead players still count as logged in, but do not move, attack, or get targeted).
- Owns: `players.hp`, `players.dead`, `players.attack_cd`, `players.kills`, `players.deaths`, `players.pos` (respawn only), `monsters.hp`, `monsters.active`, `monsters.respawn_in`, `monsters.died`, `monsters.attack_cd`.
- Note: step 2 and 3 use hp values from the start of the phase. Monster deaths from player damage are applied in step 4.
- Tests: `tests/test_combat.py`.

### `game/loot.py` (worker: loot)
- `loot(world)`:
  1. Drops from kills: for each monster with `died == True`, roll a drop from the kind's table (`ITEM_GOLD_COIN` 60%, `ITEM_MEAT` 30%, `ITEM_SWORD` 5%, nothing 5%). Use `next_key`. Place it at the monster's `pos` in the first free drop slot (`active == False`). If there are more drops than free slots, drop the extras and do nothing.
  2. Pickup: an active, living player standing on a drop tile takes the item if it has an empty inventory slot. The lowest drop index wins if several drops are on one tile. Deactivate the drop. Leave the drop if the inventory is full.
  3. Decay: `ttl -= 1`, deactivate at 0.
- Owns: `drops.*`, `players.inv`.
- Tests: `tests/test_loot.py`.

### `game/survival.py` (worker: survival)
- `survival(world, inputs)`: living players regain `REGEN_HP` every `REGEN_TICKS` ticks (capped at `PLAYER_MAX_HP`); a player with `inputs["eat"]` and meat in `inv` removes one `ITEM_MEAT` and heals `MEAT_HEAL`.
- Owns: `players.hp` (increments only; combat only subtracts) and `players.inv` (removals only; loot only adds).
- Tests: `tests/test_survival.py`.

### `bots/policies.py` (worker: bots)
- `bot_inputs(world, key) -> inputs`: vectorized over players. A living active player with a monster within manhattan 6 (use a monster grid when it needs nearest) moves toward it one step per tick, and attacks when adjacent. Without one, it moves in a random direction (uniform over the 5 options, so stay is included), 50% of ticks.
- Only reads world, returns `{"move": int32 [MAX_PLAYERS], "attack": bool [MAX_PLAYERS]}`.
- Tests: `tests/test_bots.py`.

### `bots/runner.py` and `game/persistence.py` (worker: runner)
- `make_world_batch(seed, batch) -> world` with every field batched: `vmap(make_empty_world)`, then `map.make_walls` and spawn players and monsters at random walkable positions. Spawns are done by `runner.spawn_population(world, key)` (writes the schema fields; uses walkable tiles).
- `run_games(worlds, ticks) -> (worlds, stats)` using `jax.lax.scan` over ticks and `vmap` over worlds. stats: per-world totals for kills, deaths, items picked up, and a final alive-player count.
- `persistence.save(world, path)` / `persistence.load(path)` using `numpy.savez` with flattened keys. Round-trip must be exact.
- Tests: `tests/test_runner.py`.

### `bench/bench_step.py` (worker: bench)
- Times `step` on CPU for batch sizes 1, 64, 256 with full population, prints ms/tick and ms/world. Takes an argument `--platform cpu|gpu` that just reports the backend (the GPU run happens in WSL separately).
- Owns no game files. Only `bench/`.

## Test conventions
- Tests use plain `pytest`, no fixtures files. Build small worlds with `make_empty_world` and set fields directly.
- Each test file runs fast (under 30 seconds on CPU).
- Run with `C:\Users\crp27\jaxspike\venv\Scripts\python.exe -m pytest tests/test_<module>.py -q` from the project folder.

## Done means
- Your module's tests pass.
- You report: files changed, the pytest output summary line, and any deviations from this brief with a reason.
