# Tile MMO rules engine: contract brief

## Goal
A Tibia-style single-floor tile world in JAX. Players move, attack monsters, die and respawn. Monsters chase, attack, die, drop loot, and respawn. Bots play the game in batches. Everything is jit-compiled and vmappable over worlds.

## Shared API (lead owns, do not edit)
- `game/config.py`: constants. Use these names. Do not hardcode numbers.
- `game/state.py`: world schema (see docstring), `make_empty_world(key)` (world also holds `terrain` int8 and `pz` bool, both `[MAP, MAP]`, and `dist_town` int16 `[NT, MAP, MAP]`: walking distance to each town's gate aprons), `next_key(world) -> (sub_key, world)`, `empty_inputs()`.
- Goal-directed movement uses distance fields, never pathfinding: step to the neighbour with the smallest (toward) or largest (away) `dist_town` value. Index `dist_town[home, x, y]` per entity; never gather `dist_town[home]` (that builds a `[N, MAP, MAP]` array).
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
- In `game/` and `bots/`, no coordinate literal outside `game/world_layout.py` (tests may use literal tile positions); derive positions from `MAP` and the layout spec.
- Walls: `walls[x, y] == 1` is blocked. Index as `walls[pos[:,0], pos[:,1]]`.
- Deterministic tie-breaks: when two entities want the same tile, the lowest slot index wins. Use scatter-min or argmin, not race-prone writes.
- Only update the fields your phase owns (listed per module below). Return a world with the same keys and shapes.
- Keep it readable: short functions, a one-line docstring per public function, comments only where the logic is not obvious.

## Module ownership (one worker per module)

### `game/map_gen.py` (worker: map)
- `make_terrain(key) -> int8 [MAP, MAP]`: grass base, rock border, noise lakes and forests, the towns and roads from `game/world_layout.py` (towns overwrite water and trees; roads bridge water, cut trees and open gates in town walls). `blocking(terrain) -> int8 walls`, `protection_zone(terrain) -> bool pz` (town interiors incl. gates), `make_walls(key) = blocking(make_terrain(key))`. No coordinate literals here; everything scales from `MAP`.
- `walkable(walls, pos) -> bool [N]`: in bounds and not a wall.
- `gate_aprons(terrain) -> bool [NT, MAP, MAP]` (walkable non-pz tiles one step outside each town's gates) and `distance_fields(terrain) -> int16 [NT, MAP, MAP]` (walking distance to those aprons by relaxation in a `lax.while_loop`; `DIST_UNREACHABLE` where no path exists). Called once per world at creation.
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
  4. Monsters never step onto a `pz` tile (towns are safe) and only target living players who are not on `pz`.
- Owns: `monsters.pos`, `monsters.respawn_in`, `monsters.active`, `monsters.hp`, `monsters.attack_cd` (respawn resets only).
- Tests: `tests/test_monsters.py`.

### `game/combat.py` (worker: combat)
- `combat(world, inputs)`:
  1. Respawn players: `dead` players reset to `pos=spawn`, `hp=PLAYER_MAX_HP`, `dead=False`, `attack_cd=0`, once the spawn tile holds no monster or living player (otherwise they wait). This runs first.
  2. Player attacks: an active, living player with `inputs["attack"]` and `attack_cd == 0` hits the nearest active monster at manhattan distance exactly 1 (use a monster grid). Damage `PLAYER_ATTACK`. Sets `attack_cd = PLAYER_ATTACK_COOLDOWN`, otherwise `attack_cd` decrements toward 0. Credit `kills` to the player whose hit reduces a monster to 0 HP (lowest player index wins if several).
  3. Monster attacks: an active monster with `attack_cd == 0` adjacent (manhattan 1) to a living player hits the nearest such player for `MONSTER_DAMAGE[kind]`. Sets `attack_cd = MONSTER_ATTACK_COOLDOWN`, otherwise decrement.
  4. Monster deaths: monsters with `hp <= 0` and active become `active=False`, `respawn_in = MONSTER_RESPAWN_TICKS`, `died=True`.
  5. Player deaths: players with `hp <= 0` and not yet dead become `dead=True`, `deaths += 1`, `active` stays True (dead players still count as logged in, but do not move, attack, or get targeted).
- Owns: `players.hp`, `players.dead`, `players.attack_cd`, `players.kills`, `players.deaths`, `players.pos` (respawn only), `monsters.hp`, `monsters.active`, `monsters.respawn_in`, `monsters.died`, `monsters.attack_cd`.
- A monster never hits a player standing on a `pz` tile; monster targeting ignores players on `pz`.
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
- `bot_inputs(world, key) -> inputs`: vectorized over players; stateless (every mode is derived from the world). Home town of slot `i` is `i % NT`; its field is `dist_town[i % NT]`. Priority, highest first: a bot below `HUNT_RETREAT_HP` walks toward its gate (smallest neighbour) and ignores monsters; a bot on `pz` below max HP stands still; a living bot off `pz` with a monster within manhattan 6 (monster grid) steps toward it and attacks when adjacent; a healthy bot on `pz` walks toward its gate (this leads out); a bot closer than `HUNT_RADIUS // 2` to its gate walks away (largest neighbour, never onto `pz`); beyond `2 * HUNT_RADIUS` it walks back; otherwise it follows a heading derived from (slot, tick // HEADING_TICKS), falling back to a random step when the heading is blocked. Inside town the fallback is the random walk (uniform over 5 options, 50% of ticks).
- Only reads world, returns `{"move": int32 [MAX_PLAYERS], "attack": bool [MAX_PLAYERS], "eat": bool [MAX_PLAYERS]}`; `eat` is set for living players below half HP holding meat.
- Tests: `tests/test_bots.py`.

### `bots/runner.py` and `game/persistence.py` (worker: runner)
- `make_world_batch(seed, batch) -> world` with every field batched: `vmap(make_empty_world)`, then `map_gen.make_terrain` (plus `blocking` and `protection_zone`), `dist_town` from `map_gen.distance_fields` via `lax.map` (sequential over worlds), and `runner.spawn_population(world, key)`: player `i` spawns on tile `i // len(TOWNS)` of the temple of town `i % len(TOWNS)`; `POP_MONSTERS` (1024) monsters spawn on distinct walkable non-PZ tiles more than `GATE_APRON_TILES` walking distance from any gate, rats within `RAT_RING * MAP` walking distance of a gate and orcs beyond.
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

## Networking (net/)
Milestone 2. Plain Python/numpy around the JAX tick; `game/` is untouched except for constants in `game/config.py` (`TICK_MS`, `VIEW_W`, `VIEW_H`, `PATH_RADIUS`, `PATH_STUCK_TICKS`, `SAVE_EVERY_TICKS`, `PROTOCOL_VERSION`). Spec: `docs/superpowers/specs/2026-11-02-networked-server-design.md`.

### Modules and public functions
- `net/characters.py`: `valid_name(name) -> str | None` (lower-cased `[a-z0-9_]{3,16}`); `new_character(town, tile) -> dict`; `save(chars_dir, name, fields)`; `load(chars_dir, name) -> dict | None` (None when missing or corrupt); `load_roster(chars_dir) -> dict[str, int]`; `save_roster(chars_dir, roster)`; `next_town(roster) -> int` (fewest names, lowest index on ties). Files: `<name>.npz` and `roster.json`.
- `net/pathing.py`: `find_path(walls, start, goal, max_radius=PATH_RADIUS) -> list[int] | None`: A* on the 4-neighbour grid, move codes 1 N, 2 S, 3 W, 4 E; `[]` when already there; None when unreachable, goal is a wall or off-map, or goal is outside `start +- max_radius`.
- `net/protocol.py`: `encode(msg) -> str`; `decode(text) -> dict | None`; `validate_client(msg) -> dict | None`; `view_window(pos) -> (x0, y0)`; `terrain_window(terrain, x0, y0) -> list[list[int]]` (`T_ROCK` outside the map); `host_view(world) -> dict` (one numpy pull per tick); `snapshot(hv, slot, names, send_terrain, notice, tick) -> dict`.
- `net/server.py`: `GameServer(seed, chars_dir, tick_ms=TICK_MS, max_clients=MAX_PLAYERS)` with `await start(host, port)` (`port=0` picks a free port, read from `.port`), `await stop()`, `login(name) -> (slot, error)`, `logout(slot)`, `tick_once()`, attributes `owner`, `tick`, `world`, `overruns`. CLI: `venv\Scripts\python.exe -m net.server [--host 0.0.0.0 --port 8765 --seed 0 --chars characters --tick-ms 120]`.
- `net/client.py`: `NetClient(host, port, name)` (daemon thread): `start()`, `welcome`, `error`, `connected`, `latest()` (newest `snap`), `send_input(move, attack, eat)`, `walk_to(x, y)`, `close()`. `play.py` is a thin tkinter front end over it (`--host --port --name`).
- `bench/server_load.py`: N headless `NetClient`s sending random inputs; `--embedded` runs a server in-process.

### Protocol (WebSocket, JSON; every message has `"v": 1` and `"t"`)
| t | direction | fields | notes |
|---|---|---|---|
| `login` | C->S | `name` | first message, once; reply `welcome` or `error` + close |
| `input` | C->S | `move` 0..4, `attack`, `eat` | sent on change; server keeps the latest per slot; cancels a `walk_to` path |
| `walk_to` | C->S | `x`, `y` | click-to-walk (server A*); replaces any path; unreachable or too far -> `notice` |
| `welcome` | S->C | `slot`, `map`, `tick_ms`, `view_w`, `view_h` | view size fixed per connection |
| `snap` | S->C | `tick`, `you`, `ents`, `terrain`, `notice` | every tick; `ents` limited to the view window; `terrain` only when the window moved, else null; players appear by name, never slot |
| `error` | S->C | `msg` | then the socket closes |

Unknown `t` is ignored; malformed JSON, wrong version, or any message before `login` gets `error` and a close. `you` = `{pos, hp, max_hp, kills, deaths, inv, town, safe}`. Entities: `{kind:"p", name, pos, hp, max_hp}`, `{kind:"m", mkind, pos, hp, max_hp}`, `{kind:"d", item, pos}`.

### Rules
- The tick never awaits a client: at most one in-flight snapshot send per client; a slow client skips ticks.
- Login runs synchronously on the event loop (no await between slot pick and mark), so two humans never get the same slot.
- Logout saves the character and re-seeds the slot as a bot on its town's temple.
