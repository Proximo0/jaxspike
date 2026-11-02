# M4 gathering and crafting: worker contract (ro-craft team, 2026-11-02)

Read `CLAUDE.md`, `docs/ro_contract.md`, `docs/brief.md` ("Rules every module must follow") and this file. Tables and
ids live in `game/data/craft.py` (already written, import from it, do not redefine numbers). Python:
`C:\Users\crp27\jaxspike\venv\Scripts\python.exe` (absolute path, the worktree has no venv). CPU only.
Never edit `game/config.py`, `game/state.py`, `game/step.py`, `game/map_gen.py`, `bots/runner.py`,
`bots/policies.py`, `bench/soak_check.py`: if you need a change there, write the exact lines in your report.

## Pure-JAX rules (every function below)
jit- and vmap-compatible; fixed shapes; no Python `if` on traced values; no Python loop over entities (loops over
constant tables such as the 8 node kinds or the 4 recipe inputs are fine); masks over `active`; randomness only via
`game.state.next_key(world)`; `.at[].set/min/add(mode="drop")` with out-of-range indices for "no write";
lowest slot wins ties (scatter-min); **never build a `[MAX_PLAYERS, MAP, MAP]` or `[MAX_PLAYERS, MAX_NODES]`-shaped
array where a tile map `[MAP*MAP]` + gather does the job** (see `game/loot.py::_pickup` for the pattern:
scatter the lowest entity index per tile, then look up the player's tiles). Index fields as `field[kind, x, y]`,
never `field[kind]` per player. Coordinates come from `MAP`, `game.world_layout` and `game/data/craft.py` only.

## State block: `world["craft"]` (built by `ro_craft.make_state()`)
```
"craft": {
  "nodes": {
    "pos":        int32 [MAX_NODES, 2]   tile of the node (players may stand on it; nodes never block)
    "kind":       int32 [MAX_NODES]      node kind 0..7 (game/data/craft.py NODE_*), 0 in unused slots
    "placed":     bool  [MAX_NODES]      slot holds a node (place_nodes sets it; stations and nodes alike)
    "active":     bool  [MAX_NODES]      harvestable now (stations: always True while placed)
    "respawn_in": int32 [MAX_NODES]      >0 while gathered-out, counts down 1 per tick, reactivates at 0
  },
  "dist_node":  int16 [N_NODE_KINDS, MAP, MAP]  walking distance to the nearest placed node of each kind
                                                (node_fields; zeros until the lead / with_craft fills it)
  "level":      int32 [MAX_PLAYERS, N_CRAFT_SKILLS]  starts at 1
  "exp":        int32 [MAX_PLAYERS, N_CRAFT_SKILLS]  starts at 0; level = count(exp >= LEVEL_EXP_A)
  "gathers":    int32 [MAX_PLAYERS]  successful gathers (soak counter)
  "crafts":     int32 [MAX_PLAYERS]  successful crafts (soak counter)
}
```
Slots are contiguous per kind: kind k owns slots `NODE_SLOT_START[k] .. NODE_SLOT_START[k] + NODE_COUNT[k] - 1`;
`placed` is False where fewer eligible tiles existed. Node fields: `dist_node[kind, x, y]` is 0 on a placed node tile
of that kind, `DIST_UNREACHABLE` (from `game.map_gen`) where no walkable path exists. Fields are computed once at world
creation from `placed` nodes (respawn does not change them).

## Inputs (new keys; the lead adds them to `state.empty_inputs`)
- `inputs["gather"]`: bool `[MAX_PLAYERS]`, True = try to harvest one active gatherable node within `GATHER_RANGE`
  (manhattan 1: own tile or a 4-neighbour).
- `inputs["craft"]`: int32 `[MAX_PLAYERS]`, recipe id 1..N_RECIPES-1, 0 = none.
Tests build them by hand: `{**empty_inputs(), "gather": ..., "craft": ...}`.

## Worker B: `game/ro_craft_nodes.py` + `tests/test_ro_craft_nodes.py`
- `empty_nodes() -> dict` the zero `nodes` dict above (placed/active False).
- `node_at(nodes) -> int32 [MAP*MAP]`: lowest placed node index on each tile (`x * MAP + y`), `MAX_NODES` where none
  (scatter-min over placed nodes, like `loot._pickup`'s `tile_min`).
- `nodes_near(node_at, pos) -> int32 [N, 5]`: node index (or MAX_NODES) on each pos's own tile and 4 neighbours
  (DIR_DELTAS[1:], clipped to the map; a neighbour that clips onto itself is harmless since callers take a min).
- `station_tiles() -> int32 [NT, 2, 2]`: for every town (`world_layout.town_centers()`), the furnace tile
  `(cx + TEMPLE_TILES // 2 + STATION_GAP, cy)` and the cooking-fire tile `(cx - TEMPLE_TILES // 2 - STATION_GAP, cy)`.
  Both are town floor tiles (pz) in the current layout; assert that in the test with a real terrain.
- `place_nodes(world, key) -> world`: fills `world["craft"]["nodes"]` from `world["terrain"]`, `world["walls"]`,
  `world["pz"]`. Eligible tiles per gather kind, all walkable (`walls == 0`) and not pz:
  ore (copper, iron, coal): 4-neighbour of a `T_ROCK` tile; fish: 4-neighbour of `T_WATER` and not ore-eligible;
  tree: 4-neighbour of `T_TREE` and neither of the above; herb: `T_GRASS` and none of the above. (The sets are
  disjoint by this priority, so one tile never gets two nodes. Today rock exists only on the map rim, so ore sits at
  the rim; that is accepted and reported.) Copper, iron and coal split the ore set by ranking: the first
  `NODE_COUNT[0]` ore tiles in random order are copper, the next `NODE_COUNT[1]` iron, the next `NODE_COUNT[2]` coal.
  Use **one** random score per tile (`jax.random.uniform(key, (MAP*MAP,))`), one `argsort`, and per kind a cumsum
  rank over the kind's eligible tiles in that order (the `bots/runner._distinct_tiles` idea, extended): slot
  `NODE_SLOT_START[k] + r` gets the r-th eligible tile of kind k for `r < min(NODE_COUNT[k], n_eligible_k)`;
  `placed` True there; `active = placed`; `respawn_in = 0`. Stations: slots of kinds 6 and 7 get `station_tiles()`
  (one per town, `NODE_COUNT[6] == NODE_COUNT[7] == len(TOWNS)`), placed and active regardless of terrain.
  Leave every other key of `world["craft"]` untouched (so it also works on a world whose other arrays are filled).
- `node_fields(world) -> int16 [N_NODE_KINDS, MAP, MAP]`: source tiles = placed nodes of kind k (scatter True into a
  `[N_NODE_KINDS, MAP*MAP]` bool); relaxation `d = min(d, 1 + min(4-neighbours))` over walkable tiles
  (`world["walls"] == 0`; station tiles are walkable floor) in a `jax.lax.while_loop` until unchanged, exactly as
  `game.map_gen.distance_fields` does (copy its 4-line `_shift_min`; do not import private names). Unreachable tiles
  hold `DIST_UNREACHABLE`. Walkability does not treat node tiles as blocked. Store nothing; the caller assigns it to
  `world["craft"]["dist_node"]`.
- Tests (under 30 s): node counts per kind on a real `make_terrain` world (`placed` per kind equals
  `min(NODE_COUNT[k], eligible_k)`; at least 100 of each gather kind on seed 0); every placed gather node on a
  walkable non-pz tile with the right terrain neighbour; no two placed nodes share a tile; stations on pz floor;
  `node_at`/`nodes_near` on a hand-built case; field is 0 on node tiles, 1 on their walkable neighbours, unreachable
  on walls; `jax.jit(place_nodes)` and `jax.vmap` over 2 worlds give the same shapes; `node_fields` under `jax.jit`.

## Worker A: `game/ro_craft.py` + `tests/test_ro_craft.py`
Imports `game.ro_craft_nodes` (B's module, in progress: code against the signatures above; until it lands you may
write a throwaway local copy in your test file, delete it before hand-back). Re-export `place_nodes`, `node_fields`,
`node_at`, `nodes_near`, `station_tiles` so the lead calls `ro_craft.place_nodes` / `ro_craft.node_fields`.
- `make_state() -> dict`: the block above (level filled with 1, dist_node zeros int16).
- `with_craft(world, key) -> world`: `{**world, "craft": make_state()}`, then `place_nodes`, then
  `dist_node = node_fields(world)`. Unbatched convenience for tests and the soak harness.
- `has_room(inv) -> bool [N]` and `add_item(inv, item) -> inv`: `item` int32 `[N]`, `ITEM_NONE` (0) adds nothing;
  the item goes into the player's first empty slot (`inv == ITEM_NONE`); a full inventory is left unchanged (callers
  check `has_room` first). Works on `players["inv"]` ids only; no dependency on M3.
- `count_item(inv, item) -> int32 [N]` (how many slots equal `item[i]`), `remove_item(inv, item) -> inv` (clears the
  lowest slot equal to `item[i]`, `ITEM_NONE` = nothing; see `game/survival.py` for the pattern).
- `level_from_exp(exp) -> int32` same shape: `sum(exp[..., None] >= LEVEL_EXP_A, -1)` clipped to
  `[1, CRAFT_MAX_LEVEL]`.
- `adjacent_node(world) -> (idx int32 [MAX_PLAYERS], kind int32 [MAX_PLAYERS], ok bool [MAX_PLAYERS])`: the lowest-index
  **placed and active gatherable** node (kind < N_GATHER_KINDS) within GATHER_RANGE of each player via
  `node_at` + `nodes_near`; `ok` False (idx MAX_NODES, kind 0) when none. Exported for the bot policy.
- `adjacent_station(world, kind) -> bool [MAX_PLAYERS]`: a placed node of node kind `kind` (int32 `[MAX_PLAYERS]`)
  within GATHER_RANGE; `kind == -1` means "no station needed" and gives True.
- `has_inputs(world) -> bool [MAX_PLAYERS, N_RECIPES]`: for every recipe: living active player, `level[skill] >=
  RECIPE_LEVEL`, inventory holds every input with multiplicity (for input column j: `count_item(inv, in_j) >=
  number of columns k with in_k == in_j`, ignoring 0s). Column 0 (RECIPE_NONE) is always False.
- `can_craft(world) -> bool [MAX_PLAYERS, N_RECIPES]`: `has_inputs` and the recipe's station within GATHER_RANGE
  (`adjacent_station`; -1 = anywhere). Both exported for the bot policy; `[MAX_PLAYERS, N_RECIPES]` is tiny.
- `gather(world, inputs) -> world`: living active players with `inputs["gather"]`, `has_room`, an adjacent node from
  `adjacent_node`, and `level[NODE_SKILL[kind]] >= NODE_LEVEL_REQ[kind]` claim it; lowest slot wins each node by
  scatter-min (`loot._pickup`'s `winner` pattern). The winner rolls `u * 1000 < chance`, `chance =
  min(NODE_CHANCE_BASE[kind] + (level - NODE_LEVEL_REQ[kind]) * NODE_CHANCE_PER_LEVEL, NODE_CHANCE_MAX)` with one
  `next_key` and `jax.random.uniform(sub, (MAX_PLAYERS,))`. On success: the node gets `active=False`,
  `respawn_in = NODE_RESPAWN[kind]`; the player gets `NODE_PRODUCT[kind]` via `add_item`, `exp[skill] += NODE_EXP[kind]`,
  `gathers += 1`. Failure: nothing changes (the node stays active). Then levels: `level = level_from_exp(exp)`.
- `respawn(world) -> world`: placed inactive nodes count `respawn_in` down; at 0 (after the decrement) `active = True`.
  Stations never deactivate.
- `craft(world, inputs) -> world`: `r = inputs["craft"]` (0 = none, clip to `[0, N_RECIPES-1]`); a player crafts when
  `can_craft(world)[i, r]`. Remove the 4 inputs with `remove_item` in a fixed Python loop over the 4 columns, add
  `RECIPE_OUTPUT[r]` with `add_item` (room is guaranteed since every real recipe has at least one input, but still
  mask on `has_room` after removal), `exp[RECIPE_SKILL[r]] += RECIPE_EXP[r]`, `crafts += 1`, then levels.
- `phase(world, inputs) -> world`: `respawn`, then `gather`, then `craft` (a gather and a craft may both happen in one
  tick). Returns the world with the same keys and shapes; only `world["craft"]` and `players["inv"]` change
  (`rng` advances).
- Tests (each file under 30 s, `jax.jit` at least once, one `jax.vmap` over 2 worlds): add_item/remove_item/count_item/
  has_room on hand inventories; level_from_exp at table boundaries; gather success and failure by forcing the roll
  (pick seeds, or test both outcomes over several seeds) including: node deactivates with the right respawn, product in
  the first empty slot, exp and gathers increase, level requirement blocks, full inventory blocks, dead/inactive
  players never gather, two adjacent players: lowest slot wins and the other gets nothing; respawn countdown and
  reactivation; stations never deactivate; craft: a valid recipe consumes exactly the inputs (multiplicity: two copper
  ore -> one copper bar) and adds the output, a missing input or level or station blocks, recipe 0 does nothing,
  stations needed only when RECIPE_STATION >= 0 (herb potion works anywhere); `phase` under `jax.jit` on a
  `with_craft` world of seed 0 (`make_terrain` world: build it like `bots/runner.make_world_batch` does for one
  world) with no inputs leaves `players` unchanged and keeps shapes/dtypes.

## Worker C (after A and B land): `bots/ro_craft_policy.py` + `tests/test_ro_craft_policy.py`
- Constants at the top: `GATHER_EVERY = 4` (slot `i` is a gatherer when `i % GATHER_EVERY == 1`),
  gatherer node kind `(i // GATHER_EVERY) % N_GATHER_KINDS`.
- `craft_bot_inputs(world, key) -> inputs`: calls `bots.policies.bot_inputs(world, key)` for the base dict and returns
  `{**base, "gather": bool [MAX_PLAYERS], "craft": int32 [MAX_PLAYERS]}` with `move` overridden for gatherers.
  Stateless (everything derived from the world). For a living gatherer that is not hurt (`hp >= HUNT_RETREAT_HP`), not
  resting on pz below max HP, and whose base `attack` is False (a monster adjacent keeps the hunting behaviour):
  1. `craftable = ro_craft.can_craft(world)` masked to recipes whose `RECIPE_SKILL` matches the gatherer's product
     chain (mining kinds -> SKILL_SMITHING, fish -> SKILL_COOKING, herb -> SKILL_HERBALISM, tree -> SKILL_WOODCUTTING);
     pick the highest craftable recipe id (`N_RECIPES - 1 - argmax(reversed)`), 0 if none. If it needs no station or the
     station is adjacent: `craft = id`, `move = 0`. If it needs a station that is not adjacent: walk along
     `dist_node[station_kind]` (smallest of the 4 neighbours, ties -> lowest direction, pz allowed: stations are in town).
     Note `can_craft` already requires the station to be adjacent, so "holds the inputs" must be computed without the
     station condition: ask A for `has_inputs(world) -> bool [MAX_PLAYERS, N_RECIPES]` (inputs + level only) if it is
     not exported yet, or compute it with `ro_craft.count_item`.
  2. Else, if the inventory is not full: walk toward the nearest node of its kind along `dist_node[kind, x, y]`
     (4-neighbour lookups like `bots/policies._neighbour_dist`, blocked tiles unreachable, **never onto pz**); when
     `ro_craft.adjacent_node(world)` says an active node of any gather kind is adjacent and the player has room:
     `gather = True`, `move = 0`.
  3. If the field is `DIST_UNREACHABLE` at the player's tile, or the inventory is full and nothing is craftable: use the
     base move (the bot goes hunting / home).
  Non-gatherers: base inputs with `gather = False`, `craft = 0`.
- Tests: a gatherer next to an active node gathers and stands still; walks downhill on a hand-made field; never steps
  onto pz toward a node; crafts when holding two copper ore next to a furnace; walks toward the furnace field holding
  two ore away from one; a non-gatherer never gathers; shapes/dtypes; `jax.jit` + `jax.vmap` over 2 worlds.

## Worker D (after A and B land): `bench/soak_craft.py`
Like `bench/soak_check.py` but with the craft block and policy: `make_world_batch(seed, batch)`, then
`jax.vmap(ro_craft.place_nodes)` after inserting `make_state()` per world (`jax.vmap(lambda w: {**w, "craft":
ro_craft.make_state()})` is fine), `dist_node` via `jax.lax.map(ro_craft.node_fields, worlds)` (sequential over worlds,
as the runner does for `dist_town`). Tick = `jax.vmap(step)` then `jax.vmap(ro_craft.phase)` with inputs from
`jax.vmap(craft_bot_inputs)` and `next_key` per world; `lax.scan` over ticks per chunk. Invariants: everything
`soak_check.check_invariants` checks except its `NUM_ITEMS` inventory rule (drop messages containing "inventory value
out of range" and check instead that every inventory id is 0..3 or in `CRAFT_ITEM_IDS`), plus: placed gather nodes
on walkable non-pz tiles, no two placed nodes on one tile, `respawn_in >= 0`, inactive placed nodes have
`respawn_in > 0`, stations always active, `level == level_from_exp(exp)`, levels in `[1, CRAFT_MAX_LEVEL]`.
Print per chunk: kills, deaths, gathers, crafts, invariant violations; at the end: gathers and crafts per gatherer bot
per 1000 ticks, per-skill level histogram, and **where the gatherer bots are**: counts on pz, within GATHER_RANGE of an
active node, within GATHER_RANGE of an inactive node, elsewhere; and the mean `dist_node[kind]` value at gatherer
positions. Flags `--batch --chunks --ticks --seed` as in `soak_check.py`. Run it: `--batch 4 --chunks 5 --ticks 100`
and paste the output in your report. Under 10 minutes on CPU.

## Hand-back report (every worker)
Files; the pytest summary line for your file and for the full suite (`-m pytest tests -q`); deviations from this brief
and why; anything the lead must change in a lead-owned file, as exact lines.
