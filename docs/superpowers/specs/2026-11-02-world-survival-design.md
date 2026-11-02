# World + survival: design (milestone 1 of the Tibia clone)

Date: 2026-11-02. Status: approved in conversation, awaiting written review.

## Context

The project is becoming a public Tibia-style MMO (hundreds of players on one server) with a
hybrid architecture: the JAX engine in `game/` stays the authoritative tick (movement, combat,
monster AI, loot, spatial queries); a plain-Python server layer, added in milestone 2, will own
networking and content. Milestone order: (1) world + survival, (2) networked server + client,
(3) items, equipment, spells, (4) skilling, (5) client polish with pygame-ce and Kenney tiles.

This spec covers milestone 1 only: a multi-town map at a scale that can grow, safe towns,
monster zones, and healing. It builds on the engine as it stands (62 tests, GPU soak clean at
1024 worlds for the 256x256 map).

Decisions taken with the user: procedural towns (Tiled import later), `MAP = 512` for testing
with 2048 planned for the first playable release, passive HP regen plus eating meat, no death
penalty yet, Kenney CC0 art later (not in this milestone).

## Goals

- Five towns with temples, walls, gates and roads between them, on one floor.
- Towns are safe: monsters neither enter nor hit into them.
- Players live in a home town and respawn at its temple.
- Monsters live in zones by distance from towns (rats near, orcs far).
- Players heal: passive regen and eating meat.
- Nothing in the engine depends on the map size except `config.MAP`.
- Every existing test keeps passing; the soak and GPU batch runs stay clean.

## Non-goals

Networking, persistence of players across sessions, new items or spells, NPCs, skilling,
real tile art, hand-authored (Tiled) towns, more than one floor, more monster kinds than
rat and orc, a death penalty.

## 1. Terrain layer

World schema (`game/state.py`) gains:

- `terrain: int8 [MAP, MAP]` with ids in `game/config.py`:
  `T_GRASS=0, T_ROAD=1, T_FLOOR=2, T_TEMPLE=3, T_WATER=4, T_TREE=5, T_WALL=6, T_ROCK=7`.
- `pz: bool [MAP, MAP]`, the protection zone: true on FLOOR, TEMPLE and on ROAD tiles that lie
  inside a town's wall ring.

`walls: int8 [MAP, MAP]` stays and keeps its meaning (1 = blocked). It is derived at generation:
`walls = terrain in {WATER, TREE, WALL, ROCK}`. Movement, monsters, spatial, loot and the
existing tests keep reading `walls` and are not edited for this section.

`make_empty_world` initialises `terrain` to GRASS and `pz` to false. `persistence` round-trips
the new fields (it saves the flattened dict generically; the test gains the two fields).

## 2. Layout spec and generation

### Layout spec (`game/world_layout.py`, lead-owned constants)

A layout is a tuple of towns and a tuple of roads, in map fractions so one spec serves 512 and
2048:

```
TOWNS = (  # (cx, cy, radius) as fractions of MAP; radius is half the wall-ring side
    (0.50, 0.50, 0.060),   # Center
    (0.20, 0.22, 0.050),
    (0.80, 0.20, 0.050),
    (0.78, 0.80, 0.050),
    (0.22, 0.78, 0.050),
)
ROADS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))  # town index pairs
TEMPLE_TILES = 12          # temple square side in tiles, scale-independent
RAT_RING = 0.08            # fraction of MAP: monsters closer than this to a town wall are rats
```

Town radius in tiles is `round(r * MAP)`, with a floor of `TEMPLE_TILES // 2 + 4` so the temple
and a ring of floor always fit. At MAP=512 the centre town is about 61x61 tiles and the others
about 51x51. Names are not in the engine; the server layer will attach them later.

### Generation (`game/map_gen.py`, pure JAX, runs under `vmap` in `make_world_batch`)

`make_terrain(key) -> terrain int8 [MAP, MAP]`, replacing `make_walls`. `make_walls(key)` stays
as a thin wrapper returning the derived blocking mask so existing call sites compile. Steps,
all expressed as per-tile masks (no Python loops over tiles, fixed loops over the 5 towns and
8 roads are fine because the layout is static):

1. Base: GRASS everywhere; a one-tile ROCK border ring.
2. Wilderness noise: two independent low-resolution uniform grids of size `MAP // 32` are
   upsampled with `jax.image.resize` (bilinear) to `MAP x MAP`. Tiles where the first field is
   above `WATER_THRESHOLD` become WATER (lakes); tiles where the second is above
   `FOREST_THRESHOLD` and a per-tile coin flip passes `TREE_DENSITY` become TREE (forests with
   gaps). Thresholds live in `world_layout.py`.
3. Towns, for each town in order: inside the square of half-side `radius` set FLOOR; the
   square's outline is WALL; the central `TEMPLE_TILES x TEMPLE_TILES` square is TEMPLE. Towns
   overwrite water and trees, so towns are always buildable.
4. Roads, for each pair: an L-shaped path (horizontal leg then vertical leg, 3 tiles wide) from
   centre to centre. Road tiles overwrite GRASS, WATER (bridge) and TREE, but not FLOOR,
   TEMPLE or WALL except at the exact crossing of the wall ring, which becomes a ROAD gate
   (the wall tiles the leg passes through are replaced by ROAD). Because legs leave the town
   centre, each road produces one gate per town it touches.
5. Derived: `walls`, and `pz` = FLOOR | TEMPLE | (ROAD inside any town square).

`walkable(walls, pos)` is unchanged.

Scale rule: no coordinate literal outside `world_layout.py`; everything is computed from
`MAP`. Changing `MAP` to 2048 must require no code edit.

### Tests (`tests/test_map.py`, extended)

- Every TEMPLE tile is walkable and inside `pz`.
- From each town's temple, a numpy BFS over walkable tiles reaches every other temple
  (roads connect the world; bridges and gates work).
- No WATER or TREE inside any town square.
- Each town has at least two gate tiles (ROAD on the wall ring).
- Deterministic for a fixed key; two keys differ in wilderness but not in town placement.
- Runs under `jax.jit`, and `vmap` over 3 keys gives shape `[3, MAP, MAP]`.
- Existing tests that used the 8x8 clear square at 16..23 are rewritten to place their
  entities on the centre town's temple tiles (obtained from the layout, not literals).

## 3. Spawns and zones

### Players (`bots/runner.spawn_population`)

- Home town of slot `i` is `i % len(TOWNS)`.
- Each player's `spawn` is a distinct tile of its home temple: tile `k = i // len(TOWNS)` of
  the temple square in row-major order. `len(TOWNS) * TEMPLE_TILES**2 = 720 >= MAX_PLAYERS`;
  a config-time assertion guards this.
- Initial `pos = spawn`. (Respawn already waits while the tile is occupied, and dead players
  are not physical, so shared temples cannot deadlock.)

### Monsters

- Spawn tiles are drawn, as today, from one random ranking of tiles; candidates are walkable
  tiles that are not `pz`.
- Kind by distance from the spawn tile to the nearest town square edge (Chebyshev distance to
  the wall ring): `< RAT_RING * MAP` gives rat, otherwise orc. The population split therefore
  follows the map rather than a 50/50 coin; the soak reports the resulting counts.
- `MONSTER_RESPAWN_TICKS` unchanged; a monster respawns on its own `spawn` as today.

### Protection zone rules

- `monsters._occupancy` treats `pz` tiles as blocked, so monsters never step into a town.
- `combat`: a monster hit is cancelled when the target player stands on a `pz` tile
  (`mhit &= ~pz[p_pos]`). Players can still attack monsters from inside a gate; that is
  acceptable for now (Tibia also lets you hit out of a PZ edge in places).
- Monster spawn selection excludes `pz`, so no monster is ever inside a town.

Soak invariants added: no active monster on a `pz` tile; no living player hp above
`PLAYER_MAX_HP` (already there) and no monster hp above its kind's max (already there).

## 4. Survival phase (`game/survival.py`, new, phase 5; tick counter becomes phase 6)

`survival(world, inputs) -> world`.

1. Regen: when `world["tick"] % REGEN_TICKS == 0`, living players gain `REGEN_HP`, capped at
   `PLAYER_MAX_HP`. Defaults `REGEN_HP = 1`, `REGEN_TICKS = 4` (about 75 s from 0 to 150 at
   8 ticks/s).
2. Eat: `inputs["eat"]` (bool `[MAX_PLAYERS]`, added to `empty_inputs`). A living player with
   `eat` set and at least one `ITEM_MEAT` in `inv` consumes the lowest meat slot (set to
   `ITEM_NONE`) and heals `MEAT_HEAL = 40`, capped at max. One meat per tick at most.
   Eating at full HP still consumes the meat (keeps the rule simple; bots will not do it).
3. Ownership: this phase writes `players.hp` (shared with combat, which only subtracts) and
   `players.inv` (shared with loot, which only adds). Recorded in `docs/brief.md`.

Bots (`bots/policies.py`): `eat = alive & (hp < PLAYER_MAX_HP // 2) & has_meat`.
`play.py`: key `E` eats.

Tests (`tests/test_survival.py`): regen happens only on multiples of `REGEN_TICKS` and never
exceeds max; dead players do not regen; eating removes exactly one meat and heals 40 capped;
eat without meat does nothing; jit and vmap.

## 5. Config changes

In `game/config.py`: `MAP = 512`; terrain ids `T_*`; `REGEN_HP`, `REGEN_TICKS`, `MEAT_HEAL`.
`MAX_PLAYERS`, `MAX_MONSTERS`, `MAX_DROPS` unchanged. `POP_PLAYERS`/`POP_MONSTERS` in the
runner unchanged (128 / 512 per world); with a 4x larger map the monster density drops to a
quarter, which is intended for "towns with wilderness between", and balance is re-measured
after the milestone, not during it.

`K_CELL` stays 8 in this milestone. Players start spread over 5 temples (up to 103 per temple
at 512 players), which overflows the grid in the temple cells; since monsters cannot enter or
hit into towns this is harmless for now. The spatial handoff note already records the fix
(raise `K_CELL`) for when players fight in crowds.

## 6. Scale and performance notes

At MAP=512 the per-world per-tile arrays are 4x larger: `terrain` + `walls` + `pz` are 768 KiB
per world, and `monsters._chase` scatters a `MAP*MAP` int32 claim table (1 MiB) per world per
tick, as does `combat._respawn_players` (an occupancy bool grid). Expectation: the GPU batch
ceiling under the 80% cap falls from 1024 to around 256; this is measured, not assumed, and
recorded in `docs/handoff.md`. If the ceiling falls below 128, the claim tables move to the
sort-based approach that `movement.py` already uses.

The live server in milestone 2 runs one world, where none of this matters.

## 7. play.py

Draws terrain by colour: grass greens as now, ROAD tan, FLOOR grey-brown, TEMPLE pale gold,
WATER blue, TREE dark green circle on grass, WALL as today, ROCK dark grey. The HUD shows the
home town index and "SAFE" while the human stands on a `pz` tile. Key `E` eats.

## 8. Testing and done criteria

- All existing tests pass after their coordinate rewrites; new tests in `test_map.py`,
  `test_survival.py`, `test_runner.py` (spawns on temples, monster kinds by zone) and the
  `pz` monster-blocking and no-hit-into-PZ tests in `test_monsters.py` / `test_combat.py`.
- CPU soak clean for 1000 ticks on seeds 0 and 1 with the new invariants.
- GPU soak: step up from 64 under the 80% cap, record the ceiling.
- Balance report (kills, deaths per bot per 1000 ticks) recorded for the new map, not tuned.
- `docs/brief.md` updated: new phase, new fields, ownership of hp/inv, layout rules,
  "no coordinate literals outside world_layout.py".

## Worker split (fan-out, each owns its files)

1. map: `game/world_layout.py`, `game/map_gen.py`, `tests/test_map.py`.
2. survival: `game/survival.py`, `tests/test_survival.py`, `bots/policies.py` eat rule,
   `tests/test_bots.py` additions.
3. zones: `bots/runner.py` spawns and kinds, `tests/test_runner.py`; `game/monsters.py` and
   `game/combat.py` PZ rules with their tests. (Depends on the `pz` field existing in the
   schema, which the lead lands first with `state.py`, `config.py`, `step.py`, `brief.md`.)
4. play: `play.py` terrain colours, HUD, `E` key (after 1-3 land).
5. lead: soak invariants, GPU runs, handoff.

Order: lead's schema/config/brief first; then workers 1-3 in parallel; integration soak; then
4; then GPU measurement.
