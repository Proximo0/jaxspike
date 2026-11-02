# M1 brief: RO stats, jobs and exp (`game/ro_stats.py`)

Team contract for milestone M1, written by the M1 sub-orchestrator against `docs/ro_contract.md` (read that first;
it wins on any conflict). Workers code against this file at the same time. Pure JAX, fixed shapes, no Python
`if` on traced values, no loops over players, masks over `active`, randomness only via `next_key` (M1 needs none).

## Goal
A per-player stats block `world["stats"]`, the pinned `derived(world)` cross-system read, and a phase
`phase(world, inputs) -> world` that credits exp for kills, levels up, allots stat points, changes job, regens SP
and applies the death penalty. Plus a bot allot policy, a wire doc and a soak harness. Nothing in `game/config.py`,
`game/state.py`, `game/step.py`, `game/combat.py` or any other existing engine file is edited: tests insert the
block by hand and the lead wires the real thing.

## Files and owners (disjoint; edit nothing else)
| worker | files |
|---|---|
| W1 data | `game/data/__init__.py` (empty docstring), `game/data/stats.py` |
| W2 engine | `game/ro_stats.py` |
| W3 tests | `tests/test_ro_stats.py` |
| W4 bots+wire+harness | `bots/ro_stats_policy.py`, `docs/stats_wire.md`, `bench/soak_ro_stats.py` |

## Ids
Jobs: `JOB_NOVICE=0, JOB_SWORDSMAN=1, JOB_MAGE=2, JOB_ARCHER=3, JOB_ACOLYTE=4, JOB_THIEF=5, JOB_MERCHANT=6`,
`NUM_JOBS=7`. Stats: `STAT_STR=0, STAT_AGI=1, STAT_VIT=2, STAT_INT=3, STAT_DEX=4, STAT_LUK=5`, `NUM_STATS=6`.
Monster kinds come from `game/config.py` (`MONSTER_KIND_RAT=0`, `MONSTER_KIND_ORC=1`, `len(MONSTER_HP)` kinds).
All ids are defined in `game/data/stats.py` and re-exported by `game/ro_stats.py`.

## `game/data/stats.py` (W1): tables as Python tuples, plus `jnp.int32` arrays with a `_J` suffix
Every table is a plain tuple of ints (fixed point x1000 where fractional) and a matching `jnp.asarray(..., jnp.int32)`
built at import. Names and shapes are pinned; the numbers are W1's to tune "fun, not exact" within the ranges given.
- `MAX_BASE_LEVEL = 99`, `MAX_JOB_LEVEL_NOVICE = 10`, `MAX_JOB_LEVEL_FIRST = 50`, `MAX_STAT = 99`.
- `BASE_EXP_TABLE` (len 100): base exp needed to go from level `L` to `L+1`, indexed by `L` (index 0 unused = 0,
  index 99 = 0 means cap). Monotone rising, RO-like curve: L1->2 about 9, L10 about 100s, L50 about 100k,
  L98 about 2-4M. `BASE_EXP_TABLE_J`.
- `JOB_EXP_NOVICE_TABLE` (len 11, index by job level, index 10 = 0 cap) and `JOB_EXP_FIRST_TABLE` (len 51, index 50
  = 0 cap). `JOB_EXP_TABLE` int32 `[NUM_JOBS, 51]`: row 0 is the novice table padded with 0s, rows 1..6 the first-job
  table. `JOB_EXP_TABLE_J` (only the 2-D form is needed as an array).
- `MAX_JOB_LEVEL` tuple len NUM_JOBS = (10, 50, 50, 50, 50, 50, 50); `MAX_JOB_LEVEL_J`.
- `BASE_HP_TABLE` (len 100, index by base level): job-neutral max HP before the job factor and VIT; `BASE_HP_TABLE[1]`
  must be 150 (= `PLAYER_MAX_HP` today, so a fresh Novice keeps the tuned balance) and `BASE_HP_TABLE[99]` in 3000..4000.
  `BASE_SP_TABLE` (len 100): `[1] = 10`, `[99]` in 500..800. `_J` arrays for both.
- `JOB_HP_FACTOR` x1000, len NUM_JOBS: Novice 1000, Swordsman 1500, Mage 700, Archer 1000, Acolyte 1000, Thief 1100,
  Merchant 1300. `JOB_SP_FACTOR` x1000: Novice 1000, Swordsman 700, Mage 2000, Archer 1000, Acolyte 1500, Thief 800,
  Merchant 900. `JOB_ASPD_BASE` len NUM_JOBS: ASPD before stats, in 140..160 (Thief/Archer highest, Mage lowest).
  `_J` arrays.
- `MONSTER_BASE_EXP`, `MONSTER_JOB_EXP` (len `len(MONSTER_HP)`, indexed by monster kind): rat (10, 5), orc (40, 25).
  `_J` arrays.
- `STAT_POINTS_START = 48` (a fresh character's unspent status points), `STAT_POINTS_PER_LEVEL` (len 100, indexed by the
  base level just reached): RO's `3 + (L - 1) // 5`. `SKILL_POINTS_PER_JOB_LEVEL = 1`. `STAT_COST` (len 100, indexed by
  the stat's current value): cost to raise it by one = `2 + (value - 1) // 10` (so 1..10 cost 2, 11..20 cost 3, ...,
  91..98 cost 11); index 99 = a huge sentinel (1 << 20) so 99 is never raised. `_J` arrays.
- `SP_REGEN_TICKS = 16`, `DEATH_PENALTY_PERMILLE = 10` (1%), `DEATH_PENALTY_MIN_LEVEL = 10`, `TICK_HZ_X1000 = 8333`
  (1000 * 1000 / `TICK_MS`; compute from config, do not hard-code).
- `STAT_NAMES = ("str", "agi", "vit", "int", "dex", "luk")`, `JOB_NAMES = ("Novice", "Swordsman", "Mage", "Archer",
  "Acolyte", "Thief", "Merchant")`.
Every table has a one-line comment saying what it is indexed by and its scale. A module-level `assert` checks lengths.

## `game/ro_stats.py` (W2)
```python
def make_state() -> dict          # the block under world["stats"], all arrays [MAX_PLAYERS]
def fresh_rows(stats, mask) -> dict   # reset the rows where mask (bool [MAX_PLAYERS]) is True to a fresh Novice; for the server's login/logout
def derived(world) -> dict        # pinned keys, every value int32 [MAX_PLAYERS]
def phase(world, inputs) -> world # runs after loot, before survival
```
State block keys (all `int32 [MAX_PLAYERS]` unless noted; fresh values in brackets):
`str agi vit int dex luk` [1 each], `base_level` [1], `job_level` [1], `base_exp` [0], `job_exp` [0], `job` [0],
`stat_points` [`STAT_POINTS_START`], `skill_points` [0], `sp` [= max_sp of the fresh row], `max_sp` [same],
ledgers: `kills_seen` [0], `deaths_seen` [0], `was_dead` (bool) [False].
`max_sp` is stored (it is also in `derived`) so SP regen and the snapshot need no recompute; the phase refreshes it.

`derived(world)` (reads `world["stats"]` only; weapon/armor/buffs are added by combat from the other systems):
- `max_hp = BASE_HP_TABLE[base_level] * JOB_HP_FACTOR[job] // 1000 * (100 + vit) // 100`
- `max_sp = BASE_SP_TABLE[base_level] * JOB_SP_FACTOR[job] // 1000 * (100 + int) // 100`
- `atk = str + (str // 10) ** 2 + dex // 5 + luk // 5` (RO status ATK; weapon ATK is `ro_items`' job)
- `matk = int + int * int // 49`
- `hit = 175 + base_level + dex`; `flee = 100 + base_level + agi`
- `def = vit // 2 + vit * vit // 150` (soft def, flat points); `mdef = int // 2 + vit // 5`
- `crit` per mille: `10 + luk * 3`
- `cast_mult` x1000: `clip(1000 - dex * 1000 // 150, 0, 1000)`
- `aspd = clip(JOB_ASPD_BASE[job] + (agi * 4 + dex) // 10, 150, 190)`; attacks per second x1000
  `aps = 1500 + (aspd - 150) * 45` (150 -> 1.5/s, 190 -> 3.3/s); `aspd_ticks = max(1, (TICK_HZ_X1000 + aps // 2) // aps)`
  (round to nearest; a fresh Novice lands on 5-6 ticks, close to today's `PLAYER_ATTACK_COOLDOWN`).
All int32 math; `(100 + vit)` products stay far below 2^31 (max ~4000*2000/1000*199).

`phase(world, inputs)`; `inputs["allot"]` int32 `[MAX_PLAYERS]` (-1 none, 0..5 stat index), `inputs["job"]` int32
`[MAX_PLAYERS]` (-1 none, 1..6 target first job). Order inside the phase, all masked by `players.active`:
1. **Kill credit and exp.** `p["kills"] - stats["kills_seen"]` is how many kills each player got this tick (combat
   credits the lowest-slot hitter). Which monster: build a grid over monsters that `died` this tick
   (`build_grid(m["pos"], m["died"])`), `nearest_in_block` from each player; a player with a kill delta > 0 and a died
   monster at distance <= 1 claims that monster by scatter-min of the player slot into a `[MAX_MONSTERS]` claim array
   (lowest slot wins ties). Each claimed monster then pays `MONSTER_BASE_EXP[kind]` / `MONSTER_JOB_EXP[kind]` to its
   claimant (scatter-add). A player whose kill delta is > 0 but who claims nothing (should not happen; combat only
   credits adjacent hitters) gets nothing. Comment this; parties (M5) will replace the payout step.
2. **Base level-up** (loop-free): compute how many levels the new `base_exp` buys with a fixed `lax.fori_loop` of
   `MAX_LEVELS_PER_TICK = 4` iterations over the whole array (each iteration: `need = BASE_EXP_TABLE[level]`,
   `up = (need > 0) & (exp >= need) & (level < MAX_BASE_LEVEL)`, subtract and add). Each level gained adds
   `STAT_POINTS_PER_LEVEL[new_level]` to `stat_points`. At the cap exp is clamped to 0.
3. **Job level-up** likewise with `JOB_EXP_TABLE[job, job_level]` and `MAX_JOB_LEVEL[job]`; each job level adds
   `SKILL_POINTS_PER_JOB_LEVEL` skill points.
4. **Allot**: `i = inputs["allot"]`, valid when `0 <= i < 6`, alive, `value < MAX_STAT`, `stat_points >= STAT_COST[value]`.
   Stats are stacked as `[6, MAX_PLAYERS]` for the gather/scatter (`jnp.stack([...])`), then unstacked.
5. **Job change**: `j = inputs["job"]`, valid when `1 <= j < NUM_JOBS`, alive, `job == JOB_NOVICE`,
   `job_level >= MAX_JOB_LEVEL_NOVICE`. Sets `job = j`, `job_level = 1`, `job_exp = 0`; skill points are kept.
6. **Death penalty**: `p["deaths"] - stats["deaths_seen"] > 0` and `base_level >= DEATH_PENALTY_MIN_LEVEL`:
   `base_exp = max(0, base_exp - BASE_EXP_TABLE[base_level] * DEATH_PENALTY_PERMILLE // 1000)`. No de-levelling.
7. **SP**: recompute `max_sp` from `derived`; alive and `tick % SP_REGEN_TICKS == 0` regens `1 + max_sp // 100 + int // 6`;
   a player who respawned this tick (`was_dead & ~p["dead"]`) gets `sp = max_sp`; clamp `sp` to `[0, max_sp]`.
8. Ledgers: `kills_seen = p["kills"]`, `deaths_seen = p["deaths"]`, `was_dead = p["dead"]`.
Inactive players keep their rows untouched (mask every write with `active`). `fresh_rows` zeroes the ledgers too.
The phase must jit and vmap over a batch of worlds; no `[N, MAP, MAP]` tables (the died-monster grid is `build_grid`
from `game/spatial.py`, which is already batch-safe).

## `bots/ro_stats_policy.py` (W4)
```python
JOB_BUILD = ((0, 2), (0, 2), (3, 4), (4, 1), (3, 2), (1, 0), (0, 2))  # per job: (primary stat, secondary stat)
def bot_allot(world, key) -> int32 [MAX_PLAYERS]    # -1 or a stat index
def bot_job(world, key) -> int32 [MAX_PLAYERS]      # -1 or a first job; slot i picks 1 + i % 6 once job_level >= 10
```
`bot_allot`: when `stat_points >= STAT_COST[primary]` raise the primary unless it is already 2x the secondary (+10),
then the secondary; -1 otherwise. Pure functions of the world (and an unused key, for signature symmetry with
`bot_inputs`); the lead merges them into `bot_inputs` as `inputs["allot"]` and `inputs["job"]`.

## `bench/soak_ro_stats.py` (W4)
A copy of `bench/soak_check.py`'s loop that inserts `"stats": ro_stats.make_state()` (batched with `jax.vmap` over a
dummy axis, or built per world inside `make_world_batch`'s output by broadcasting) into the worlds from
`make_world_batch`, wraps the tick as `step` then `ro_stats.phase`, adds `inputs["allot"]`/`inputs["job"]` from
`bots/ro_stats_policy.py`, keeps all of `check_invariants`, and adds stats-block invariants: stats in 1..99,
levels in range, exp in `[0, table[level])`, `stat_points >= 0`, `sp in [0, max_sp]`, `job` in range, job_level <=
`MAX_JOB_LEVEL[job]`. Prints per chunk: mean base level, max base level, mean job level, number of job changes,
total exp. Same CLI flags as `soak_check.py`. Ends with "all invariants held" or "FIRST VIOLATIONS".

## `docs/stats_wire.md` (W4)
Per `docs/ro_contract.md`'s "Server and client hooks": new `snap.you` fields (`level`, `job_level`, `job`, `stats`
as a 6-list, `stat_points`, `skill_points`, `sp`, `max_sp`, `base_exp`, `base_exp_next`, `job_exp`, `job_exp_next`,
`max_hp`), new client messages `{"type": "allot", "stat": 0..5}` -> `inputs["allot"]`, `{"type": "job", "job": 1..6}`
-> `inputs["job"]` (one-shot, cleared after the tick they apply to), and `welcome` tables for tooltips (`stat_cost`,
`job_names`, `stat_names`, exp tables' lengths). Plus the persistence note: the character file gains the 16
scalar stats fields.

## `tests/test_ro_stats.py` (W3): under 30 s on CPU, one `jax.jit` and one `jax.vmap` test at least
Build worlds with `make_empty_world(key)` + `{**w, "stats": ro_stats.make_state()}`; inputs with `empty_inputs()` plus
`allot`/`job` arrays of -1. Cases: fresh Novice derived values (max_hp in 150..153, hit 177, flee 102, aspd_ticks in
4..7, cast_mult 994 or so); a rat kill pays (10, 5) exp to the lowest-slot adjacent killer only (set up a died monster
and a kills delta by hand); two players adjacent to one died monster -> only the lower slot is paid; level 1->2 on
enough exp with stat points gained; multi-level in one tick (exp for 3 levels) stays consistent; cap at 99; allot
raises a stat and charges the rising cost, refuses at 99 or without points, ignores -1; job change only at Novice job
10 and resets job level; death at level >= 10 loses 1% of the level's requirement and not below 0, no loss below
level 10; SP regen on the regen tick, full SP on respawn, never above max_sp; inactive rows untouched; jit of
`phase` runs and matches eager; vmap over 2 worlds runs. Use `game.data.stats` tables in assertions, not literals.

## Rules for every worker
- Only edit the files listed above. Never commit or push. Never touch `main`. Report only to the M1 lead.
- Python: `C:\Users\crp27\jaxspike\venv\Scripts\python.exe` (absolute; this worktree has no venv). Run your own
  checks with it; never more than 2 engine processes at once, none left running.
- Pure JAX in `game/` and `bots/`: no Python `if` on traced values, no loops over players, fixed shapes, `jnp.where`
  with masks, `.at[].add/min(mode="drop")` for scatters, tables indexed by id.
- Final message: files changed, the checks run with their exact output, the numbers you saw, and any line the lead
  must add elsewhere.
