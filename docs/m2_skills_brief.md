# M2 contract brief: RO skills, buffs and elements (team ro-skills)

Sub-orchestrator-owned, 2026-11-02. Workers code against this file and `docs/ro_contract.md`. Pure JAX in
`game/`, `bots/`: no Python `if` on traced values, no loops over entities, masks over fixed shapes, randomness only
via `next_key`. Do not edit `game/config.py`, `game/state.py`, `game/step.py`, `game/combat.py`.

## Goal
A skills phase inside the JAX tick: players learn skills with skill points, cast them (SP cost, cast time, cooldown,
range), effects resolve as fixed-shape array ops (damage to monsters with element multipliers, heals, buffs, AoE),
buffs tick down and feed stat modifiers to the stats team. Python is `C:\Users\crp27\jaxspike\venv\Scripts\python.exe`.

## Files and owners (disjoint; "don't edit anything else")
| worker | files |
|---|---|
| A data | `game/data/__init__.py` (empty), `game/data/skills.py`, `game/data/elements.py` |
| B core | `game/ro_skills.py` |
| C effects | `game/ro_skills_effects.py` |
| D bot+soak | `bots/ro_skills_policy.py`, `bench/ro_skills_soak.py` |
| E tests+wire | `tests/test_ro_skills.py`, `docs/skills_wire.md` |

## Ids and tables (`game/data/skills.py`, worker A)
Everything a plain Python tuple / `numpy` array turned into a `jnp` array at import, indexed by id.

Jobs (the stats team owns `world["stats"]["job"]`; these ids are what we assume, the lead reconciles):
`JOB_NOVICE=0, JOB_SWORDSMAN=1, JOB_MAGE=2, JOB_ARCHER=3, JOB_ACOLYTE=4, JOB_THIEF=5, JOB_MERCHANT=6, N_JOBS=7`.

Skills: `SKILL_NONE = 0`, then ids 1..32 in this exact order (`N_SKILLS = 33`, `MAX_SKILL_LEVEL = 10`):
```
1 SK_BASIC_SKILL  2 SK_FIRST_AID  3 SK_PLAY_DEAD                                    (Novice)
4 SK_BASH  5 SK_MAGNUM_BREAK  6 SK_PROVOKE  7 SK_ENDURE  8 SK_HP_RECOVERY  9 SK_SWORD_MASTERY   (Swordsman)
10 SK_FIRE_BOLT 11 SK_COLD_BOLT 12 SK_LIGHTNING_BOLT 13 SK_NAPALM_BEAT 14 SK_SOUL_STRIKE 15 SK_FIRE_WALL (Mage)
16 SK_DOUBLE_STRAFE 17 SK_ARROW_SHOWER 18 SK_OWLS_EYE 19 SK_VULTURES_EYE               (Archer)
20 SK_HEAL 21 SK_BLESSING 22 SK_INCREASE_AGI 23 SK_ANGELUS 24 SK_CURE                   (Acolyte)
25 SK_DOUBLE_ATTACK 26 SK_STEAL 27 SK_HIDING 28 SK_ENVENOM                              (Thief)
29 SK_MAMMONITE 30 SK_DISCOUNT 31 SK_VENDING 32 SK_PUSHCART                             (Merchant)
```
Per-skill arrays, all `jnp.int32 [N_SKILLS]` (row 0 = none, all zeros):
- `SKILL_JOB` job id that may learn it (Novice skills: 0; every job may learn Novice skills).
- `SKILL_MAX_LEVEL` 1..10 (RO values: Bash 10, Magnum Break 10, Provoke 10, Endure 10, HP Recovery 10, Sword
  Mastery 10, bolts 10, Napalm 10, Soul Strike 10, Fire Wall 10, Double Strafe 10, Arrow Shower 10, Owl's Eye 10,
  Vulture's Eye 10, Heal 10, Blessing 10, Inc AGI 10, Angelus 10, Cure 1, Double Attack 10, Steal 10, Hiding 10,
  Envenom 10, Mammonite 10, Discount 10, Vending 10, Pushcart 10, Basic Skill 9, First Aid 1, Play Dead 1).
- `SKILL_EFFECT` effect kind: `EFFECT_PASSIVE=0` (cannot be cast; learned level feeds `buff_stat_mods`),
  `EFFECT_ATK=1` (weapon damage: base = derived atk), `EFFECT_MATK=2` (magic damage: base = derived matk),
  `EFFECT_HEAL=3`, `EFFECT_BUFF=4` (applies `SKILL_BUFF` for `SKILL_BUFF_TICKS[skill, lvl]` to the target),
  `EFFECT_NONE=5` (castable, costs SP, does nothing in v1: Steal, Cure, Vending, Pushcart-lite stay as stubs).
  Passives: Basic Skill, HP Recovery, Sword Mastery, Owl's Eye, Vulture's Eye, Double Attack, Discount, Pushcart.
- `SKILL_ELEMENT` element id of the damage (Fire Bolt fire, Cold Bolt water, Lightning Bolt wind, Napalm Beat and
  Soul Strike ghost, Fire Wall fire, Magnum Break fire, Envenom poison, everything else neutral; see elements).
- `SKILL_TARGET`: `TARGET_SELF=0, TARGET_ALLY=1, TARGET_ENEMY=2, TARGET_GROUND=3`. Ground skills (Fire Wall,
  Arrow Shower, Magnum Break is self-centred AoE so `TARGET_SELF`) resolve at the tile where the targeted entity
  stood when the cast started.
- `SKILL_AOE` radius in tiles (Chebyshev, `max(|dx|,|dy|) <= r`): Magnum Break 1 (around the caster), Arrow
  Shower 1, Fire Wall 1, Napalm Beat 1 around the target, all others 0 (single target).
- `SKILL_BUFF` buff id applied by `EFFECT_BUFF` skills, else 0.
- `SKILL_HITS` number of hits (Double Strafe 2, Soul Strike 1..5 by level handled via POWER instead; keep 1
  everywhere except Double Strafe 2); power is per hit.
Per-skill-per-level arrays, `jnp.int32 [N_SKILLS, MAX_SKILL_LEVEL + 1]`, column 0 unused (zeros):
- `SKILL_SP` SP cost (RO pre-renewal tables: Bash 8..15, Magnum Break 20..30-ish, bolts 10..34 stepping, Heal
  13..52, Blessing 28..64, Inc AGI 18..45, etc. Use RO's real tables where you know them, otherwise a linear ramp).
- `SKILL_CAST` cast ticks (tick = 120 ms, so RO 0.7 s bolt per level ~ 6 ticks per level: Fire Bolt lvl n ≈ 6n
  ticks, Heal 1 s = 8, Blessing / Inc AGI 1 s = 8, Bash / Double Strafe / Mammonite 0, Magnum Break 0).
- `SKILL_COOLDOWN` ticks before the same skill can start again (after-cast delay; Bash 4, bolts 8, Double Strafe
  4, Arrow Shower 8, Heal 8, buffs 8, Magnum Break 16, Mammonite 4; never 0 for castable skills).
- `SKILL_RANGE` tiles, manhattan (melee 1; bolts and Heal 9; Double Strafe / Arrow Shower 9; buffs 9; Provoke 9).
  The skills phase checks range itself with a manhattan distance over the target's pos, so ranges above `CELL`
  are fine here (the spatial grid is not used for the range check).
- `SKILL_POWER` x1000 per hit: Bash 1000 + 300*lvl... use RO: Bash 100%+30%/lvl → 1300 at lvl 1; Magnum Break
  100%+20%/lvl; Mammonite 150%+50%/lvl; Double Strafe 90%+10%/lvl per hit; Arrow Shower 75%+5%/lvl; bolts 100%*lvl
  → 1000*lvl; Napalm Beat (70%+10%*lvl); Soul Strike lvl hits folded in → 1000 * ((lvl+1)//2) ... keep
  ((lvl + 1) // 2) * 1000; Envenom 100%+15%/lvl; Fire Wall 50%/lvl per tick-hit → 500*lvl once (v1 single hit);
  Heal power = 1000 * (4 + 8*lvl) / 8 → heal amount = (10 + matk) * POWER // 1000 ... see effects;
  First Aid heals a flat 5 (POWER 5000 with base 1: effects treats `EFFECT_HEAL` with `SKILL_FLAT_HEAL`).
- `SKILL_BUFF_TICKS` duration for `EFFECT_BUFF` skills per level (Blessing 60 s +... use RO: Blessing/Inc AGI
  (40 + 20*lvl) s → *8.33 ticks/s ≈ (40+20*lvl)*8 ticks, Angelus (30*lvl) s*8, Endure 10 s*8 = 80 (flat), Hiding
  (30+30*lvl)*8, Provoke 30 s*8, Play Dead 0 = until cancelled: use 100000).
- `SKILL_FLAT_HEAL` `[N_SKILLS]` flat heal added (First Aid 5, else 0).
- `SKILL_NAME` tuple of 33 strings for the wire doc and tooltips.

Buffs (`N_BUFFS = 8`): `BUFF_NONE=0, BUFF_BLESSING=1, BUFF_INC_AGI=2, BUFF_ANGELUS=3, BUFF_ENDURE=4,
BUFF_PROVOKE=5, BUFF_HIDING=6, BUFF_PLAY_DEAD=7`. Buff level matters (Blessing lvl 5 = +5 STR/INT/DEX), so the
state keeps `buff_level int32 [MAX_PLAYERS, N_BUFFS]` next to the ticks. Provoke in v1 is a self-buff on the
caster's chosen monster's... no: monsters have no buff array, so **Provoke v1 is `TARGET_ENEMY` + `EFFECT_NONE`**
(costs SP, nothing else). Document it.

SP: `SP_REGEN_TICKS = 8`, `SP_REGEN_FLAT = 1` (per regen: `1 + max_sp // 100`), `SP_START = 40`.

## Elements (`game/data/elements.py`, worker A)
`ELEM_NEUTRAL=0, ELEM_WATER=1, ELEM_EARTH=2, ELEM_FIRE=3, ELEM_WIND=4, ELEM_POISON=5, ELEM_HOLY=6,
ELEM_SHADOW=7, ELEM_GHOST=8, ELEM_UNDEAD=9, N_ELEMENTS=10`. `ELEM_MULT = jnp.int32 [10, 10]` percent,
`ELEM_MULT[attack_elem, defend_elem]`, RO pre-renewal level-1 defender table (the classic one: Neutral row all 100
except Ghost 25; Water vs Earth 100... use the real table: e.g. Fire vs Water 50? no: Water attack vs Fire defender
150, Fire vs Water 50, Fire vs Earth 150, Earth vs Wind 150, Wind vs Water 150, Water vs Earth... follow the published
pre-renewal element table, level 1 column, exactly, and cite it in a comment). `ELEM_NAME` tuple.
`MONSTER_ELEMENT = jnp.int32 [len(MONSTER_HP)]`: rat Neutral (0), orc Earth (2). Keep it in `elements.py`
(it indexes `game.config.MONSTER_HP` length only).
`def elem_mult(attack_elem, defend_elem) -> int32` pure gather, broadcastable.

## State block `world["skills"]` (`ro_skills.make_state() -> dict`, worker B)
```
"learned":     int32 [MAX_PLAYERS, N_SKILLS]   level 0..10; column 0 always 0
"cooldown":    int32 [MAX_PLAYERS, N_SKILLS]   ticks until the skill may start; 0 = ready
"cast_left":   int32 [MAX_PLAYERS]             ticks until the current cast resolves
"casting":     int32 [MAX_PLAYERS]             skill id being cast, 0 = idle
"cast_target": int32 [MAX_PLAYERS]             global entity id (players 0..MP-1, monsters MP+i), -1 = none
"cast_tile":   int32 [MAX_PLAYERS, 2]          target tile snapshotted at cast start (ground / AoE centre)
"buffs":       int32 [MAX_PLAYERS, N_BUFFS]    remaining ticks, 0 = off; column 0 always 0
"buff_level":  int32 [MAX_PLAYERS, N_BUFFS]    level the buff was applied at
"sp":          int32 [MAX_PLAYERS]             current SP (SP_START on a fresh state)
"casts":       int32 [MAX_PLAYERS]             casts resolved (counter, for soak numbers)
"skill_dmg":   int32 [MAX_PLAYERS]             total damage dealt by skills (counter)
```
`active_buffs(world) -> int32 [MAX_PLAYERS, N_BUFFS]` returns `world["skills"]["buffs"]`.

## Inputs (new keys; the lead adds them to `empty_inputs()`)
- `inputs["skill"]` int32 `[MAX_PLAYERS]` skill id to cast this tick, 0 = none.
- `inputs["target"]` int32 `[MAX_PLAYERS]` global entity id (players `0..MAX_PLAYERS-1`, monsters
  `MAX_PLAYERS + i`), -1 = none. Self-target skills ignore it.
- `inputs["learn"]` int32 `[MAX_PLAYERS]` skill id to learn one level of, 0 = none.
Tests and the bot build them by hand: `{**empty_inputs(), "skill": ..., "target": ..., "learn": ...}`.

## Cross-system reads
- `derived`: `ro_stats.derived(world)` keys `atk, matk, hit, flee, def, mdef, aspd_ticks, cast_mult (x1000),
  max_hp, max_sp, crit`, each int32 `[MAX_PLAYERS]`. **`ro_skills.phase(world, inputs, derived=None)`**: when
  `derived` is None the phase calls `ro_stats.derived(world)` if `game.ro_stats` imports, else
  `ro_skills.fallback_derived(world)` (atk `PLAYER_ATTACK`, matk 10, max_hp `PLAYER_MAX_HP`, max_sp 50,
  cast_mult 1000, others 0 / aspd `PLAYER_ATTACK_COOLDOWN`). The fallback is a documented stopgap the lead
  removes when M1 merges. Tests pass an explicit stub to pin numbers.
- Job: `world["stats"]["job"]` int32 `[MAX_PLAYERS]` if `"stats" in world and "job" in world["stats"]`, else
  every player is `JOB_NOVICE`. Skill points: `world["stats"]["skill_points"]` int32 `[MAX_PLAYERS]` if present,
  decremented on learn; if absent learning is free (and documented as such). These are Python dict checks on
  the static pytree structure, not on traced values, so they are jit-safe.
- Monster element: `MONSTER_ELEMENT[world["monsters"]["kind"]]`.

## The phase (`ro_skills.phase(world, inputs, derived=None) -> world`, worker B), in this order
1. **Learn:** `inputs["learn"] = s` with `s != 0`, player active, `learned[p, s] < SKILL_MAX_LEVEL[s]`,
   `SKILL_JOB[s] in (JOB_NOVICE, job[p])`, and (if present) `skill_points[p] > 0` → `learned[p, s] += 1`,
   `skill_points[p] -= 1`. Dead players may learn.
2. **Tick timers:** `cooldown = max(cooldown - 1, 0)`, `buffs = max(buffs - 1, 0)`, `buff_level` zeroed where
   buffs hit 0. Dead players lose every buff and their cast (`casting = 0`, `cast_left = 0`).
3. **Start casts:** `s = inputs["skill"]`, valid when: player alive, `casting == 0`, `s != 0`,
   `learned[p, s] > 0`, `SKILL_EFFECT[s] != EFFECT_PASSIVE`, `cooldown[p, s] == 0`, `sp >= SKILL_SP[s, lvl]`,
   target valid for `SKILL_TARGET[s]` (self: always; ally: `0 <= t < MAX_PLAYERS`, target active and not dead;
   enemy / ground: `MAX_PLAYERS <= t < MAX_PLAYERS + MAX_MONSTERS` and that monster active), and manhattan
   distance from the caster to the target entity's pos `<= SKILL_RANGE[s, lvl]` (self: 0). On start: SP is
   deducted, `cooldown[p, s] = SKILL_COOLDOWN[s, lvl]`, `casting = s`, `cast_target = t` (self: own slot id),
   `cast_tile = target pos` (self: own pos), `cast_left = SKILL_CAST[s, lvl] * cast_mult // 1000`.
4. **Resolve:** `resolve = alive & (casting != 0) & (cast_left == 0)` (an instant skill resolves the tick it
   starts). Call `ro_skills_effects.resolve(world, resolve_mask, derived) -> world` (worker C). Then
   `casting = where(resolve, 0, casting)`, `casts += resolve`.
5. **Count down casts:** `cast_left = where(casting != 0, max(cast_left - 1, 0), 0)` for casts not resolved.
6. **SP regen:** alive players, every `SP_REGEN_TICKS` ticks: `sp = min(sp + SP_REGEN_FLAT + max_sp // 100,
   max_sp)`. Also `sp = min(sp, max_sp)` always.
Everything is a mask over `[MAX_PLAYERS]`; per-skill lookups are `TABLE[s, lvl]` gathers with
`lvl = learned[p, s]` (gather with `jnp.take_along_axis` or `learned[pidx, s]`).

Public helpers in `ro_skills.py` for the stats team and tests:
- `active_buffs(world)` as above.
- `buff_stat_mods(world) -> dict` of int32 `[MAX_PLAYERS]`, keys exactly:
  `str_flat, agi_flat, vit_flat, int_flat, dex_flat, luk_flat, atk_flat, atk_pct, matk_pct, hit_flat, flee_flat,
  def_pct, mdef_flat, aspd_pct, crit_flat, hp_regen_pct` (pct = percent points, 0 = no change). From active
  buffs: Blessing lvl L: `str_flat, int_flat, dex_flat += L`; Inc AGI lvl L: `agi_flat += 2 + L`,
  `aspd_pct += L`; Angelus L: `def_pct += 5 * L`; Endure L: `mdef_flat += L`; Hiding: `flee_flat += 0` (no-op in v1,
  documented); Play Dead: none. From learned passives: Sword Mastery L: `atk_flat += 4 * L`; Owl's Eye L:
  `dex_flat += L`; Vulture's Eye L: `hit_flat += L` (range bonus is not modelled in v1); Double Attack L:
  `crit_flat += 0` (reported as a defect: needs a hit-count hook in combat); HP Recovery L: `hp_regen_pct += 10 * L`.
  Pure, no randomness, jit-safe.
- `fallback_derived(world)`.

## Effects (`game/ro_skills_effects.py`, worker C): `resolve(world, mask, derived) -> world`
`mask` bool `[MAX_PLAYERS]`: casters whose cast resolves this tick; read `s = casting`, `lvl = learned[p, s]`,
`t = cast_target`, `tile = cast_tile`. Small pure functions, each testable alone:
- `base_damage(derived, effect) -> int32 [MP]`: `where(effect == EFFECT_ATK, atk, where(effect == EFFECT_MATK, matk, 0))`.
- `skill_damage(base, power, hits, attack_elem, defend_elem) -> int32`: `base * power // 1000 * hits *
  elem_mult(attack_elem, defend_elem) // 100`, minimum 0.
- `heal_amount(matk, power, flat) -> int32`: `(10 + matk) * power // 1000 + flat`.
- **Single-target damage** (`SKILL_AOE == 0`, effect ATK/MATK, target a monster): scatter-add damage onto the
  monster, exactly as `game/combat.py` does (`.at[tgt_w].add(dmg, mode="drop")` with out-of-range index for
  non-casters).
- **AoE damage** (`SKILL_AOE == r > 0`): never build `[MAX_PLAYERS, MAX_MONSTERS]`. Build a monster-at-tile
  index map once: `occ = full((MAP, MAP), MAX_MONSTERS).at[mpos_x, mpos_y].min(midx)` over active monsters
  (scatter-min, lowest slot wins), then for each caster and each of the `(2*R_MAX+1)^2` offsets (`R_MAX = 1`, so
  9 offsets, a fixed `[9, 2]` constant) gather `occ[clip(tile + off)]`, mask offsets with `max(|off|) <= r` and
  in-bounds, and scatter-add damage from the `[MP, 9]` gathered monster indices (`mode="drop"`, out-of-range for
  masked entries). Single-target is the same path with `r = 0` only if you find it simpler; either way one scatter.
- **Deaths and credit:** after the scatter: `new_hp = hp - dmg`, `killed = active & (hp > 0) & (new_hp <= 0)`;
  credit the lowest-slot caster that hit the monster (`full(MM, MP).at[idx].min(pidx)` scatter-min over the same
  `[MP, 9]` indices); `kills[credit] += 1`; monsters: `hp = new_hp`, `active &= ~killed`,
  `respawn_in = where(killed, MONSTER_RESPAWN_TICKS, respawn_in)`, `died |= killed`. This mirrors `combat.py`
  so loot and respawn work unchanged. `skill_dmg[p] += damage dealt (sum over its 9 slots, clipped to the
  monster's remaining hp is NOT required; count raw damage)`.
- **Heal** (`EFFECT_HEAL`, target a player incl. self): `hp[t] = min(hp[t] + heal, max_hp[t])` via scatter-add then
  clamp with `derived["max_hp"]`; only living targets.
- **Buff** (`EFFECT_BUFF`): `buffs[t, b] = max(buffs[t, b], ticks)`, `buff_level[t, b] = lvl` where applied (self
  or ally target). Scatter with `.at[t, b].max(...)` and level set by `where`.
- `EFFECT_NONE`: nothing. Return the world with updated `players` (hp, kills), `monsters`, and `skills`
  (`buffs`, `buff_level`, `skill_dmg`). No randomness in v1 (no miss/crit; HIT/FLEE come later: keep the
  per-hit damage in `skill_damage` so a hit-roll can wrap it).

## Bot and soak harness (worker D)
- `bots/ro_skills_policy.py`: `bot_inputs(world, key) -> inputs` pure: calls `bots.policies.bot_inputs` for move /
  attack / eat, then adds `learn` (learn the job's **first castable skill** — the table
  `JOB_FIRST_SKILL = (SK_FIRST_AID, SK_BASH, SK_FIRE_BOLT, SK_DOUBLE_STRAFE, SK_HEAL, SK_ENVENOM, SK_MAMMONITE)`
  indexed by job — while its level is below 5), `skill` (that skill when a monster is adjacent (dist == 1 via
  `nearest_in_block`, same as the base policy's `attack`), the skill is learned, not casting, cooldown 0 and SP
  enough; Acolytes cast Heal on themselves when hp < half instead), `target` (the global id `MAX_PLAYERS + idx` of
  that nearest monster, or own slot for Heal). Job per bot: `world["stats"]["job"]` if present else slot `% N_JOBS`.
- `bench/ro_skills_soak.py`: copies the `main()` shape of `bench/soak_check.py` but with a local `step` that
  inserts `ro_skills.phase(world, inputs, derived=stub)` between `monsters.monster_ai` and `combat.combat`, a
  stub `stats` block `{"job": slot % N_JOBS, "skill_points": 20}` and `world["skills"] = make_state()` added to
  the batch in `make_world_batch` output, the bot policy above, the same invariant checks (import
  `check_invariants` from `bench.soak_check`) **plus**: `sp in [0, max_sp]`, `cooldown >= 0`, `buffs >= 0`,
  `cast_left >= 0`, `casting in [0, N_SKILLS)`, `learned in [0, 10]`, no monster with `active & hp <= 0`. Prints
  per chunk: kills, deaths, casts resolved, skill damage, and at the end **casts per bot per 1000 ticks and the
  share of monster damage from skills** (skill_dmg / (skill_dmg + kills_by_auto_attack * ... ) — simplest: also
  count auto-attack damage by summing `PLAYER_ATTACK` per `attack` input landing; if that needs combat internals,
  report `skill_dmg` and total monster hp removed = kills * mean MONSTER_HP as the denominator and say so).
  Args like the original: `--batch 4 --chunks 5 --ticks 100`.

## Tests (`tests/test_ro_skills.py`, worker E), under 30 s on CPU, `jax.jit(phase)` once
A `_world()` helper like `tests/test_combat.py`'s, with `"skills": make_state()` and a stub `derived`
(`atk=10, matk=20, max_hp=PLAYER_MAX_HP, max_sp=50, cast_mult=1000, ...`) passed to `phase`. Cover: make_state
shapes and dtypes; learn (job gate, max level, skill points decrement when `stats` block present, free when
absent); cast start (SP deducted, cooldown set, cast_left from table and cast_mult); instant Bash resolves the
same tick with `atk * 1300 // 1000` damage on a neutral rat; Fire Bolt lvl 1 takes `SKILL_CAST` ticks and does
`matk * 1000 // 1000 * ELEM_MULT[fire, elem of orc] // 100` on an orc; kill via skill sets `died`, `active`
False, `respawn_in`, kill credit; AoE Magnum Break hits all monsters within Chebyshev 1 of the caster and none at
distance 2; range check rejects a far target and a dead/inactive target; Heal clamps at max_hp; Blessing lvl 3
sets `buffs` ticks and `buff_level` and `buff_stat_mods` reports `str_flat == 3`; buffs tick down and vanish; a
dead caster's cast is cancelled; cooldown blocks a second cast; SP regen on schedule; `ELEM_MULT` spot checks
(fire vs water 50? per the published table, neutral vs ghost 25, holy vs undead 125/150); the full phase is
`jax.jit`-able and `jax.vmap`-able over a batch of 2 worlds with identical results per world; passives cannot be
cast. Also `docs/skills_wire.md` per `docs/ro_contract.md`: `snap.you` fields `sp`, `max_sp`, `buffs`
(list of `[buff_id, ticks_left, level]` for nonzero), `cooldowns` (list of `[skill_id, ticks]` nonzero),
`learned` (list of `[skill_id, level]`), `casting` / `cast_left`; client messages `{"t": "skill", "id":, "target":}`
→ `inputs["skill"]`, `inputs["target"]` for that player's slot for one tick; `{"t": "learn", "id":}` →
`inputs["learn"]`; `welcome` adds the skill table (names, max level, sp/cast/cooldown/range per level, target
type, effect, element) and `ELEM_NAME`.

## Standard preamble for every worker
- Report only to the lead (the session that launched you); never message the user.
- Only edit the files listed for you. Don't commit or push. Don't touch `game/config.py`, `game/state.py`,
  `game/step.py`, `game/combat.py`, `bots/policies.py`, `bench/soak_check.py`.
- Python: `C:\Users\crp27\jaxspike\venv\Scripts\python.exe` (run from the worktree root
  `C:\Users\crp27\jaxspike\.claude\worktrees\ro-skills-47ef0c`). Keep engine runs short, at most 2 at once, never
  leave processes running, CPU only.
- Final message: files changed, the checks run with their output (pytest summary line), any deviation from this
  brief and why, and open defects.
