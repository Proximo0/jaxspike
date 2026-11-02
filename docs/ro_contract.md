# RO systems contract: how parallel teams add Ragnarok systems to the JAX tick without colliding

Lead-owned (2026-11-02). Every milestone team (M1 stats/jobs/exp, M2 skills/buffs/elements, M3 items/cards/refine,
M4 gathering/crafting, M5 MVPs/parties, plus world, accounts, art) works on its own branch in its own worktree and
codes against this file and `docs/brief.md`. The lead merges branches in milestone order and wires each system into
the lead-owned files. Vision, scope and feel targets: project memory and `docs/production-handover.md`.

## The ordering rule
**Anything that can be made to work in JAX goes into the JAX tick, first, for speed** (the user). Python keeps only
irregular-shaped work (sockets, login, chat text, a human's click A*). Godot draws. If you find yourself writing a
Python loop over players, stop and write the array op.

## Where a system lives (one module, one state block, one phase)
- Code: `game/ro_<system>.py` (`ro_stats`, `ro_skills`, `ro_items`, `ro_craft`, `ro_party`, `ro_mvp`, …). One public
  phase function `phase(world, inputs) -> world` (or `(world) -> world`), plus pure helpers. Tests in
  `tests/test_ro_<system>.py`, under 30 s on CPU each.
- **Constants and tables** for the system live at the top of its own module (`ro_<system>.py`) or in
  `game/data/<system>.py` (plain tuples / numpy arrays turned into `jnp` arrays at import). This is a deliberate
  exception to "all constants in `game/config.py`": `config.py`, `state.py` and `step.py` are lead-owned and must
  not be edited by any team (report the lines instead). Tables are indexed by id: `JOB_HP_FACTOR[job]`,
  `SKILL_COST[skill]`, `CARD_EFFECT[card]`, `ELEM_MULT[atk_elem, def_elem]`, `MONSTER_TABLE[kind]`.
- **State:** each system contributes one dict under its own world key, built by `make_state() -> dict` in its module.
  The lead adds `"<system>": ro_<system>.make_state()` to `make_empty_world` and the phase call to `step.py` at
  merge. Until merged, your tests build worlds with `make_empty_world(key)` and insert your block:
  `world = {**world, "stats": ro_stats.make_state()}`. Never add keys to `players`, `monsters` or `drops`; add
  per-player arrays inside your own block (`world["stats"]["str"]` is `int32 [MAX_PLAYERS]`).
- **Phase order** (lead wires; design for it): movement -> monsters -> **skills (casts resolve, buffs tick)** ->
  combat (reads derived stats from `stats`, element multipliers, weapon from `items`) -> loot -> **exp/levels** ->
  survival -> **craft** -> **party/mvp** -> tick. If your phase must run elsewhere, say so in the report.
- **Inputs:** new per-player inputs are new keys in the inputs dict (`inputs["skill"]` int32 id or 0,
  `inputs["target"]` int32 entity id or -1, `inputs["use_item"]`, `inputs["craft"]`, `inputs["allot"]` for stat
  points, `inputs["party"]`). Document each new key with its dtype and "0 / -1 means none". The bots module and the
  server fill them; your tests build them by hand. `empty_inputs()` is lead-owned: report the line.
- **Cross-system reads** go through small pure functions with pinned names, so teams can code in parallel:
  - `ro_stats.derived(world) -> dict` of `int32 [MAX_PLAYERS]`: `atk`, `matk`, `hit`, `flee`, `def`, `mdef`,
    `aspd_ticks` (ticks between auto-attacks), `cast_mult` (x1000 fixed point), `max_hp`, `max_sp`, `crit`.
    Until M1 lands, other teams call it through `ro_stats.derived` if the module exists, else use a local stub
    with the same keys **inside their tests only**.
  - `ro_items.equipped(world) -> dict`: `weapon_atk`, `weapon_elem`, `armor_def`, `armor_elem`, `card_bonus`
    (int32 `[MAX_PLAYERS, N_BONUS]`), each `[MAX_PLAYERS]` unless noted.
  - `ro_skills.active_buffs(world) -> int32 [MAX_PLAYERS, N_BUFFS]` remaining ticks (0 = off).
  - `ro_party.party_of(world) -> int32 [MAX_PLAYERS]` party id or -1.
  Fixed point: fractions are ints scaled by 1000 (no float in the tick except where unavoidable; say where).
- **Timers are per-entity counters** (cast_left, cooldown[skill], buff ticks, respawn, mvp_timer); never Python time.
- **Conflicts by scatter-min**: lowest slot wins a tile, a drop, an ore node, an MVP's last hit tie.
- **No per-entity map-sized tables** under vmap (`[N, MAP, MAP]`). Index fields as `field[idx, x, y]`.
- Randomness only via `next_key(world)`. Fixed shapes. Masks over `active`. `jax.jit` once in every test file.

## RO numbers to follow (classic pre-renewal unless noted; keep it fun, not exact)
- Stats 1..99 each, base level 1..99, job level 1..10 (Novice) / 1..50 (first jobs). Stat point cost rises with the
  stat (`1 + (value - 1) // 10` ... use RO's table: 2,3,4,...). Status points from base level; skill points from job level.
- HP = (base_hp_table[level] * job_hp_factor) * (1 + VIT/100); SP similarly with INT. ATK = STR + weapon;
  MATK = INT + INT^2/49 (fixed point); HIT = 175 + level + DEX; FLEE = 100 + level + AGI; ASPD from AGI, DEX and
  weapon class so that ASPD 150..190 gives 1.5..3.3 attacks/s at the server tick (`aspd_ticks = round(TICK_HZ / aps)`).
- Exp: base from monster table, job exp separately; level tables as tuples; death loses 1% base exp at level >= 10.
- Elements: Neutral, Water, Earth, Fire, Wind, Poison, Holy, Shadow, Ghost, Undead; `ELEM_MULT[10, 10]` percent.
- Skills v1: 3 Novice (Basic Skill, First Aid, Play Dead) + ~6 per first job (Swordsman: Bash, Magnum Break,
  Provoke, Endure, HP Recovery, Sword Mastery; Mage: Fire Bolt, Cold Bolt, Lightning Bolt, Napalm Beat, Soul Strike,
  Fire Wall-lite; Archer: Double Strafe, Arrow Shower, Owl's Eye, Vulture's Eye; Acolyte: Heal, Blessing, Increase
  AGI, Angelus, Cure; Thief: Double Attack, Steal, Hiding, Envenom; Merchant: Mammonite, Discount, Vending-lite,
  Pushcart-lite). Each skill: level 1..10, SP cost table, cast ticks, cooldown ticks, range, effect kind id.
- Items: weapon classes (dagger, sword, bow, staff, mace, axe), armor slots (head, body, shield, garment, shoes,
  accessory x2), cards with 0..4 slots per item, refine 0..10 with RO's success chances by level (safe to +4/+7),
  zeny as `int32 [MAX_PLAYERS]`, drop tables per monster kind with per-mille chances. Inventory stays `inv`
  `[MAX_PLAYERS, INV_SLOTS]` item ids (M3 may add `inv_count` and `inv_refine` in its own block, keyed by slot).
- Gathering: ore/fish/herb nodes as `world["craft"]["nodes"]` fixed arrays with respawn counters; a distance field
  per node class for bots (`map_gen.distance_fields`-style, computed once); recipes `[N_RECIPES, 4]` (inputs) ->
  output; skill exp per craft skill.
- Parties: up to 12, shared exp within 15 levels on the same map region; MVP: a monster kind flag with a spawn
  timer and a per-kind `mvp_drop` table; MVP credit = most damage (keep a `[MAX_MONSTERS]` damage ledger per
  attacker? No: keep `[MAX_MONSTERS]` top-damager slot and top-damage, updated by scatter-max).

## Server and client hooks (do not implement in engine teams; note what the net layer must expose)
Each team lists in `docs/<system>_wire.md` (your file): new fields in `snap.you` (level, job, stats, sp, exp, zeny,
equipment, buffs, party), new client messages (`skill`, `use_item`, `equip`, `craft`, `party`, `allot`) with their
input mapping, and any new `welcome` data (tables the client needs for tooltips).

## Working rules for every team
- Branch `ro/<system>` in your own worktree. Commit on your branch only; never touch `main`; never push `main`.
  Push your branch (`git push -u origin ro/<system>`) when green so the lead can merge.
- Python: `C:\Users\crp27\jaxspike\venv\Scripts\python.exe` (absolute path; the worktree has no venv). Run
  `-m pytest tests -q` and `bench/soak_check.py --batch 4 --chunks 5 --ticks 100` before every hand-back; CPU only.
  GPU soaks are run by the lead (one at a time under the 80% cap).
- Workers are Sonnet with disjoint file ownership (your sub-orchestrator writes a short contract per worker:
  signatures, table shapes, files owned); up to 6 engine runs at once per team (24 cores shared by ~9 teams).
- Report: files, pytest summary line, soak invariant line, numbers (e.g. kills/deaths per bot per 1000 ticks before
  and after your phase is wired), the exact lines for `state.py` / `step.py` / `empty_inputs` / `config.py`, your
  wire doc, and a ranked "next 3 defects".
