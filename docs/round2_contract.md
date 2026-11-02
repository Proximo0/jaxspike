# Round 2 contract: Ragnarok Online fidelity, animation and movement feel in the JAX core

Lead-owned (2026-11-02, night). Builds on `docs/ro_contract.md` (still in force: module/state/phase pattern,
lead-owned files, branch and report rules) and `docs/m0_brief.md` (protocol v2). The user's direction for this
round: **model the game as a clone of Ragnarok Online (classic, pre-renewal), not Tibia; make animations work well
and smoothly, with the animation state computed in JAX; go wide (up to 200 Sonnet workers across 10+ teams).**

## 0. RO facts every team designs against (researched 2026-11-02; cite irowiki / rAthena when you add more)
- **Movement:** cell grid, **8 directions**, default walk speed **150 ms per cell**; diagonal steps cost 14/10 of a
  straight step (rAthena convention). Speed is a per-entity number in ms/cell (AGI Up / Increase AGI lower it,
  Decrease AGI raises it; Peco Peco 150 -> 120-ish). Movement is click-to-walk along a path; a new click replaces
  the path. Monsters have their own speed per kind (`MOVE_TICKS` today becomes `speed_ms`).
- **Attack timing:** amotion (attack animation) = `2000 - ASPD * 10` ms; attack delay = 2 x amotion for players,
  so **swings per second = 50 / (200 - ASPD)**: ASPD 150 = 1/s, 170 = 1.67/s, 190 = 5/s (cap 190 pre-renewal).
  The damage number and the target's flinch happen at the attack frame of the attacker's clip ("atk" event), not
  at the start of the swing. The feel target in memory ("ASPD 150-190 = 1.5-3.3/s") is replaced by this.
- **Casting:** cast time (reduced by DEX: `cast * (1 - DEX/150)` pre-renewal), then **after-cast delay** during
  which no skill can be used; the cast bar shows above the caster; moving does not cancel in classic (it does on
  some servers): keep "no cancel" in v1.
- **Sprites (ACT/SPR):** one clip per action x 8 directions; direction 0 = south, then clockwise through SW, W,
  NW, N, NE, E, SE (the client mirrors: RO sprites only draw 5 of the 8 and mirror the rest). Player action slots:
  idle 0, walk 8, sit 16, pick up 24, standby 32, attack 40, hurt 48, frozen 56, dead 64, attack2 80, attack3 88,
  cast 96 (+ direction 0..7). Monster slots: idle 0, walk 8, attack 16, hurt 24, dead 32. Frame delays are per
  frame in the ACT (units of ~25 ms); walk clips are scaled by walk speed, attack clips by amotion.
- **Monster modes:** passive / aggressive (sight range ~ 10-12 cells) / assist / looter / cast-sensor /
  immovable; random walk when idle (about every 3-5 s a short wander), chase radius, return home; respawn timers
  per spawn group (classic: 5 s to minutes); MVPs with 1-2 h timers in their own maps.
- **World:** RO is separate maps joined by portals (warps) at map edges; towns (Prontera, Geffen, Payon, Alberta,
  Morroc, Izlude...) with kafra (save, storage, teleport for zeny), NPC shops (buy/sell), fields `prt_fild01..`,
  dungeons by floor. Our one-big-map engine models each RO map as a **zone** (a rectangle of the 2048 map) with
  portal tiles at its edges; the client shows the zone name on entry.
- **Feel references the user cares about:** fun, not over-complicated; RO's snappy click-to-walk, visible swings
  and flinches, damage numbers, the skill bar and cast bar, sitting to regen, Porings bouncing.

## 1. Animation and movement in the JAX core (the model, lead-designed; teams ro-move and ro-anim build it)

Everything timing-related lives in the tick as per-entity arrays; the client plays clips from that state and
interpolates positions. **No animation logic on the server's Python side.**

### 1.1 Movement with speed accumulators (ro-move)
- `world["move"]` block (players and monsters, `N = MAX_PLAYERS + MAX_MONSTERS`, indexed by entity id order):
  `speed_ms int16 [N]` (150 default; from stats/buffs for players, from the monster table for monsters),
  `acc_ms int16 [N]` (accumulated time toward the next step), `dir int8 [N]` (0..7 RO convention, facing),
  `path_dir int8 [N]` (the next step the entity wants: 0..7 or -1), `moving bool [N]`.
- Each tick: `acc += TICK_MS` for entities that want to move; a step happens when `acc >= cost`, with
  `cost = speed_ms` for straight and `speed_ms * 14 // 10` for diagonal; `acc -= cost` (the remainder carries).
  With TICK_MS 100 and speed 150: steps at ticks 2, 3, 5, 6, 8... exactly 1 cell per 150 ms on average.
  Occupancy and conflicts stay as today (lowest slot wins); a blocked step keeps `acc` (retry next tick).
- `DIR_DELTAS` grows to 9 entries: 0 stay, 1..8 = S, SW, W, NW, N, NE, E, SE (`game/config.py`, lead applies).
  Diagonal moves require both orthogonal neighbours walkable (no corner cutting through walls), as RO.
- Click-to-walk: `net/pathing.py` A* becomes 8-connected with the 14/10 cost; the server feeds one `path_dir`
  per step (not per tick): the engine consumes it when the accumulator allows. Humans' keyboard moves set
  `path_dir` directly. Bots: policies output a direction 0..8.
- Monsters use the same accumulator with their kind's `speed_ms` (table in `game/data/monsters.py`).

### 1.2 Animation state (ro-anim)
- `world["anim"]` block: `action int8 [N]` (0 idle, 1 walk, 2 sit, 3 pickup, 4 standby, 5 attack, 6 hurt,
  7 dead, 8 cast), `start_tick int32 [N]` (tick the action began), `dur_ms int16 [N]` (clip duration the
  client should fit the clip into: amotion for attack, hurt_ms 300 for hurt, cast ticks x TICK_MS for cast, 0 =
  loop), `face int8 [N]` (copied from move.dir each tick), `hit_tick int32 [N]` (tick the entity was last hit, for
  the damage number), `hit_dmg int32 [N]` (last damage taken; 0 = none this tick).
- Pure API other phases call (fixed-shape, mask-based): `ro_anim.set_action(world, mask, action, dur_ms)`;
  `ro_anim.note_hit(world, mask, dmg)`. Rules: movement sets walk when a step happens and idle when an entity
  with no `path_dir` has been still for one tick; combat sets attack on the attacker (dur = amotion) and hurt on
  the target unless the target is attacking or casting (RO: flinch interrupts walking but not an attack in
  progress... keep: hurt only overrides idle/walk/standby); death sets dead (sticky until respawn); skills set
  cast with dur = cast ticks x TICK_MS; after-cast delay: `skills.after_delay` counter blocks new casts.
  Standby (4) after an attack until 3 s without a swing, then idle.
- On the wire (protocol v2, additive): `ents[*]` gain `a` (action), `d` (face), `t` (start_tick), `m` (dur_ms),
  `s` (speed_ms) and, when `hit_dmg != 0` this tick, `h` (damage number). `changed_mask` includes action,
  face, start_tick and hit_tick. `you` gains `speed_ms`, `amotion_ms`, `after_delay`.

### 1.3 Client playback (client-anim)
- The client keeps a clock in server ticks (`render_tick` = newest snap tick - interp_delay); an entity's
  position is interpolated between its previous and current cell over `speed_ms` (the time a step takes), not
  over the tick interval, which is what makes 150 ms/cell walking smooth at a 100 ms tick. Diagonal steps take
  `speed_ms * 1.4`.
- Clips: action + face select the clip; the frame is `(now_ms - start_ms) / frame_delay` for looping clips (walk
  frame delay scaled so one walk cycle = 2 cells of travel at speed_ms), and for one-shot clips the clip is
  fitted into `dur_ms` (attack: the atk frame at ~60% of amotion shows the damage number of the target).
- Damage numbers float up from the target when `h` arrives, timed to the attacker's atk frame when the attacker
  is visible (else immediately). Hurt flinch plays 300 ms. Dead plays once and holds the last frame.
- Lab metrics (new feel targets, add to memory when agreed): **walk smoothness**: p95 of per-frame position
  jump of a walking entity <= 1.2 x (frame_dt / speed_ms) cells (no teleports, no stalls); **animation sync**:
  the client's render clock drift vs server ticks < 1 tick p95; **attack timing**: the damage number appears
  within 1 frame of the atk frame; **fps** unchanged (p95 >= 55 with 200 animated entities).

### 1.4 Sheets (art-anim)
- **Art direction (user, 2026-11-02): everything we draw must have RO's anime feel.** Classic RO sprites are
  chibi (about 2.5 heads tall, big head and eyes, small body), clean dark outlines, soft cel shading with 2-3
  tones, bright saturated palettes (Prontera reds and golds, Poring pink), expressive hurt/dead poses, bouncy
  walk cycles, head-gear visible. Monsters are cute-or-creepy cartoon shapes, not realistic. Placeholders made
  procedurally still follow this: chibi proportions, outline + two-tone shading, saturated colours; no realistic
  or photo-derived art, no grey boxes once a sheet exists.
- Sheet format: one PNG per entity kind per action, rows = 8 directions (0..7 as above, mirroring allowed: rows
  5..7 may be absent and mirrored from 3..1), columns = frames; a JSON "act" next to it:
  `{"frame_w", "frame_h", "actions": {"idle": {"row0": 0, "frames": 4, "delay_ms": 150, "mirror": true}, ...},
  "anchor": [x, y], "atk_frame": 3}`. Loader: `client/game/art/act_sheet.gd` (pure) -> `SpriteFrames`.

## 2. Teams (one Opus sub-orchestrator each, Sonnet workers with disjoint files; branch `ro2/<name>`)
Each team: read `CLAUDE.md`, `docs/ro_contract.md`, this file, the files it owns and their tests; write a short
contract per worker; cap engine runs at 4 Godot processes or 4 pytest runs at once per team (24 cores shared by
12 teams); CPU only (the lead runs GPU soaks); commit on your branch; push when green; final report in the
`ro_contract.md` format with the exact lines for lead-owned files.

| team | owns | delivers |
|---|---|---|
| **ro-move** | `game/movement.py`, `game/monsters.py` (movement part), `net/pathing.py`, `bots/policies.py` (direction output), `game/ro_move.py` (new), tests | 1.1: 8-dir movement with speed accumulators for players and monsters, facing, 8-connected A*, bots walking 8-dir; report `DIR_DELTAS`/state lines |
| **ro-anim** | `game/ro_anim.py`, `game/view.py` (anim fields in entity_table/changed_mask), `net/protocol.py` (ents/you anim fields only; coordinate with netperf2 by touching only `entity_rows`/`ro_you`), tests, `docs/anim_wire.md` | 1.2 block + API + hooks reported as lines for combat/skills/movement (or small anchored edits in `combat.py`/`ro_skills.py` if ro-move is not touching them: say which) |
| **client-anim** | `client/game/anim/*` (new), `client/game/world/entity_sprite.gd`, `entity_layer.gd`, `entity_track.gd`, `client/game/art/act_sheet.gd`, `client/tests/lab/*` (anim metrics), tests | 1.3 playback, interpolation by speed_ms, damage numbers, flinch, lab metrics; works with generated placeholder clips until art lands |
| **art-anim** | `client/assets/anim/**`, `client/assets/slots.json` (anim entries), `client/game/art/asset_slots.gd` (anim slots), `client/tools/gen_placeholder_anim.py` (new) | 8-direction animated sheets + act JSON for 7 jobs and 20 monsters: CC0 sources where they exist (record licences), else procedurally generated chibi anime-style placeholders (outline, two-tone cel shading, big head, bouncy walk/attack/hurt frame motion) per 1.4's art direction, so the pipeline is complete; a render contact sheet looked at by the team |
| **ro-fidelity** | `game/data/*.py` (numbers), `docs/ro_fidelity.md`, `tests/test_ro_fidelity.py` | audit every table and formula against classic RO (web research allowed: irowiki, rAthena db): stats/HP/SP/ASPD/HIT/FLEE formulas, exp tables, job change rules, the skill list and their numbers, cast/after-cast delays, item and monster numbers; fix what is wrong in the data files; list what the engine must change (lines for the owners); correct the feel targets (ASPD) |
| **ro-monsters** | `game/monsters.py` (AI part; coordinate with ro-move via the `path_dir` interface: monsters AI writes `move.path_dir`, ro-move moves), `game/data/monsters.py` (modes, sight, speed_ms, respawn), tests | RO monster modes (passive/aggressive/assist/looter/immovable), random walk, sight and chase radius, return home, per-group respawn timers, Poring-style loot pickup |
| **ro-world2** | `game/world_layout.py`, `game/map_gen.py`, `game/ro_warps.py`, `game/data/terrain.py`, `docs/world_*.md`, `bench/world_scale.py` | RO zones: named rectangular zones with edge portals (prt_fild01-style), Prontera-like capital layout (streets, central fountain square, shop row), kafra teleport destinations, the **MAP 2048 build executed** on CPU with timing (and a note for the lead's GPU run), zone data in `welcome.world` |
| **ro-items2** | `game/ro_items.py`, `game/ro_drops.py`, `game/data/items.py`, `game/data/drops.py`, `game/ro_shop.py` (new), `net/kafra.py`, tests, `docs/items_wire.md` | 20-kind RO drop tables, craft items in the item table (ids 100..299 with names/icons), job equip restrictions, weapon ranges (bows 9 cells, needs ro-move's facing/range support: expose `weapon_range(world)`), NPC shops (buy/sell at shop tiles with zeny, `inputs["buy"]`/`["sell"]`), storage id validation with N_ITEMS |
| **netperf2** | `net/server.py`, `net/protocol.py` (everything except the anim fields), `net/client.py`, `bench/server_load.py`, `game/view.py` (nodes only: coordinate with ro-anim), `docs/proto_v2_notes.md`, tests | `you` fields on change, binary or msgpack snaps behind a `format` flag, nodes in the view table, drop/node kind collision fixed, 500 clients at **TICK_MS 100** with 0 overruns (change `TICK_MS` is lead-owned: report), GPU server mode (`--platform gpu` in WSL) |
| **client-feel** | `client/game/ui/hud.*`, `client/game/ui/theme/*`, `client/game/ui/minimap.*` (new), `client/game/world/walk_marker.gd`, `client/game/world/map_renderer.gd`, `client/game/input/*`, `client/game/audio/*` (new), tests | RO-style HUD (basic info window with HP/SP/exp bars, zone name on entry, hotkey bar styling, chat window), minimap with portals/kafra, click target highlight and path preview, cursor states (attack/talk/walk), camera zoom bounds, audio hooks (walk/attack/hit/level-up placeholders) |
| **bots2** | `bots/ro_*_policy.py`, `bots/ro_policies.py`, `bench/soak_*.py`, `tests/test_bots*.py`, `tests/test_ro_*_policy.py` | bots that equip looted gear, use skills by job, party up, sit to regen, enter dungeons, shop; a leveling-curve soak (ticks to base 10 / job 10 per job) compared with RO's early game; report where bots ARE |
| **ops2** | `tools/ops/*`, `docs/ops.md`, `docs/lessons_learned.md`, `.github/workflows/*` (new workflows only), kit harvest notes | ops board refreshed for 12 branches, CI job that exports the Windows build on manual dispatch (Linux templates), lessons harvested into `docs/lessons_learned.md` and a PR-ready diff for game-starter-kit |

## 3. Lead-owned, unchanged by teams
`game/config.py`, `game/state.py`, `game/step.py`, `game/combat.py` (ro-anim may make small anchored edits ONLY
for `set_action`/`note_hit` calls and must say so), `bots/runner.py`, `client/project.godot`, `client/game/main.*`,
`client/game/world/game_view.*`, `client/tests/sim/sim_suite.gd`, `.claude/*`. Report lines.

## 4. Done means (per team)
Tests green in your files and the full suite (`venv\Scripts\python.exe -m pytest tests -q` from the repo root,
or `tools/check_all.sh` from `client/` with the Windows env line), a CPU soak clean where you touched the engine
(`bench/soak_check.py --batch 4 --chunks 5 --ticks 100`), numbers before/after, a looked-at render for anything
visual, branch pushed, and the final report with the lines for lead-owned files and the next 3 defects.
