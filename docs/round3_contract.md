# Round 3 contract: balance, RO engine fidelity, tick cost, the dead on the wire, exp and parties, polish

Lead-owned (2026-11-02, evening). Builds on `docs/ro_contract.md`, `docs/m0_brief.md` and `docs/round2_contract.md`
(all still in force). Round 2 is on main (35a31a2): 8-direction accumulators, the anim block, monster modes, classic
RO tables, zones and portals, drop tables and shops, bots that equip/cast/party/dive, binary snaps, anime sheets,
the RO HUD. This round fixes what the integration measured, ranked in the lead's note
`lead-2026-11-02-round2-integration.md` (shared notes folder). Same mode as round 2: **shared checkout, no git by
teams, disjoint ownership, anchored edits with a re-read right before every edit of a shared file**, Opus
sub-orchestrators with Sonnet workers, up to 200 workers in all, CPU only (the lead runs the GPU soaks).

## 0. Measured state (GPU soak 64 worlds x 1000 ticks, 2026-11-02 16:00)
- kills 2.2 / bot / 1000 ticks, **deaths 1.1** (target 0.3-0.6), 8192 bots, 377 s wall (was 128 s before round 2:
  the tick costs ~3x; `net/server.py --bench-ticks 200` gives p50/p95 per tick at 512 viewers on CPU).
- Level-1 Novice: 40 hp, 10 sp; swings/s still `1500 + (aspd-150)*45` ms (E1 open); HIT/FLEE without level (E3).
- Wire: entities carry `a d t m s h`; dead entities drop out of `vis` (no dead clip for other players); misses
  carry no `h`; no sit or pickup input; `you.range` and shops exist but no shop panel in the client.
- Exp: paid only to a player within 1 cell of the corpse; `ro_party.share_exp` is never called; skill kills pay no
  zeny; cards drop at 1 per mille (10x classic).

## 1. Rules of this round
- A rested entity steps on its first intent (`game/ro_move.py` acc = ms since the last step, capped at one cost).
- Phase order (`game/step.py`): movement, warps, monsters, skills, combat, loot, stats, items, shop, survival,
  craft, party, mvp, kafra, anim. New phases are reported as lines for the lead, never inserted by a team.
- Balance changes go through data (`game/data/*.py`, `game/ro_spawns.py`) with the soak numbers before/after.
- Anything that can be an array op in the tick is one. The server's Python stays irregular work only.
- Every lab/soak number is measured, not estimated; say what you could not measure.

## 2. Teams (one Opus sub-orchestrator each, Sonnet workers with disjoint files)
Each team: read `CLAUDE.md`, `docs/ro_contract.md`, `docs/round2_contract.md` 0-1, this file, the lead's
integration note, the files it owns and their tests; cap runs at 4 pytest or 4 Godot processes at once per team;
no git; final report in the `ro_contract.md` format with exact lines for lead-owned files and the next 3 defects;
hand-back note in the shared notes folder.

| team | owns | delivers |
|---|---|---|
| **ro3-engine** | `game/ro_stats.py` (`derived`, tables, regen, death penalty: NOT the exp payout function, see ro3-exp), `game/combat.py` (formulas: HIT/FLEE, DEF, swing speed; the lead keeps the phase shape), `game/ro_skills_effects.py` (buff effects, heal), `game/data/stats.py` (per-job HP/SP/ASPD tables if columns are missing), `net/char_fields.py` (pad 33->42 skill levels), `docs/ro_fidelity.md` section E, `tests/test_ro_fidelity.py` (flip the 3 xfails) | E1-E11 from `docs/ro_fidelity.md`: swings/s = 50/(200-ASPD) with ASPD by job and weapon class, exact per-job HP/SP, HIT = level+DEX and FLEE = level+AGI with monster AGI/DEX columns, flat VIT defence, Increase AGI as move speed (`move.speed_ms`), Heal formula, death penalty, HP regen by max HP (sitting doubles it once ro3-wire adds `sit`), old saves padded. Numbers before/after on `bench/soak_check.py`. |
| **ro3-balance** | `game/ro_spawns.py`, `game/data/monsters.py` (spawn density / band columns only; coordinate with ro3-engine for AGI/DEX columns by appending, never re-ordering), `bench/soak_levels.py`, `bench/soak_check.py` (report columns only), `bots/policies.py` (retreat and rest thresholds as % of max hp) | deaths 0.3-0.6 and kills 3-5 per bot per 1000 ticks at base level 1-10 with 128 bots per world: band-0 density (73 monsters per world is too few), spawn mix by band, retreat thresholds, `EXP_RATE` lever in `game/config.py` reported as a line (classic 1x stays the default), a leveling curve table (ticks to base 10/20) from `soak_levels`. |
| **ro3-perf** | `game/ro_anim.py`, `game/ro_monster_modes.py`, `game/view.py`, `game/ro_shop.py`, `bench/bench_view.py`, `bench/bench_tick.py` (new: per-phase timing under jit) | tick back toward 2 ms per world-tick on CPU at batch 4 (measure per phase first, then cut: fewer gathers, int16/int8, no per-entity MAP-sized ops, shared occupancy, one spatial grid per tick reused by monsters/combat/skills). Keep every test green; numbers per phase before/after; no behaviour change (soak kills/deaths within noise). |
| **ro3-wire** | `net/protocol.py`, `net/server.py` (snapshot/input parts), `game/loot.py` (pickup hook), `game/survival.py` (sit regen), `game/ro_anim.py` (ONLY the `pickup`/`sit` set_action calls: anchored, re-read, tell ro3-perf), `client/game/net/*`, `client/game/world/entity_layer.gd` (dead clip + miss text only), `docs/anim_wire.md` | the dead stay visible for 3 s with `a = 7` (vis includes the dying for that window, then `gone`), misses send `h = 0` and the client shows "Miss", new input `sit` (bool; lead line in `state.empty_inputs`) with the sit clip and doubled regen, pickup plays the pickup clip, warp camera snap flag on `you` (`warped`), the client sends binary snaps by default. |
| **ro3-exp** | `game/ro_stats.py` exp payout function only (anchored; tell ro3-engine), `game/ro_mvp.py` (damage credit), `game/ro_party.py` (`share_exp` wiring), `game/ro_skills_effects.py` damage reporting through `ro_mvp.record_damage` (anchored; tell ro3-engine), `game/loot.py` zeny credit (anchored; tell ro3-wire), tests | exp to the killer by damage credit (ranged and skill kills pay), party exp share per `ro_party.share_exp` (even split, +10% per member as classic, within the same zone), skill kills pay zeny, MVP top damager cumulative. |
| **ro3-items3** | `game/data/drops.py`, `game/ro_drops.py`, `game/ro_items.py`, `game/data/items.py`, `client/game/ui/shop_panel.*` (new), `client/game/ui/inventory_panel.*` (drop/sell affordances), `docs/items_wire.md` | drop chances per 10000 (cards 1 = 0.01%), ground drops keep count/refine/cards, coins carry the rolled zeny, a shop panel (buy/sell with prices after Discount/Overcharge, opens near a shop tile, `buy`/`sell` messages), equipment compare tooltips. |
| **ro3-world3** | `game/ro_craft_nodes.py`, `game/ro_move.py` (ONLY `_first_claims`/`_occupied` claim keys: int32 overflow at MAP 2048; tell ro3-perf), `game/ro_warps.py`, `game/world_layout.py`, `game/map_gen.py`, `bench/world_scale.py` | ore nodes off the zone split lines, MAP 2048 tick unblocked (claim keys that fit, measured in a scratch copy at 2048 on CPU: build time and tick time), zone-aware kafra teleport (report the input lines), portal tiles on the client minimap data (`welcome.world.portals` already sent). |
| **ro3-client** | `client/game/ui/*` (except shop/inventory panels: ro3-items3), `client/game/world/game_view.gd` layout constants ONLY via reported lines, `client/game/input/*`, `client/game/world/hud_text.gd`, `client/tests/sim/sim_feel.gd`, `client/tests/unit/test_feel_*.gd` | one status window (merge the Status panel into the basic-info window: stats and allot buttons under it; `C` toggles the stat section), chat notices once, F2 `camera_zoom` live again, warp camera snap on `you.warped`, skill bar greys out during `after_delay`, cast bar reads `cast_left`, damage numbers and "Miss" styled as RO (white normal, yellow crit, red on you). |
| **ro3-anim3** | `client/game/anim/*`, `client/assets/anim/**`, `client/assets/slots.json` (anim entries), `client/tools/gen_placeholder_anim.py`, `client/tests/lab/lab_anim_feed.gd`, `client/tests/unit/test_anim_*.gd` | sit and pickup clips played from `a = 2/3`, the dead clip from `a = 7` then fade, attack sync with the atk frame at 60% of `m` (lab metric already there: keep it 0 frames), hurt flinch without position pops, orc_warrior hurt frames fixed (semi-transparent pink), sheets for the two MVPs and the shop/kafra NPC sprites in the anime chibi style (`docs/round2_contract.md` 1.4), `anim_fps` p95 >= 55 headless at 200 changed rows (configure skipped when unchanged, pooled dicts). |
| **ro3-bots3** | `bots/ro_nav_policy.py`, `bots/ro_party_policy.py`, `bots/ro_skills_policy.py`, `bots/ro_items_policy.py`, `bots/ro_craft_policy.py`, `tests/test_bots_ro.py`, `tests/test_ro_*_policy.py` | bots cross portals (zone-aware descent on `dist_dungeon`/`dist_town`), enter dungeons from base 10 (seen in a 16k-tick CPU soak: report the first dive tick), form parties of 2-4 and share exp once ro3-exp lands, gatherers level, use the shop (`buy` potions) and `sit` to rest; census per zone every 1000 ticks. |
| **ro3-ops3** | `tools/ops/*`, `docs/ops.md`, `docs/lessons_learned.md`, `docs/kit_harvest.md`, `.github/workflows/*`, `.github/actions/*` | main CI green (the engine job had 2 failures on the contract commit; find and fix the cause in the workflow or tests, not by skipping), the Windows export job verified by a manual dispatch (the lead pushes; tell the lead what to click), ops board refreshed for round 3 with the GPU soak numbers, kit harvest opened as a PR to `Proximo0/game-starter-kit` (the user allowed GitHub). |

netperf2 (round 2) continues as **ro3-netperf**: finish the bench matrix, then 500 clients at 100 ms on the pipelined
server (CPU) with the numbers, bandwidth per client at 200 and 500, and `--bench-ticks` on GPU (ask the lead to run
it). Owns `bench/server_load.py`, `net/client.py`, `docs/proto_v2_notes.md`; `net/server.py` tick-loop parts
(coordinate with ro3-wire by file region: ro3-wire owns `_snapshot_view`/`_on_message`/protocol, ro3-netperf the
tick loop and benches).

## 3. Lead-owned, unchanged by teams
`game/config.py`, `game/state.py`, `game/step.py`, `bots/runner.py`, `client/project.godot`, `client/game/main.*`,
`client/game/world/game_view.*`, `client/tests/sim/sim_suite.gd`, `.claude/*`. Report lines. `game/combat.py` is
lent to ro3-engine this round (formulas only; keep the phase signature and the `ro_anim.on_combat` call).

## 4. Done means (per team)
Tests green in your files and the full suite (`venv\Scripts\python.exe -m pytest tests -q` from the repo root, or
`tools/check_all.sh` from `client/` with the Windows env line), a CPU soak clean where you touched the engine
(`bench/soak_check.py --batch 4 --chunks 5 --ticks 100`), numbers before/after, a looked-at render for anything
visual, and the final report with the lines for lead-owned files and the next 3 defects. Nothing committed: the
lead commits by name from your report.
