# Round 4 contract: balance to target, PvP in the tick, rare cards that feel rare, tick cost, town tiles

Lead-owned (2026-11-03, evening). Builds on `docs/ro_contract.md`, `docs/round2_contract.md` 0-1 and
`docs/round3_contract.md` (all in force). Round 3 is on main (ad46eff, 7c4076e, ab0fb93): RO formulas E1-E11,
damage-credit exp and party share, the dead and misses on the wire, sit/pickup, per-10000 drops, shop panel, one
status window, the Aurelune rename (`docs/naming.md`: the world is Aurelune, towns Solmera/Lunavell/Pinehollow/
Tidewick/Kharsand, the Waybell service, original monster and item names; `tests/test_naming.py` guards it) and 27
chibi sprite kinds rendered by `client/tools/art`. The user's directions for this round: **graphics, art and
animation come first; the JAX core exists to raise complexity and scale; PvP is a core feature the JAX core should
handle well; the rare-card system should feel rare and valuable; keep original names.**

## 0. Measured state (GPU soak 64 worlds x 1000 ticks, 2026-11-03 19:20)
- kills **0.95** / bot / 1000 ticks (target 3-5), deaths 0.43 (target 0.3-0.6, met), base level mean 1.1 after
  1000 ticks; 198 s wall (was 377 s in round 2: the tick halved), still ~1.5x round 1.
- `bench/bench_tick.py --batch 4` gives per-phase ms (ro3-perf note: ro_move sorts ~2.3 ms, drop rolls ~2 ms, combat
  2.2 ms per world-tick; MAX_MONSTERS 4096 vs 1024 populated).
- Wire: no crit column (no yellow numbers), kills credit (lowest adjacent hitter) differs from exp credit (top
  cumulative damager).
- Client: terrain is flat colours; sprites are the only art; no PvP concepts anywhere; cards work (slots, compound,
  effects, 1/10000 drops) but nothing announces or shows them.

## 1. Load limits (CLAUDE.md "Load limits", the user's rule)
At most 6 team sessions at once with at most 10 workers each (60 agents in all); ONE heavy process per team
(pytest on your own files, a Godot run, a Blender render); no full suites, `check_all` or soaks (the lead runs them
at integration); the guard hook blocks heavy commands when Claude's processes would pass 60% of RAM or under 4 GB
stays free: a block means wait for your own process, never retry in a loop. Shared checkout, no git, re-read a
shared file right before each edit, anchored edits. Report lines for lead-owned files.

## 2. Teams

| team | owns | delivers |
|---|---|---|
| **ro4-balance** | `game/ro_spawns.py`, `game/data/monsters.py` (band/spawn columns), `game/data/stats.py` (exp tables untouched; a documented `EXP_RATE` lever only), `bots/policies.py`, `bots/ro_nav_policy.py` (target choice: fight what you can hit), `bench/soak_levels.py`, `bench/soak_check.py` (report columns) | kills 3-5 per bot per 1000 ticks at base 1-10 with deaths kept 0.3-0.6, by data: band-0 monster levels and density (Novices must have 80%+ hit chance on their first prey), spawn mix, bots fighting only kinds within their hit/flee band and retreating by % hp; a leveling curve table (ticks to base 10 and 20) measured with `soak_levels` at `--batch 1`; `EXP_RATE` as a reported `game/config.py` line (default 1). Never soften the RO formulas. |
| **ro4-pvp** | new `game/ro_pvp.py`, `game/combat.py` (lent: player-vs-player damage path, crit column for the wire, kills credit from `ro_mvp.credit_of`), `game/world_layout.py` (PvP flags per zone, one arena zone inside Solmera's walls or a field corner), `game/ro_anim.py` (crit column, anchored), `net/protocol.py`/`net/server.py` (`pvp` toggle message, `you.pvp`, `h < 0` = crit on the wire), `bots/ro_pvp_policy.py` (new: bots that duel in the arena for the soak), client: `client/game/world/entity_layer.gd` (name colour for PK-enabled players, crit yellow numbers already decoded), `client/game/ui/hud.gd` (PvP badge) | PvP as array ops in the tick: a zone PvP flag (arena always on; fields off in v1; a `pvp_zone` table), a per-player PK toggle (`pvp` one-shot input, off in towns), player-vs-player damage with the RO formulas (HIT/FLEE/DEF, cards and elements apply), kill credit and a `pvp_kills`/`pvp_deaths` pair in the stats block, no exp loss for PvP deaths, respawn as usual; crit on the wire as negative `h`; bots duelling in the arena so the soak reports PvP kills per 1000 ticks and the tick cost of the PvP path (must be within noise). |
| **ro4-cards** | `game/ro_items.py` (card tables are data; card effects unchanged), `game/data/items.py` (card rows: names and effects per `docs/naming.md` monsters), `net/protocol.py` (`card` server message: a world-wide announcement when a card drops; `welcome.cards` table), `net/server.py` (announce from the loot pull), client: new `client/game/ui/card_album.*` (A key: every card, owned ones lit, effect text), `client/game/ui/inventory_panel.*` (card icon in slots, compound tooltip), `client/game/world/entity_layer.gd` (a sparkle on a card drop entity), `client/assets/cards/**` (card art in the anime look: a framed portrait per monster rendered with `client/tools/art` from the same models; coordinate with ro3-art by message, they own the pipeline), `bench/soak_check.py` (cards found per 1000 kills column) | cards that feel rare: an announcement line in every client's chat when one drops ("Someone found a Blobbin Card!"), a card album panel, card icons and tooltips, card art per monster, MVP cards for Gildscarab and Scorchstripe with strong effects, a soak column. |
| **ro4-perf** | `game/ro_move.py` (claim sorts), `game/loot.py` + `game/ro_drops.py` (compact died monsters before rolling), `game/spatial.py` (packed tables), `game/monsters.py` (one occupancy), `game/ro_stats.py` `derived` memo per tick (anchored; tell ro4-balance), `bench/bench_tick.py`, `bench/bench_view.py` | the per-phase "after" table at batch 4 on a quiet box; cut ro_move sorts (inverse permutation by scatter), drop rolls only over died rows, combat lookups on packed tables, one occupancy per tick; target 1.2 ms per world-tick on CPU at batch 4 (round 1 level) with behaviour unchanged; report the `MAX_MONSTERS` decision (2048 vs 4096) as a `game/config.py` line with numbers. |
| **ro4-tiles** | new `client/assets/tiles/**` and `client/tools/art/tiles/*` (reuse ro3-art's Blender setup by message; ro3-art owns `client/tools/art/*.py`), `client/game/world/map_renderer.gd`, `client/game/world/view_math.gd` (terrain colours stay as the fallback), `client/tests/unit/test_map_*.gd`, `client/assets/LICENSES.md` rows | the world drawn in the same anime chibi look as the characters: a tileset for the 13 terrain ids (grass, road, floor, temple, water, tree, wall, rock, warp, waybell, station, cave, shop) with autotile edges for water/road/wall, props on temple/waybell/shop/station tiles (fountain, bell, awning, furnace), rendered from free CC0 models or hand-drawn by the pipeline, drawn by `map_renderer.gd` from the map once at login (one TileMap or a baked texture per chunk; keep draw time within the lab's `map_draw_ms`); a looked-at windowed render of Solmera. |
| **ro3-art** (continuing, session already running) | `client/tools/art/*`, `client/assets/anim/**`, `client/assets/slots.json` anim entries | per-job hair, side-view eyes, mirrored weapon hand, the four weak faces, then NPC and effect sprites (cast circle, hit spark, level-up) in the same look. |

## 3. Lead-owned, unchanged by teams
`game/config.py`, `game/state.py`, `game/step.py`, `bots/runner.py`, `bots/ro_policies.py` (one-line additions
reported), `client/project.godot`, `client/game/main.*`, `client/game/world/game_view.*`, `client/tests/sim/
sim_suite.gd`, `.claude/*`. `game/combat.py` is lent to ro4-pvp this round.

## 4. Done means (per team)
Tests green in your own files (one pytest or Godot run at a time), numbers before/after where the row names them,
a looked-at render for anything visual (review PNGs in `client/assets/anim/_review/` or `client/assets/tiles/_review/`,
git-ignored), no git, final report in the ro_contract.md format with exact lines for lead-owned files and the next
3 defects, hand-back note `ro4-<team>-handback.md` in the shared notes folder. The lead runs the full suite,
`check_all`, the soaks and the build at integration and posts the build to the user.
