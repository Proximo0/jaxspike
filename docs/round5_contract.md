# Round 5 contract: deaths back in band, PvP on the wire, item presentation, shores and props, tick structure

Lead-owned (2026-11-03, night). Builds on `docs/round4_contract.md` (in force: load limits, shared checkout, no
git by teams, anchored edits, report lines for lead-owned files). Round 4 is on main through 334b48c: PvP in the
tick, cards applied, chibi tileset, perf cuts, balance by data, icons. Measured (GPU soak 64x1000, 209 s): kills
3.88 per bot per 1000 ticks (target met), deaths 0.06 (target 0.3-0.6: too safe), base level 1.7 after 1000
ticks, 2 cards per 31815 kills.

| team | owns | delivers |
|---|---|---|
| **ro5-balance** | `bots/ro_nav_policy.py`, `bots/policies.py`, `bots/ro_stats_policy.py` (DEX in the allot plan), `game/ro_spawns.py`, `bench/soak_levels.py` | deaths 0.3-0.6 with kills kept 3-5: relax the fight-cost margins or send bots one band out one level earlier (measure both), bots that raise DEX so hit chance grows, Novice starter gear reported as `game/state.py`/`net/characters.py` lines (Knife + Cotton Shirt, equipped), the leveling curve (ticks to base 10 and 20 at `--batch 1 --density`) and an `EXP_RATE` recommendation with the number. |
| **ro5-pvp2** | `game/ro_pvp.py`, `game/combat.py` (PvP path only), `net/protocol.py`/`net/server.py` (pvp rows), `net/char_fields.py` (persist pvp_kills/pvp_deaths), `client/game/world/entity_layer.gd` (PK colour from the wire), `client/game/ui/hud.gd`, `bots/ro_pvp_policy.py` | other players' PK flag on the wire (an entity flag `k`, on change only), an explicit PvP target through `inputs["target"]` when it names a player (click to duel), PvP kill to the top cumulative damager, the LUK crit shield, kills/deaths saved with the character; soak line with PvP kills per duellist. |
| **ro5-items** | `client/game/ui/inventory_panel.*`, `client/game/ui/hud_text.gd`, `client/game/ui/shop_panel.*`, `client/game/ui/card_album.*`, `client/game/art/asset_slots.gd` (item icon keys), `game/data/items.py` (display names only, additive column) | proper item names everywhere ("+4 Sword", "Gildscarab Card", capitalised, refine and slots shown), the new 64 px icons resolved for every item id (no loader errors), a card-only album key that does not collide with movement (keep Alt+A, add a HUD button), the shop panel using the icons, tooltips consistent across inventory, shop and album. |
| **ro5-tiles2** | `client/tools/art/tiles/*`, `client/assets/tiles/**`, `client/game/world/map_renderer.gd` (autotile lookup only; the PropLayer switch belongs to ro3-art), `client/game/world/view_math.gd` autotile helpers, `client/tests/unit/test_map_*.gd` | 8-neighbour (47-tile blob or corner-overlay) autotiles for water, road and wall so shores and corners stop staircasing; furnace and fire props bound to node kinds 406/407 (an `EntitySprite` texture from the props atlas: send the `entity_layer.gd` line to the lead); dungeon interiors and the four other towns looked at and fixed where tiles clash; map build still under 300 ms. |
| **ro3-art** (continuing) | `client/game/world/prop_layer.gd` (new), `client/tools/art/*`, `client/assets/anim/**`, `client/assets/icons/**` | y-sortable props (characters walk behind trees and the fountain), then whatever raises the visual bar most. |

Lead at integration: one spatial structure per tick in `game/step.py` is deferred to a quiet-box measurement
(`bench/bench_tick.py --batch 4` with nothing else running); the `MAX_MONSTERS` decision likewise.

Done means, per team: tests green in your own files (one process at a time), numbers before/after, a looked-at
render for anything visual, no git, final report with exact lines for lead-owned files and the next 3 defects,
hand-back note `ro5-<team>-handback.md` in the shared notes folder.
