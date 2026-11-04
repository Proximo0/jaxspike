# Round 6 contract: starter kit with a retune, PvP chase and skills, rich tooltips, buildings

Lead-owned (2026-11-03, late night). Builds on `docs/round5_contract.md` (in force: load limits, shared checkout,
no git by teams, anchored edits, report lines for lead-owned files). Round 5 is on main through dede132.
Measured (CPU soak 4x500): kills 5.16, deaths 0.18 per bot per 1000 ticks, base level 2.6 after 500 ticks at
EXP_RATE 3, 2.8 PvP kills per duellist, invariants held.

| team | owns | delivers |
|---|---|---|
| **ro6-balance** | `bots/ro_nav_policy.py`, `bots/ro_stats_policy.py`, `bots/ro_items_policy.py` (potions), `game/ro_spawns.py`, `bench/soak_levels.py`, `tests/test_ro_*_policy.py`, `tests/test_ro_spawns.py` | the Novice starter kit (Knife + Cotton Shirt equipped) reported as exact `game/state.py` and `net/characters.py` lines AND the retune that goes with it in the same report (BAND1_LEVEL earlier, COST_SAFETY scaled by level so deaths stay 0.3-0.6 past 1000 ticks, bots that drink potions instead of sitting when they hold them), measured with the kit patched in through a scratch harness: kills 3-5, deaths 0.3-0.6 at 300 and 1000 ticks, seeds 0 and 1. |
| **ro6-pvp3** | `game/ro_pvp.py`, the PvP path in `game/combat.py`, `game/ro_skills_effects.py` (player targets), `game/ro_skills.py` (target validation for players: anchored), `bots/ro_pvp_policy.py`, `tests/test_ro_pvp.py` | auto-chase: a locked duel target out of range writes a move intent toward it (through the same `path_dir` interface monsters use, within the PvP area, stopping at weapon range); PvP skills: Bash, bolts, Double Strafe and heals resolve on player targets where the pair is allowed; the per-target ledger decays after 10 s without hits; soak line with PvP kills and skill kills per duellist. |
| **ro6-ui** | `client/game/ui/inventory_panel.*`, `client/game/logic/item_text.gd`, `client/game/ui/shop_panel.*`, `client/game/ui/card_album.*`, `client/game/ui/hud.gd`, `client/game/ui/theme/*`, `client/tests/unit/test_ui_*.gd`, `client/tests/unit/test_item_text.gd` | themed rich tooltips (`_make_custom_tooltip`: icon, coloured refine, card portraits, gear bonus text from a `bonus_text` field added to `welcome.items` by an anchored one-line addition in `game/data/items.py` `table_for_client`), a consistent panel look (titles, close buttons, drag), the HUD notice line and chat System tab de-duplicated, the PvP badge and the Cards button placed so nothing overlaps at 1920x1080 and 1280x720. |
| **ro3-art** (continuing) | `client/tools/art/*`, `client/assets/anim/**`, `client/assets/icons/**`, `client/assets/tiles/props.png` + the prop rows of `tiles.json` (coordinate with the lead) | building and roof props for the decor wall blocks of Solmera and Kharsand (houses with awnings, a well, lamp posts) in the chibi look; then skill effect sprites per skill (bolt, arrow shower, magnum break). |

Lead at integration: quiet-box per-phase bench and `MAX_MONSTERS` once the playtest server is down and no
worker runs; GPU soak likewise. Done means, per team: tests green in your own files (one process at a time),
numbers before/after, a looked-at render for anything visual, no git, final report with exact lines for
lead-owned files and the next 3 defects, hand-back note `ro6-<team>-handback.md` in the shared notes folder.
