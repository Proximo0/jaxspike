# M3 items: wire additions (protocol v2)

Spec for the lead and the net worker. Nothing here is implemented in `net/` yet. The envelope is unchanged: every
message has `"v"` and `"t"`, JSON text frames, unknown `t` ignored, wrong types give `error` "malformed message".
Item ids are the `game/data/items.py` ids (`N_ITEMS = 300`, the full id space; 0 = none; see round 2 below). Slot numbers are inventory slot indices
`0..INV_SLOTS-1` (16).

## `welcome` additions

```
"items":  game.data.items.table_for_client()     list of N_ITEMS (300) dicts, index == item id:
          {id, name, kind, cls, atk, def, matk, element, slots, weight, buy, sell, heal_hp, heal_sp,
           weapon_level, stackable, bonus, bonus_value, card_bonus, card_value}
"drops":  game.data.drops.table_for_client()     list per monster kind:
          {kind, drops: [{item, chance, count}], zeny_min, zeny_max}      chance is per 10000 (round 3; was per mille)
```
Both are static, built once per server (like the maps) and sent only in `welcome`. `kind` is the item kind id
(0 none, 1 etc, 2 consumable, 3 weapon, 4 armor, 5 card); `cls` is the weapon class for weapons and the slot type for
armor and cards (see `items_brief.md`); `bonus`/`card_bonus` are `BONUS_*` ids 0..13.

## `snap.you` additions

```
"zeny":  int                                      0..2_000_000_000
"inv":   [{"item": int, "count": int, "refine": int, "cards": [int, int, int, int]} x INV_SLOTS]
"equip": [int x 8]                                inventory slot index worn in EQ_WEAPON, HEAD, BODY, SHIELD,
                                                  GARMENT, SHOES, ACC1, ACC2; -1 = nothing
```
`inv` replaces the old flat list of ids; always exactly `INV_SLOTS` rows, an empty slot is
`{"item":0,"count":0,"refine":0,"cards":[0,0,0,0]}`. Source arrays: `players.inv` (item), and `world["items"]`
`inv_count`, `inv_refine`, `inv_cards`, `equip`, `zeny`. They join the tick's single host pull for the viewer rows
(one numpy transfer, no per-client device reads). Only the viewer's own row is sent; other players' gear is not
on the wire in M3. Ground drops keep `{kind:"d", item, pos}` on the wire; the engine-side drop fields (count, refine, cards) are
round 3 (below) and are not sent.

## Client messages (all C->S)

| t | fields | meaning |
|---|---|---|
| `use_item` | `slot` | drink/eat a consumable, or compound a card into the worn item it fits |
| `equip` | `slot` | toggle: wear the item (replacing the occupant of its equip slot) or take it off |
| `refine` | `slot` | upgrade the worn-or-carried weapon/armor in `slot` (pays zeny and one ore; may destroy it) |
| `drop` | `slot` | put one unit (whole item for gear) on the ground at the player's tile |

`validate_client` accepts them when `slot` is an int (not bool); any int is passed through, because the engine
ignores values outside `0..INV_SLOTS-1` (and slots that do not hold a fitting item). Non-int `slot`: `error`
"malformed message" and close, like every other bad type.

## Input mapping (server side)

The engine reads four `int32 [MAX_PLAYERS]` input arrays, `use_item`, `equip`, `refine`, `drop`, `-1` = none
(`ro_items.empty_item_inputs()`). They are **one-shot**, like the `eat` tap: `_on_message` stores the latest slot per
player in a pending array (a later message in the same tick overwrites the earlier one, one action per kind per
tick); `tick_once` builds `{**inputs, "use_item": pend_use, "equip": pend_equip, "refine": pend_refine,
"drop": pend_drop}` for exactly one `step`, then resets the pending entries to `-1`. A human-owned slot with no pending
message has `-1`; bot-owned slots take their values from `bots.ro_items_policy.item_inputs` instead.
Human `use_item` etc. do not cancel a `walk_to` path (they are not movement).

Order inside the tick is fixed by `ro_items.phase`: normalize, use_item, equip, refine, drop, normalize. So a client
that sends `equip` and `refine` for the same slot in one tick sees the equip happen first. Failure feedback is
implicit in the next `snap` (item gone after a failed refine, same counts when refused); a `notice` string may be
added later by the server by diffing `inv`/`zeny`.

## Round 2 (ro-items2) additions

### Item id space
`game/data/items.py`: ids `0..N_BASE_ITEMS-1` are RO items (93 today: the M3 73, 17 drop-table etc items and cards,
ro-fidelity's Cutter, Fly Wing, Butterfly Wing), ids `100..299` are craft materials and products with craft's own
ids (`game/data/craft.py`, 100..105 and 200..208 today), every other id is a reserved row (`name ""`, `kind 0`).
`N_ITEMS == config.ITEM_ID_LIMIT == 300`; every column is indexed over the whole space, so a craft id in an inventory
reads its own row (before, it clamped onto the Willow Card row and stacked). Craft rows are NOT stackable (ro_craft
counts slots per recipe input). `welcome.craft_items` stays for older clients; `welcome.items[id]` now names them too.

`welcome.items[id]` gains:
```
"icon": str        slots.json key: item.dagger/sword/bow/staff/mace/axe, item.hat/armor/shield/garment/boots/ring,
                   item.card, item.potion_red/potion_blue, item.meat, item.gold_coin, item.ore/herb/fish, or
                   "item.etc" (no art yet: art team); "" for reserved ids
"jobs":  [int]     job ids (game/data/stats.py: 0 Novice .. 6 Merchant) allowed to wear it; all 7 for non-gear
```
`welcome.drops`: 20 rows (one per monster kind of `game/data/monsters.py`), up to 8 entries each, unused entries
left out; the last entry of every row is the kind's card at 1 per mille.

### Drops and zeny (engine, already in the tick through `loot.loot`)
A kill rolls the kind's row (independent per-mille rolls). Zeny (`2*level..5*level`, MVPs `100*level..200*level`)
goes straight to the killer's `items.zeny` (killer = `world["mvp"]["top_damager"]`, if logged in); a kill nobody is
credited with (skill-only kills today: ro_skills_effects does not call `record_damage`) leaves a gold coin on the
ground as before. Clients see zeny rise in `snap.you.zeny` without a coin drop.

### Job equip restrictions
`items.EQUIP_JOBS bool [N_ITEMS, N_JOBS]` (classic pre-renewal classes: daggers all but Acolyte, swords
Novice/Swordsman/Thief/Merchant, bows Archer/Thief, staves Novice/Mage/Acolyte, maces Novice/Swordsman/Acolyte/
Merchant, axes Novice/Swordsman/Merchant; Battle Axe Swordsman/Merchant, Wand Mage, Helm and Shield
Swordsman/Acolyte/Merchant, Buckler/Mantle/Boots all but Novice). `equip` silently refuses a forbidden item (the
next snap shows nothing worn); taking an item off is always allowed, so a job change never traps gear.

### Weapon range
`ro_items.weapon_range(world) -> int32 [MAX_PLAYERS]`: 1 for melee and bare hands, 9 for bows
(`items.WEAPON_RANGE[id]`). Combat still hits at distance 1 until the lead/ro-move wire it (report line).
Suggested `snap.you` field: `"range": int`.

### NPC shops (`game/ro_shop.py`, `phase(world, inputs)`, not yet in `step.py`)
A living player trades when it stands on or next to a `T_SHOP` tile (terrain id 12, ro-world2's shop rows; the
kafra tile and its neighbours if the terrain module has no T_SHOP) inside a town box. Stock is per town
(`ro_shop.SHOP_STOCK [5, 16]`, `ro_shop.stock_for_client()`).

`welcome` addition:
```
"shops": ro_shop.stock_for_client()     [{"town": int, "name": str, "items": [item id, ...]}] per town
```
Client messages (C->S), one-shot like `use_item`:

| t | fields | input | meaning |
|---|---|---|---|
| `buy` | `item` (int) | `inputs["buy"]` int32 [MAX_PLAYERS], item id, `0` or `-1` = none | buy one unit from the town's stock |
| `sell` | `slot` (int) | `inputs["sell"]` int32 [MAX_PLAYERS], inventory slot, `-1` = none | sell one unit of a stack, or the whole item; worn items refused |

`validate_client`: int, not bool, like `slot`. Same pending-array pattern as the M3 messages (latest per tick wins,
reset to none after one step). Order inside the phase: sell, then buy. Prices: buy `BUY * (100 - discount) // 100`
(at least 1), sell `SELL * (100 + overcharge) // 100` per unit, with discount/overcharge `min(5 + 2*level, 24)` %
from learned Discount / Overcharge. A buy needs the zeny and a stack of the same stackable id below MAX_STACK or a
free slot. Refusals are silent; the server may diff `zeny`/`inv` for a notice ("not enough zeny", "inventory full",
"no shop here").

## Round 3 (ro3-items3) additions

### Drop chances per 10000
`game/data/drops.py` `DROP_CHANCE` is per 10000 (rAthena's rate column, docs/ro_fidelity.md section 2): every card is
`1` = 0.01 % (classic; was 1 per mille, 10x). `ro_drops.roll_drops` rolls `U[0,1) * 10000 < chance`.
`welcome.drops[k].drops[i].chance` is per 10000 too (no client reads it yet).

### Ground drops keep count, refine and cards
The drops block grows three fields (lead lines for `game/state.py`, fresh arrays from `ro_drops.drop_fields()`):
```
"count":  int32 [MAX_DROPS]               units in the pile (a gold coin pile's count is its zeny)
"refine": int32 [MAX_DROPS]               refine level of a dropped weapon/armor
"cards":  int32 [MAX_DROPS, MAX_CARDS]    compounded card ids, 0 = empty socket
```
Every writer spreads `**drops` so the fields survive (`ro_drops.place_drops`, `ro_items._drop`,
`ro_monster_modes._place`); each also works without the fields (one unit, no refine, no cards) until the lead adds
them, by a Python check on the dict structure, never on values. Player `drop` of a +5 slotted sword puts that sword
on the ground; `ro_drops.pickup(world) -> (world, picked)` replaces `loot._pickup` (ro3-wire's line:
`world, picked = ro_drops.pickup(world)`), copying count/refine/cards into `items.inv_count/inv_refine/inv_cards`;
`picked` is bool [MAX_PLAYERS] for the pickup clip. A looter monster that picks a pile up and spills it on death
keeps one unit (ro3-perf's `_place` sets count 1): defect, see the hand-back.

### Coins carry the rolled zeny
`items.ZENY_PER_COIN = 1`: a gold coin stack's count IS its zeny. A kill nobody is credited with leaves one coin pile
with `count = rolled zeny` (2..5 x level; MVPs 100..200 x level); `ro_items._normalize` turns a picked-up pile into
exactly that zeny (no flat 10 any more). Credited kills still pay `items.zeny` directly (loot.py, ro3-exp).

### Client shop panel (`client/game/ui/shop_panel.gd/.tscn`)
No key: it opens when `you.pos` is on or next to (Chebyshev distance 1, the engine's `_NEAR`) a tile of
`welcome.world.shops` and shows that town's `welcome.shops[town]` stock. Prices mirror `ro_shop`: buy
`max(BUY * (100 - d) // 100, 1)`, sell `SELL * (100 + o) // 100` with `d`/`o` = `min(5 + 2 * level, 24)` from the
player's learned Discount / Overcharge (`you.learned` pairs, skill ids looked up by name in
`welcome.skills.names`). Buy sends `{"t": "buy", "item": id}`, Sell (non-empty, not-worn slots) sends
`{"t": "sell", "slot": slot}`; zeny and the sell list refresh from `snap.you`. Close dismisses the panel until the
player leaves the shop's range. Lead lines in `client/game/world/game_view.gd`: `PANEL_SCENES` gets
`"res://game/ui/shop_panel.tscn",` and `PANEL_POS` gets `"ShopPanel": Vector2(1300, 250),`.

### Inventory panel: compare tooltips, Sell and Drop affordances
Hovering an unworn weapon/armor shows `ATK 25 (worn Knife 17, +8)` / `DEF 4 (worn Hat 2, +2)` and
`Slots 3 (worn 1)` against the item in the equip slot it would take (refine included: weapon atk + refine *
(0, 2, 3, 5, 7)[weapon_level], armor def + refine). The right-click menu adds `Sell` (same `sell` message) when the
player is next to a shop tile, and the Drop entry reads `Drop 1` on a stack (one unit leaves, like the engine).
