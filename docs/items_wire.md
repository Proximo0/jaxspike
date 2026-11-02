# M3 items: wire additions (protocol v2)

Spec for the lead and the net worker. Nothing here is implemented in `net/` yet. The envelope is unchanged: every
message has `"v"` and `"t"`, JSON text frames, unknown `t` ignored, wrong types give `error` "malformed message".
Item ids are the `game/data/items.py` ids (`N_ITEMS = 73`; 0 = none). Slot numbers are inventory slot indices
`0..INV_SLOTS-1` (16).

## `welcome` additions

```
"items":  game.data.items.table_for_client()     list of 73 dicts, index == item id:
          {id, name, kind, cls, atk, def, matk, element, slots, weight, buy, sell, heal_hp, heal_sp,
           weapon_level, stackable, bonus, bonus_value, card_bonus, card_value}
"drops":  game.data.drops.table_for_client()     list per monster kind:
          {kind, drops: [{item, chance, count}], zeny_min, zeny_max}      chance is per mille
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
on the wire in M3. Ground drops keep `{kind:"d", item, pos}`; until `drops["count"]` exists each drop is one unit.

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
