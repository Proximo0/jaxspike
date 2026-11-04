# Round 4 cards: wire additions (ro4-cards)

Envelope unchanged (`v`, `t`, JSON text frames, unknown `t` ignored by older clients).

## `welcome.cards`
`ro_items.card_table_for_client()`: one dict per card item (27 today), ascending id.
```
{id, name ("Blobbin Card"), monster ("Blobbin", "" when no monster drops it), kind (monster kind, -1 = none),
 slot (SLOT_* it compounds into), slot_name ("Armor"), effects [[bonus id, value], ...] (negatives are drawbacks),
 text ("LUK +2, FLEE +1"), chance (per 10000, 0 = not dropped), mvp (bool), where ("Solmera Culvert" / field band
 text), icon ("card.<snake>" -> res://assets/cards/<snake>.png)}
```
Effects: the item row's `card_bonus`/`card_value` pair plus `items.CARD_EXTRA` (up to 3 more pairs), all folded into
`ro_items.equipped(world)["card_bonus"]` in the same scatter-add.

## `card` (S->C, world-wide)
```
{"t": "card", "item": int, "kind": int, "finder": str|null, "tick": int, "mvp": bool,
 "text": "<finder or Someone> found a <Monster> Card!"}
```
Sent to every connected client the tick a monster drop roll places a card pile. Source: `net.cards.card_events`
inside the server's jitted tick (fixed shape, at most `MAX_CARD_EVENTS = 8` per tick): drop slots that are active
card piles after `step` and were empty, held another item, or had their ttl refreshed, matched to a monster that
died this tick on the same tile (so a player's own `drop` of a card is not announced). Finder: the monster's top
damager before the killing tick (`mvp.top_damager`) when still active, else the nearest living player within
`FIND_RADIUS = 9` tiles, else none. Bots and none are `finder: null` ("Someone"). The host formats the message
(`protocol.card_message`) from the one per-tick pull.

## Client
`CardAlbum` (`client/game/ui/card_album.*`, Alt+A: plain A is WASD walk-left) handles `welcome` and `card`: a gold
chat line (pink for MVP cards) pushed into the ChatBox sibling, a 5 s banner with the portrait, and the album
(owned = carried stacks + socketed in carried gear + `you.storage`). `InventoryPanel` shows socket pips and card
effect / compound tooltips. `EntityLayer` draws `CardSparkle` over drop entities whose `item` is a card id.
