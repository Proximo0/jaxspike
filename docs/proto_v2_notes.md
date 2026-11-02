# Protocol v2 bandwidth notes (netperf)

Implemented server-side: snap JSON encoding runs in the default thread pool, one executor job per tick for all
viewed clients. A client's `send_task` covers encode + send, so a client still in flight is skipped (full resync
next snap) and per-client order is strictly by tick. The tick never awaits a client. Wire format is unchanged.

Not implemented (would break the current Godot client; lead decision):

1. `max_hp` only on first appearance. `EntityStore._upsert` rebuilds each entity from the message with
   `int(src.get("max_hp", 0))` and does not carry the old value, so a missing key becomes 0. To adopt: server sends
   `max_hp` only when `full` or the id is new to that viewer (id not in `known` last tick); client must change
   `_upsert` to `int(src.get("max_hp", old.get("max_hp", 0)))` when the entity already exists.
2. `snap.you.inv` only when changed. `apply_snap` replaces `you` wholesale (`_entity_store_norm_you`) and
   `hud_text.gd` reads `you.get("inv", [])`, so an omitted `inv` empties the inventory display. To adopt: client
   merges `you` over the previous one (keep `inv` if absent); server omits `inv` when equal to last sent (and always
   on `full`).
