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

# RO systems over protocol v2 (netwire)

`PROTOCOL_VERSION` stays 2 (additive). Messages, welcome tables and `snap.you` fields follow `docs/*_wire.md`.

Server behaviour worth knowing:
- Human RO inputs are one-shot: `GameServer.pend` (numpy arrays per input key) is copied into `_tick_fn` as `hro`,
  applied on human slots only (`where(hmask, hro, bot_policy)`), and reset to "none" right after the tick starts.
  `skill` and `party invite` both use `inputs["target"]`; if one client sends both in the same tick the last wins.
- Extra messages (chat lines, `mvp`) go out ahead of the next snapshot in the same send task (`Client.outbox`, cap 128);
  a client whose snapshot send is still in flight keeps them queued. Chat is therefore at most one tick late.
- `members` / `storage` in `you` are sent on full snaps and when they change; the client must MERGE `you` over the
  previous one (the current Godot `apply_snap` replaces `you` wholesale, so those two keys vanish until fixed).
  `craft`, `inv`, buffs etc. are sent every snap (no change suppression).
- Login: `password` / `create` optional. No password: allowed only when the name has no account (compat for name-only
  characters); a name with an account always needs its password. `GameServer(require_password=True)` /
  `--require-password` turns the compat path off. Known gap: `create` for a name that has an old name-only character
  file lets the creator claim it.
- Storage ops run before the tick; the town test is the town box, not the kafra tile. Storage holds item ids only, so
  gear with refine/cards is refused and a stack moves one unit.
- Slot reuse: login and logout reset the slot's stats/skills/items/craft/party rows to fresh (one jitted call).
  Parties and cooldowns/buffs do not persist; bots taking a slot over start as fresh Novices.
- `sp`: the engine keeps two SP counters (`stats.sp` regens, `skills.sp` is what casting spends and also regens). The
  wire sends `skills.sp` (the one that gates casts) with `max_sp` from `derived`. The two should be merged in the engine.

Not done (lead-owned `game/view.py`): node entities (kinds 200..207, ids after the drops, `docs/craft_wire.md`) are not
in the entity table, so `snap.ents` has no nodes; players' `max_hp` in `ents` is patched in `snapshot` from the derived
stats, but `changed_mask` ignores `max_hp`, so other viewers see a level-up max_hp only when hp or pos changes.
`entity_table` should use derived max_hp and `changed_mask` should compare it.
Also: `kafra.character_storage` validates ids against `NUM_ITEMS` (4); the server uses `char["storage"]` from
`characters.load` instead.
