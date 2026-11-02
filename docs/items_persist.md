# M3 items: what a saved character must hold

For `net/characters.py` (lead or net worker). A character file is the player's row of `world["players"]` plus the
row of the items block `world["items"]`. Add these five fields next to the existing `inv` (item ids, int32
`[INV_SLOTS]`, which stays where it is). Names are the file keys.

| key | dtype, shape | valid values |
|---|---|---|
| `inv` (existing) | int32 `[INV_SLOTS]` | each in `0 .. N_ITEMS-1` (was `< NUM_ITEMS = 4`: raise it to `N_ITEMS = 73`) |
| `inv_count` | int32 `[INV_SLOTS]` | `1 .. MAX_STACK (9999)` where `inv != 0`, exactly 0 where `inv == 0` |
| `inv_refine` | int32 `[INV_SLOTS]` | `0 .. MAX_REFINE (10)`; 0 where `inv == 0` |
| `inv_cards` | int32 `[INV_SLOTS, 4]` | each 0 or an item id whose kind is `KIND_CARD`; a non-zero card only at a socket index `< SLOTS[inv[slot]]` and only where `inv != 0` |
| `equip` | int32 `[8]` | each `-1 .. INV_SLOTS-1`; a non-negative entry points at a slot with `inv != 0` whose item has the right type for that equip slot (weapon kind at 0; armor kind with `CLASS == slot` at 1..5; armor with `CLASS == SLOT_ACCESSORY` at 6 and 7); no inventory slot appears twice |
| `zeny` | int (scalar; store as int64 or Python int) | `0 .. MAX_ZENY (2_000_000_000)` |

## Rules

- `new_character` writes defaults: `inv_count` zeros, `inv_refine` zeros, `inv_cards` zeros `[INV_SLOTS, 4]`,
  `equip` all -1 (8 entries), `zeny` 0.
- `load` returns `None` (as for any corrupt file) when a present field has the wrong shape or an out-of-range value
  as listed above. Check shapes first (`np.asarray(x).shape`), then ranges, so a bad file never reaches the world.
- Migration: a file from before M3 has none of the five new keys. Accept it and fill the defaults, except that
  `inv_count` is `1` where `inv != 0` (`ro_items._normalize` would do the same on the first tick). A file with some
  but not all of the new keys is corrupt (`None`). Old `inv` ids are 0..3 and stay valid. `_KEYS` (the required
  set) must not grow, so old saves keep loading.
- Login copies the validated row into the slot: `players.inv[slot]` from `inv`, and `items.inv_count[slot]`,
  `inv_refine[slot]`, `inv_cards[slot]`, `equip[slot]`, `zeny[slot]` from the new keys (one batched scatter, like the
  existing fields). Logout and the periodic save pull the same rows from the host copy of the world.
- Re-seeding a slot as a bot after logout, and a respawn, reset nothing in the items block; a fresh bot slot gets
  the defaults above so a previous human's gear does not leak to the bot.
- Ground drops are not persisted (as today). Zeny and gear survive death in M3 (no death penalty yet).
- The engine keeps the invariants in `bench/items_soak.py` (`check_item_invariants`); the same checks are the
  right shape for a `characters.validate_items(fields)` helper, and the persistence tests should round-trip a
  character holding a refined, carded, worn weapon plus coins-turned-zeny and a stack of potions.
