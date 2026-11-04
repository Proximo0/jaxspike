# Persistence of the RO blocks: server lines (netwire integrates)

Code: `net/char_fields.py` (`character_fields`, `apply_character`), `net/characters.py` (`new_character`, `save`,
`load`). `net/server.py` is untouched by this worker. Line numbers are from `net/server.py` at 5a4b579.

## npz keys (all optional except the legacy `_KEYS`)
Legacy (required): `town pos spawn hp inv kills deaths`.
- stats (16 scalars): `stats_str stats_agi stats_vit stats_int stats_dex stats_luk stats_base_level stats_job_level
  stats_base_exp stats_job_exp stats_job stats_stat_points stats_skill_points stats_sp stats_max_sp stats_was_dead`
- skills: `skills_learned` (int32 [33]), `skills_sp`
- items: `items_inv_count` [16], `items_inv_refine` [16], `items_inv_cards` [16,4], `items_equip` [8], `items_ryn`
- craft: `craft_level` [6], `craft_exp` [6], `craft_gathers`, `craft_crafts`
- `storage` (int32 [100], waybell)
Not saved: `kills_seen`, `deaths_seen`, cooldowns, casts, buffs, party block (parties do not persist), `mvp`.

`characters.load` always returns every key above (fresh values for absent groups), so `apply_character` can be
called with its result directly. A group that is partial, mis-shaped or non-integer falls back to fresh as a whole.

## Save (logout and periodic)
`_fields(slot, pl)` is the only place that builds a save dict. Replace its body so it reads from one host pull of the
whole world, and keep the dead-player rule (`pos = spawn`, `hp = PLAYER_MAX_HP`):

```python
from net import char_fields
# periodic: in tick_once where `pull` is built, add the RO blocks to the same device_get:
#   pull["world_np"] = jax.device_get({b: self.world[b] for b in
#                                      ("players", "stats", "skills", "items", "craft")})
# _player_arrays() (logout path) does the same pull when no `pl` is given.
def _fields(self, slot, world_np=None):
    if world_np is None:
        world_np = jax.device_get({b: self.world[b] for b in ("players", "stats", "skills", "items", "craft")})
    f = char_fields.character_fields(world_np, slot, town=self.town[slot], storage=self.storage[slot])
    if bool(world_np["players"]["dead"][slot]):
        f["pos"] = f["spawn"].copy()
        f["hp"] = PLAYER_MAX_HP
    return f
```
`self.storage` is the per-slot waybell array the server keeps (`new_storage()` / `character_storage(char)`); it is
already in `char["storage"]` after `load`. `_collect_batch` passes `pull["world_np"]` instead of `pull["players"]`.
`inv` in the dict is the full `[INV_SLOTS]` row, so the old `[:INV_SLOTS]` slice is not needed.

## Login
In `login`, after `char` is loaded or created (`new_character` now carries fresh RO fields and `storage`), replace the
`_write_slot(...)` call with `_write_slot` for `active/dead/attack_cd` (keep it) followed by:

```python
self.world = char_fields.apply_character(self.world, slot, {**char, "pos": pos})   # pos may have been moved off a taken tile
self.storage[slot] = char["storage"].copy()
st = self.world["stats"]
self.world = {**self.world, "stats": {**st,
    "kills_seen": st["kills_seen"].at[slot].set(char["kills"]),     # first tick sees a zero delta
    "deaths_seen": st["deaths_seen"].at[slot].set(char["deaths"]),
}}
```
`apply_character` already writes pos/spawn/hp/inv/kills/deaths, so `_write_slot` may shrink to
`active/dead/attack_cd`. It does not clear the transient blocks (`skills.cooldown/cast_*/buffs`, `party.*`): on
login clear them for the slot with `ro_stats.fresh_rows`-style resets or leave to the bot reseed (below).

## Bot reseed after logout (`logout`, line 316)
The slot goes back to a bot, so give it fresh RO rows so a previous human's gear and levels do not leak:
`char_fields.apply_character(self.world, slot, {**char_fields.fresh_ro_fields(), "inv": zeros})` in place of the
`_write_slot` inventory zeroing (and `self.storage.pop(slot, None)`).
