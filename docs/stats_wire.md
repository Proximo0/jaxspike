# Stats wire contract (M1)

What the net layer must expose for `game/ro_stats.py` (see `docs/m1_stats_brief.md`). Messages follow the existing
envelope: every message is a JSON object with `"v"` (protocol version) and `"t"` (type). The engine is pure JAX; this
file only lists what `net/` reads from `world["stats"]` / `ro_stats.derived(world)` and what it writes into
`inputs`. Bump `PROTOCOL_VERSION` when this lands.

## `snap.you` new fields
All ints unless noted; read from `world["stats"]` (and `derived(world)` for `max_hp`) in the tick's single host pull.

| field | source | notes |
|---|---|---|
| `level` | `stats["base_level"]` | 1..99 |
| `job_level` | `stats["job_level"]` | 1..`MAX_JOB_LEVEL[job]` |
| `job` | `stats["job"]` | 0..6, names in `welcome.job_names` |
| `stats` | `[str, agi, vit, int, dex, luk]` | list of 6, order = `welcome.stat_names` |
| `stat_points` | `stats["stat_points"]` | unspent |
| `skill_points` | `stats["skill_points"]` | unspent |
| `sp`, `max_sp` | `stats["sp"]`, `stats["max_sp"]` | |
| `base_exp` | `stats["base_exp"]` | exp into the current level |
| `base_exp_next` | `BASE_EXP_TABLE[base_level]` | 0 at the level cap |
| `job_exp` | `stats["job_exp"]` | |
| `job_exp_next` | `JOB_EXP_TABLE[job, job_level]` | 0 at the job level cap |
| `max_hp` | `derived(world)["max_hp"]` | replaces the constant `PLAYER_MAX_HP` in `snapshot` |

The `*_next` values are table lookups the server can do in numpy per viewer (or the tick can emit them); either is fine.
Clients render "exp / next" and treat `next == 0` as "max". Fields appear in every snapshot, not just full ones.

## New client messages
Both are one-shot: the server stores the request on the player's slot, passes it as the input for exactly the next
tick, then clears it (like `walk_to`'s one-shot semantics), so a held key does not allot twice. Without a pending
request the input is `-1`.

| message | validation (server, `validate_client`) | engine input |
|---|---|---|
| `{"v": N, "t": "allot", "stat": 0..5}` | `stat` is an int (not bool) in 0..5, else malformed | `inputs["allot"][slot] = stat` |
| `{"v": N, "t": "job", "job": 1..6}` | `job` is an int (not bool) in 1..6, else malformed | `inputs["job"][slot] = job` |

The engine validates the rest (points, cost, cap, Novice job level 10, alive); an invalid request is silently ignored,
and the client learns the result from the next snapshot. Each allot raises one stat by one point of its value; the
client sends one message per click. Inputs arrays are `int32 [MAX_PLAYERS]`, `-1` for no action; bots fill them from
`bots/ro_stats_policy.py`.

## `welcome` additions (static tooltip data, sent once)
| field | value |
|---|---|
| `stat_names` | `["str","agi","vit","int","dex","luk"]` |
| `job_names` | `["Novice","Swordsman","Mage","Archer","Acolyte","Thief","Merchant"]` |
| `stat_cost` | `STAT_COST` list, index = current stat value (1..98 meaningful; 99 is a sentinel = never) |
| `max_base_level`, `max_job_level` | `99`, and the per-job list `MAX_JOB_LEVEL` (length 7) |
| `base_exp_len`, `job_exp_len` | lengths of the exp tables (100 and 51) so the client can bound lookups |

The exp tables themselves are not sent; `base_exp_next` / `job_exp_next` in `snap.you` carry what a bar needs.

## Persistence note
`net/characters.py` files (`characters/<name>.npz`) gain 16 scalar stats fields, all ints (`was_dead` stored as 0/1):

- Saved: `str, agi, vit, int, dex, luk, base_level, job_level, base_exp, job_exp, job, stat_points, skill_points, sp, max_sp`
  and `was_dead` (16 scalars in total).
- Not saved: `kills_seen`, `deaths_seen`. On login the server sets them from the loaded `kills` / `deaths` so the
  first tick sees a zero delta (no phantom exp or death penalty). `fresh_rows(stats, mask)` zeroes them for a brand
  new character.
- `characters.load` must treat the new keys as optional: a file without them (an old character) gets a fresh Novice
  row, i.e. `fresh_rows`. Do not add them to the required `_KEYS`, or old characters would be rejected.
- Validation on load: clamp stats to 1..99, levels to their ranges, `job` to 0..6, points >= 0, `sp` to
  `[0, max_sp]`, `base_exp` below its table entry; anything else falls back to a fresh row.
- Logout and the periodic save write them from the same host pull as `pos`/`hp`; the guard in `save` is unchanged.
