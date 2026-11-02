# M5 contract brief: parties, MVP bosses, kafra storage, chat

Sub-orchestrator-owned (2026-11-02). Workers code against this file plus `docs/ro_contract.md` and `docs/brief.md`.
Branch `ro/party`. Nothing here edits `game/config.py`, `game/state.py`, `game/step.py`, `game/combat.py`,
`net/protocol.py`, `net/server.py` or `net/characters.py`: every worker reports the exact lines the lead must add.

## The rule
Anything that can work in JAX goes into the JAX tick. Pure JAX in `game/` and `bots/`: no Python `if` on traced
values, no loops over entities, fixed shapes, masks over `active`, `.at[].min/max/add(mode="drop")` scatters,
ties by **lowest slot**, randomness only via `game.state.next_key`. `jax.jit` once in every test file and one
`jax.vmap` test over a batch of 2 worlds. Tests under 30 s on CPU each. Python: `C:\Users\crp27\jaxspike\venv\Scripts\python.exe`.

## Worlds in tests
`world = make_empty_world(jax.random.PRNGKey(0))`, then `world = {**world, "party": ro_party.make_state(), ...}`.
Positions are `world["players"]["pos"]` int32 `[MAX_PLAYERS, 2]`, `active`, `dead`, `spawn`, `hp`, `kills`.
Monsters: `pos`, `hp`, `kind`, `active`, `spawn`, `respawn_in`, `died`. `MAX_PLAYERS = 512`, `MAX_MONSTERS = 4096`,
`INV_SLOTS = 16`, `MAP = 512`, `len(TOWNS) = 5`. Base levels: `world["stats"]["base_level"]` int32 `[MAX_PLAYERS]`
if the key exists in the world, else treat everyone as level 1 (`_base_level(world)` helper in `ro_party.py`:
`world.get("stats", {}).get("base_level", jnp.ones(...))` is a Python dict lookup, fine under jit).

## Shared inputs (new keys; `empty_inputs()` is lead-owned, report the lines)
| key | dtype | shape | meaning |
|---|---|---|---|
| `party` | int32 | `[MAX_PLAYERS]` | 0 none, 1 create, 2 leave, 3 invite `target`, 4 accept pending invite |
| `target` | int32 | `[MAX_PLAYERS]` | player slot for party invite, -1 none (shared with other systems) |
| `save` | bool | `[MAX_PLAYERS]` | kafra save: set spawn to the temple of the town the player stands in |

## game/ro_party.py (worker: party)
```
MAX_PARTIES = 128
PARTY_MAX_MEMBERS = 12
SHARE_LEVEL_RANGE = 15        # base-level gap allowed for shared exp
SHARE_RADIUS = 64             # manhattan tiles from the killer to share exp
INVITE_TTL = 100              # ticks a pending invite lives
make_state() -> {"party_id": int32 [MAX_PLAYERS] (-1 none), "leader": bool [MAX_PLAYERS],
                 "member_count": int32 [MAX_PARTIES], "invite_from": int32 [MAX_PLAYERS] (-1 none),
                 "invite_ttl": int32 [MAX_PLAYERS]}
party_of(world) -> int32 [MAX_PLAYERS]            # party id or -1 (pinned by ro_contract.md)
share_exp(world, exp_gain: int32 [MAX_PLAYERS]) -> int32 [MAX_PLAYERS]
phase(world, inputs) -> world
```
Semantics:
- **create** (1): a living active player with `party_id == -1` becomes leader of a new party. Free party ids are
  those with `member_count == 0`; creators are ranked by `cumsum` and the k-th creator takes the k-th free id
  (same rank trick as `game/loot.py::_spawn_drops`). Creators beyond the free count are ignored.
- **invite** (3): leader with `inputs["target"]` naming an active player with no party and no pending invite sets
  the target's `invite_from = leader slot` and `invite_ttl = INVITE_TTL` when the party has room. Two leaders
  inviting the same target the same tick: lowest leader slot wins (scatter-min). Invites tick down and clear at 0.
- **accept** (4): a player with a pending invite whose inviter is still a leader of a party with room joins it.
  Several accepts into one party the same tick are admitted in slot order up to `PARTY_MAX_MEMBERS` (rank
  accepts per party with a segment cumsum: `jnp.cumsum` over a `[MAX_PLAYERS]` mask ordered by slot combined
  with a per-party running count via `segment_sum` is fine; simplest: accept all, then compute per-party counts,
  and reject accepts whose rank-within-party exceeds room, rank computed as cumsum of mask restricted to the
  party by a `[MAX_PARTIES, MAX_PLAYERS]` one-hot cumsum is too big: use `jnp.cumsum(mask)` after sorting by
  `party_id` with `jnp.argsort` (stable) and un-sort, O(N log N), no 2-D table).
- **leave** (2): the member leaves; a leaving leader passes leadership to the lowest remaining member slot; a party
  whose `member_count` reaches 0 is freed. Logged-out players (`active == False`) count as having left
  (clean them every tick: `party_id = -1` where `~active`).
- `member_count` is recomputed every tick as `segment_sum(ones where party_id >= 0, party_id, num_segments=MAX_PARTIES)`.
- `share_exp`: for each player with `exp_gain > 0` in a party, the gain is split equally (integer division, the
  killer keeps the remainder) among party members that are active, living, within `SHARE_LEVEL_RANGE` base
  levels of the killer **and** within `SHARE_RADIUS` manhattan tiles of the killer. Without a 2-D `[N, N]` table:
  for each party compute the sum of gains and the count of eligible members with `jax.ops.segment_sum`
  (`num_segments=MAX_PARTIES`), then each eligible member receives `party_sum // eligible_count`, and the killers
  receive the remainder `party_sum - share * eligible_count` added to their own share (one killer per party per tick
  is the common case; several killers in one party pool their gains, which is what RO does). Eligibility of
  member m relative to "the killer" uses the party's killer centroid: since there is no `[N, N]` table, use the
  killer with the lowest slot in the party as the reference (`segment_min` of killer slots) for both the level gap
  and the radius. Players not in a party keep their own gain. Output dtype int32, shape `[MAX_PLAYERS]`, sum of
  output <= sum of input (never creates exp).
- `phase` order inside: clean inactive -> ttl tick -> leave -> create -> invite -> accept -> recount.
Tests `tests/test_ro_party.py`: create/invite/accept/leave, tie-break, 13th member rejected, invite expiry, leader
hand-off, `share_exp` equal split with remainder, out-of-range level and out-of-radius members excluded, solo keeps
gain, conservation, jit + vmap.

## game/ro_mvp.py (worker: mvp)
```
MVP_KIND = (False, False, True)       # per monster kind; kind 2 = "Orc Lord" MVP (table may be longer than config's)
MVP_SPAWN_TICKS = (0, 0, 3000)        # respawn timer per kind (0 for non-MVP kinds)
MVP_HP = (0, 0, 2000)                 # hp per MVP kind (non-MVP kinds unused)
MVP_SPAWN_TILE = ((0, 0), (0, 0), (256, 200))  # fixed spawn tile per kind (world_layout coordinates rule: derive
                                      # from world_layout.town_centers() offsets, do not hardcode literals: e.g.
                                      # centre town + (0, -RAT_RING*MAP*1.5) rounded; compute at import with numpy)
MVP_DROP = ((0,)*4, (0,)*4, (ITEM_SWORD, ITEM_GOLD_COIN, ITEM_GOLD_COIN, ITEM_MEAT))  # item ids per MVP kind, 4 rolls
N_MVP_KINDS = len(MVP_KIND)
EVENT_RING = 8
make_state() -> {"top_damager": int32 [MAX_MONSTERS] (-1), "top_damage": int32 [MAX_MONSTERS],
                 "damage_ledger": int32 [MAX_MONSTERS], "next_spawn": int32 [N_MVP_KINDS],
                 "mvp_slot": int32 [N_MVP_KINDS] (monster slot each MVP kind occupies, -1 none),
                 "event": int32 [EVENT_RING, 3] (kind, killer slot, tick; -1 unused), "event_head": int32 []}
record_damage(world, attacker_idx: int32 [K], monster_idx: int32 [K], dmg: int32 [K]) -> world
is_mvp(world) -> bool [MAX_MONSTERS]
phase(world) -> world
```
Semantics:
- `record_damage` is pure and K-shaped (the lead calls it from combat and skills with `K = MAX_PLAYERS`; entries
  with `dmg <= 0` or `monster_idx < 0` are no-ops): `damage_ledger.at[monster].add(dmg)`; then per-attacker
  cumulative damage is not kept (no `[N, M]` table), so `top_damage` compares the *per-tick* damage of each
  attacker against the stored top: `top_damage = max(top, dmg)`, and `top_damager` is the attacker with the new
  max, **lowest slot on ties** (scatter-max of damage, then scatter-min of slot among attackers whose dmg equals
  the new max). Document this approximation in the module docstring (RO uses cumulative per-attacker damage;
  v1 uses best single hit with the ledger as total, so a later worker can switch to a `[MAX_MONSTERS, K]` ledger).
- `phase(world)`: for monsters with `died` set this tick **and** `MVP_KIND[kind]`: credit `top_damager` (if >= 0)
  with `players.kills += 1`? No: `combat` already credits kills; `phase` only writes an event row
  `(kind, top_damager, tick)` at `event_head % EVENT_RING` (several MVP deaths in one tick: write the lowest
  monster slot's only, bump head once), sets `next_spawn[kind] = MVP_SPAWN_TICKS[kind]` and `mvp_slot[kind]` stays,
  and puts the `MVP_DROP[kind]` items into the top damager's first free `inv` slots (up to 4; same first-free rank
  trick as `game/loot.py::_pickup`, extended to 4 items by a loop over the 4 drop columns, which is a loop over a
  constant, not entities). Reset `top_damager=-1, top_damage=0, damage_ledger=0` for monsters that died.
- Respawn: `next_spawn` counts down each tick for kinds with `mvp_slot >= 0` and `~monsters.active[mvp_slot]`; at 0
  the monster at `mvp_slot` is revived on `MVP_SPAWN_TILE[kind]` with `MVP_HP[kind]` (`active=True`,
  `respawn_in=0`, `pos=spawn=tile`). `monsters.respawn_in` for an MVP slot is set to a huge value at death by the
  lead's combat? No: `phase` runs after combat each tick and sets `respawn_in[mvp_slot] = 0` while
  `next_spawn > 0`, so the generic respawn never fires for MVP slots (document this; it is one line).
- `place_mvps(world) -> world` helper: assign monster slots `MAX_MONSTERS - 1 - k` to each MVP kind k, set kind,
  hp, pos, spawn, active; used by the test harness and the bot runner wrapper.
Tests `tests/test_ro_mvp.py`: record_damage ledger and tie by lowest slot, event written on MVP death with the
right killer and tick, drops land in the killer's inventory, non-MVP death writes no event, respawn after the timer
on the fixed tile, ring wraps after 8 events, jit + vmap.

## game/ro_kafra.py (worker: kafra, JAX half)
```
make_state() -> {}  # no state block; stateless phase
town_of_tile(pos: int32 [N, 2]) -> int32 [N]   # town index whose box (world_layout.town_boxes) holds pos, -1 outside
phase(world, inputs) -> world
```
`phase`: for living active players with `inputs["save"]` standing inside a town box (any tile of the town: the kafra
is "anywhere in town" in v1; the kafra tile rule can narrow this later), set `players.spawn` to that town's
**temple centre tile** (`world_layout.town_centers()[town]`, which is a `T_TEMPLE` tile since the temple square
is centred on the town centre: assert this in the test with `map_gen.make_terrain`). Pure JAX, fixed shape.
Tests `tests/test_ro_kafra.py`: save in town moves spawn, save outside town does nothing, dead player ignored,
spawn tile is `T_TEMPLE` on a generated terrain, jit + vmap.

## net/kafra.py (worker: kafra, Python half; numpy only)
```
STORAGE_SLOTS = 100
new_storage() -> np.ndarray int32 [STORAGE_SLOTS] of ITEM_NONE
transfer(inv: np.ndarray [INV_SLOTS], storage: np.ndarray [STORAGE_SLOTS], slot: int, direction: str) -> (inv, storage, ok: bool)
    direction "in": move inv[slot] into the first empty storage slot; "out": move storage[slot] into the first empty inv slot.
    Pure: returns new arrays, never mutates. ok False (arrays unchanged) when the slot is empty, out of range, or the
    destination is full.
character_storage(char: dict) -> np.ndarray   # char.get("storage") or new_storage(); validates shape/dtype, resets if bad
```
`net/characters.py` is lead-owned: report the lines (`storage` as an optional key in `_KEYS` handling in `load`:
missing storage => `new_storage()`; `new_character` adds `"storage": new_storage()`). Tests `tests/test_kafra.py`:
in/out round trip, full destination refuses, empty slot refuses, out-of-range refuses, purity, saving a character
dict with `storage` through `characters.save`/`load` round-trips the array (load keeps unknown keys: verify).

## net/chat.py (worker: chat; pure Python, no sockets)
```
CHAT_MAX_LEN = 200
CHAT_RATE_PER_TICK = 3                 # messages per slot per tick before drops
CHAT_VIEW_RADIUS = (VIEW_W // 2, VIEW_H // 2)   # public chat reaches players whose |dx|, |dy| fit the view window
class ChatRouter:
    def __init__(self): ...
    def route(self, tick: int, msgs: list[dict], pos: np.ndarray [MAX_PLAYERS, 2], active: np.ndarray bool,
              party_id: np.ndarray int32 [MAX_PLAYERS], names: dict[int, str]) -> dict[int, list[dict]]
```
`msgs` items: `{"slot": int, "kind": "say" | "party" | "whisper", "text": str, "to": str | None}`. Output: per
recipient slot, a list of `{"t": "chat", "kind", "from": name, "text", "tick"}` ready to encode. Rules: text is
stripped, empty dropped, cut to `CHAT_MAX_LEN`; more than `CHAT_RATE_PER_TICK` messages from one slot in one tick
are dropped (counter resets per tick, keyed by the `tick` argument); `say` reaches every active slot (the sender
included) within the view radius of the sender; `party` reaches every active member of the sender's party
(`party_id >= 0`, else dropped); `whisper` reaches the active slot whose name is `to` (lower-cased), else the
sender gets `{"t": "chat", "kind": "system", "text": "no such player"}`. Names map slot -> name; slots with no
name are bots and never receive anything. Use numpy for the radius mask, not a loop over players. Tests
`tests/test_chat.py`: say radius, party routing, whisper by name and the failure notice, rate limit, length cap,
empty dropped, bots never receive.

## bots/ro_party_policy.py (lead-written after the party worker lands)
Pure: `party_inputs(world, key) -> {"party": int32, "target": int32, "save": bool}` by home town: slot i's town is
`i % len(TOWNS)`; the lowest active slot of each town with no party creates (party 1), leaders invite the lowest
same-town partyless slot (3), invitees accept (4). Combined with `bots.policies.bot_inputs` in a harness.

## docs/party_wire.md (worker: docs)
Lists `snap.you` additions (`party`: id or -1, `leader`, `members`: names), client messages `party`
(`{"t":"party","op":"create|leave|invite|accept","target":name}`), `chat` (`{"t":"chat","kind","text","to"}`),
`storage` (`{"t":"storage","op":"in|out","slot"}`), server broadcasts `chat` and `mvp`
(`{"t":"mvp","kind","killer":name,"tick"}`) and the storage view in `snap.you.storage` (100 ids, sent on change).
Plus the exact lines for `empty_inputs`, `state.py`, `step.py` (phase order: ... survival -> craft -> party -> mvp ->
kafra -> tick), `combat.py` (`ro_mvp.record_damage(world, pidx, tgt_w, dmg)` after the player hit scatter),
`characters.py` (storage key) and `server.py` (`inputs["party"]`/`target`/`save` from messages, `ChatRouter` per
tick, `mvp.event` ring read each tick and broadcast new rows).

## Standard preamble for every worker
- Only edit the files listed. Never commit, push, or edit lead-owned files. Report lines for them.
- Keep engine runs short, at most 2 at once, none left running. CPU only.
- Final message: files changed, the pytest summary line for your tests, numbers, and the lines for lead-owned files.
