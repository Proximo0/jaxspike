# Protocol v2 bandwidth and tick time (netperf, netperf2)

Wire format stays protocol v2 (additive). Server flow per tick (`net/server.py`):

- **Pipelined tick loop** (`GameServer(pipeline=True)`, the default; `--no-pipeline` turns it off). Each period
  collects the tick dispatched one period earlier, dispatches the next tick (jit dispatch returns in about 1 ms on
  CPU; XLA computes on its own threads without the GIL), then builds and sends the collected tick's snapshots. The
  device tick (about 45 ms on a CPU at 512 viewers) overlaps the snapshot work. Cost: snaps leave about one device-tick
  later. Chat lines are routed with the tick dispatched after they arrive, so routing sees the same positions as
  without the pipeline. `tick_once()` stays synchronous (dispatch, collect, publish) for the tests.
- **One snapshot pass** (`net.protocol.build_snaps`): each entity row is encoded once per tick and shared by every
  viewer that is sent it; per-viewer row counts and gathers run once over all viewers in numpy. An earlier attempt
  that encoded in the default executor lost to the GIL and is gone.
- **No permessage-deflate** (`websockets.serve(..., compression=None)`). With 480 Python clients zlib per message per
  client took 27% of the event loop (profile). The Godot `WebSocketPeer` does not negotiate the extension, so every
  client now gets what the Godot client already got; the byte numbers below are wire payload bytes.
- `gc.freeze()` after `prepare()`: the world, the welcome tables and the jit caches stay out of every gc pass.
- A client whose previous send is still in flight when its tick is dispatched is skipped (viewer -1, full resync on
  its next snap). With the pipeline a send can still be in flight at publish time: the new send waits for it, so
  per-client order stays strictly by tick.

## `you` on change
`snap.you` carries `pos` and `hp` always, and every other key only when its value differs from what this client last
got (`net.protocol.you_delta`, per-key compare against `Client.last_you`); `full` snaps carry everything. A slot whose
source rows did not change since the previous tick (one vectorised compare per pulled array, `_dirty_slots`) skips
building `you` at all. `members` / `storage` keep their own change detection. Clients MUST merge `you` over their
previous copy: the Godot `EntityStore.apply_snap` and `net.client` consumers do (test
`test_store_merges_you_over_change_only_snaps`).

## Binary snaps (`"format": "bin"` in `login`)
msgpack is not in the venv, so the binary format is a fixed little-endian layout (`net.protocol.BIN_HEADER`,
`BIN_ROW`, decoders `net.protocol.decode_snap_bin` and Godot `Codec.decode_snap_bin`). The default stays JSON.
Only `snap` messages are binary frames; `welcome`, `chat`, `mvp` and `error` stay JSON text frames.

| offset | type | field |
|---|---|---|
| 0 | 2 bytes | magic `JS` |
| 2 | u8 | layout version (1) |
| 3 | u8 | flags: bit0 `full`, bit1 animation columns valid |
| 4 | u32 | tick |
| 8 | u16 | n_ents |
| 10 | u16 | n_gone |
| 12 | u32 | json_len |
| 16 | json_len bytes | UTF-8 JSON `{"you": {...}, "notice": str or null, "names": {"<id>": name}}` (human players among the ents; other players are "bot") |
| then | n_ents x 30 bytes | rows: id u16, kind u16, x u16, y u16, hp i32, max_hp i32, t (start_tick) i32, h (hit) i32, s (speed_ms) i16, m (dur_ms) i16, a (action) i8, d (face) i8 |
| then | n_gone x u16 | gone ids |

Decoded entities have the JSON shape: `name` for kind 0, `mkind = kind - 1` for 1..99, `item = kind - 100` for
100..399, nodes (kind >= 400) `{id, kind, pos, hp}` only; `a d t m s` for kind < 100 when flag bit1 is set, `h` only
when != 0. Ids fit u16 while the entity table has fewer than 65536 rows (asserted).

## Resource nodes in the entity table
`game/view.py` appends the craft nodes after the drops: ids `MAX_PLAYERS + MAX_MONSTERS + MAX_DROPS + i`, table
length `N_VIEW_ENTITIES = N_ENTITIES + MAX_NODES` (8704; `config.N_ENTITIES` does not include nodes yet). `vis` =
`placed` (a depleted node stays visible), `hp` = `active` 0/1, `max_hp` 1, `kind = NODE_KIND_BASE + node kind` with
`NODE_KIND_BASE = 100 + ITEM_ID_LIMIT = 400`: drops already use `100 + item id` up to 399, so 200..207 (craft_wire.md)
and 300..307 would both collide with drops of craft items. `view_update` takes E from the table's shape.
JSON node rows are `{id, kind, pos, hp}`.

## Measuring
`bench/server_load.py --spawn --clients N --tick-ms 100 --format json|bin` starts the server in its own process and
runs light asyncio clients in worker processes, then reads overruns from the server's tick log for the measured
window only. `--embedded` (server thread plus one client thread each, one process) measures the bench as much as the
server: its 480-client runs starve the server of the GIL. `python -m net.server --bench-ticks N [--platform gpu]`
times the device tick plus host pull at 512 viewers without sockets.

### Numbers (2026-11-02, Windows CPU box shared with 11 other sessions at ~90% load, so single runs are noisy)
Before = the code before netperf2, `--embedded` (the only mode then), JSON. After = `--spawn`, all changes above.
TICK 100 ms, 20 s windows; bytes = snap payload bytes per client per second; overruns = ticks over budget or late.

| clients | run | snaps/client/s | bytes/client/s | overruns / ticks |
|---|---|---|---|---|
| 50 | before, JSON | 7.82 | 14015 | 19 / 146 |
| 200 | before, JSON | 4.27 (71 of 200 refused at the handshake) | 11633 | 56 / 142 |
| 480 | before, JSON | not measured (the import broke mid-run) | | |
| 50 | after, JSON | 10.00 | 7436 | 0 / 200 |
| 50 | after, bin | 10.00 | 3137 | 7 / 200 (noise: max tick 116 ms at trivial load) |
| 200 | after, JSON | 10.00 | 15924 | 0 / 200 |
| 200 | after, bin | 9.80 | 6979 | 3 / 190 |
| 480 | after, JSON | 9.05 | 28103 | 14 / 180 |
| 480 | after, bin | 9.98 | 12570 | 0 / 200 |
| 480 | after, JSON, `--no-pipeline` | 8.14 | 27476 | 28 / 160 |
| 480 | after, bin, `--no-pipeline` | 8.00 | 12340 | 26 / 160 |

Bytes per client grow with the client count because every bench client starts in one of 5 temple squares: at 480
each one sees about 100 moving players. Earlier "after" runs with permessage-deflate on: 480 JSON 6.1 snaps/s and 35
overruns (profile: zlib 27% of the loop); turning compression off took 480 bin from 5.7 to 10.0 snaps/s.

Device tick plus host pull at 512 viewers (`--bench-ticks 60`, E = 8704): CPU p50 43.8 ms, p95 53.5 ms; GPU (WSL,
CUDA) p50 26.8 ms, p95 34.3 ms. `bench/bench_view.py` (step + view, E with and without the 2048 node rows) shows no
measurable cost from the nodes at this noise level (p50 51 ms with nodes, 66-80 ms without, run in that order).

### Round 3 (after the round-2 integration, 2026-11-02 evening, box at 97% CPU with 13 teams; paused by the lead)
Device tick plus pull at 512 viewers on CPU is now p50 229.5 ms, p95 357.9 ms, min 183.9 ms (was 43.8 / 53.5 / 37.4
before the integration; the lead's GPU soak shows the tick about 3x dearer, the rest is load). At a 100 ms tick on
CPU the device tick alone is over budget, so 500 clients cannot hold 10 Hz whatever the wire does: 500 JSON gave
6.41 snaps/s with 104 overruns in 200 ticks, 500 bin 1.50 snaps/s (70 ticks in a 30 s window). These runs measure the
loaded box, not the server; rerun on a quiet box (command in the ro3-netperf hand-back). `--bench-ticks` now runs
without the websockets package (the WSL GPU venv has none); `start()` says to install it.

Still not done: `max_hp` only on first appearance (`EntityStore._upsert` needs `old.get("max_hp")` first).

# RO systems over protocol v2 (netwire)

`PROTOCOL_VERSION` stays 2 (additive). Messages, welcome tables and `snap.you` fields follow `docs/*_wire.md`.

Server behaviour worth knowing:
- Human RO inputs are one-shot: `GameServer.pend` (numpy arrays per input key) is copied into `_tick_fn` as `hro`,
  applied on human slots only (`where(hmask, hro, bot_policy)`), and reset to "none" right after the tick starts.
  `skill` and `party invite` both use `inputs["target"]`; if one client sends both in the same tick the last wins.
- Extra messages (chat lines, `mvp`) go out ahead of the next snapshot in the same send task (`Client.outbox`, cap 128);
  a client whose snapshot send is still in flight keeps them queued. Chat is therefore at most one tick late.
- `members` / `storage` in `you` are sent on full snaps and when they change, like every other `you` key (see
  "`you` on change" above); clients merge `you` over the previous one.
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

Nodes are in the entity table now (kinds 400..407, see above). `entity_table` uses the derived max_hp for players,
but `changed_mask` still ignores `max_hp`, so other viewers see a level-up max_hp only when hp or pos changes.
Also: `kafra.character_storage` validates ids against `NUM_ITEMS` (4); the server uses `char["storage"]` from
`characters.load` instead.
