# M0 contract brief: protocol v2, Godot debug layer, thin client

Lead-owned. Workers code against this at the same time and never change it; deviations go in the final report with a
reason. Read `CLAUDE.md` (root) and, for client work, `client/CLAUDE.md`, `client/docs/agent_brief.md` and
`client/docs/lessons_learned.md` first. Engine contracts that already exist: `docs/brief.md`.

## Goal
A Godot 4.7.2 client that logs in to the existing Python/JAX server, receives the whole map once, draws the
viewport from per-tick entity deltas, walks by click and keys, and carries the kit's debug layer (tuning panel,
debug frame + snapshot/restore, flight recorder + replay, overlay, lab, sims). On the server, protocol v2 with the
per-viewer visibility and delta computation **inside the JAX tick** (the user's rule: anything that can work in JAX
goes in JAX first, for speed).

## Units, ids, keys
- Tiles are int32 `[x, y]`, origin top-left, x right, y down, `MAP = 512` per side (`game/config.py`).
  The client draws tile `(x, y)` at pixel `(x * tile_px, y * tile_px)`, `tile_px` from `Tuning`.
- Move codes: 0 stay, 1 north (y-1), 2 south (y+1), 3 west (x-1), 4 east (x+1).
- **Entity id** (new, global): players `0 .. MAX_PLAYERS-1`, monsters `MAX_PLAYERS + i`, drops
  `MAX_PLAYERS + MAX_MONSTERS + i`. `E = MAX_PLAYERS + MAX_MONSTERS + MAX_DROPS` (512 + 4096 + 2048 = 6656).
- Server tick `TICK_MS = 120` for M0 (the 100 ms feel target is M1 work).
- Client input keys taken: WASD / arrows move, left click walk_to, Space attack, E eat, F1 overlay, F2 tuning panel,
  F3 recorder start/stop, F4 hide debug, F6-F9 render toggles (kit), F10 frame stamp (kit), Esc quit to login.

## Part A: engine, `game/view.py` (worker **view**, Sonnet). Owns `game/view.py`, `tests/test_view.py`, new
constants in `game/config.py` listed here (lead adds them: `VIEW_K = 256`, `MAX_VIEWERS = MAX_PLAYERS`, `N_ENTITIES`).

Pure JAX, jit-compatible, no Python loops over entities.

```python
def entity_table(world) -> dict
    # {"pos": int32 [E, 2], "vis": bool [E], "hp": int32 [E], "max_hp": int32 [E], "kind": int32 [E]}
    # vis: players active & ~dead; monsters active; drops active.
    # kind: players 0, monsters 1 + mkind, drops 100 + item id (one int the wire layer decodes).

def changed_mask(prev: dict, now: dict) -> bool [E]
    # pos, vis, hp or kind differs.

def view_update(now: dict, changed: bool [E], viewer_slot: int32 [MAX_VIEWERS], known: bool [MAX_VIEWERS, E])
    -> (send_ids: int32 [MAX_VIEWERS, VIEW_K], gone_ids: int32 [MAX_VIEWERS, VIEW_K], known_next: bool [MAX_VIEWERS, E])
```
- `viewer_slot[v]` is the player slot that viewer v watches, or `-1` for an unused viewer row (rows with -1 produce
  all `-1` ids and `known_next = False`).
- `visible[v, e]` = `now.vis[e]` and `|pos[e] - pos[viewer]| <= (VIEW_W // 2, VIEW_H // 2)` per axis (same window as
  `net.protocol.view_window`). A viewer always sees itself.
- `send[v, e] = visible & (changed | ~known)`, compacted to `VIEW_K` ids per row, **nearest first** (argsort on
  manhattan distance, ties by id); `-1` pads. Overflow entities stay unknown and are retried next tick.
- `gone[v, e] = known & ~visible`, compacted likewise (ties by id, `-1` pads).
- `known_next = (known & visible & ~gone_sent) | sent` where `sent`/`gone_sent` are the ids actually emitted.
- Tests: window edges, self always sent, overflow retry, gone after death and after leaving the window, viewer -1
  rows, jit once, and a shape check `MAX_VIEWERS x E` bools under `jax.jit` on CPU at batch 1.

## Part B: server, protocol v2 (worker **proto**, Sonnet). Owns `net/protocol.py`, `net/server.py`,
`net/client.py`, `play.py`, `bench/server_load.py`, `tests/test_protocol.py`, `tests/test_server.py`,
`tests/test_client.py`. Uses Part A's signatures exactly (code against them; the view worker lands in parallel).

`PROTOCOL_VERSION = 2`. JSON over WebSocket stays for M0 (binary is a later lever). Every message has `"v": 2, "t"`.

| t | dir | fields |
|---|---|---|
| `login` | C->S | `name` (unchanged) |
| `input` | C->S | `move`, `attack`, `eat` (unchanged semantics) |
| `walk_to` | C->S | `x`, `y` (unchanged) |
| `welcome` | S->C | `slot`, `map` (= MAP), `tick_ms`, `view_w`, `view_h`, `terrain`, `pz` |
| `snap` | S->C | `tick`, `you`, `ents`, `gone`, `notice`, `full` |
| `error` | S->C | `msg`, then close |

- `welcome.terrain`: base64 of zlib (level 6) of the int8 terrain, row-major `[MAP * MAP]` with index `y * MAP + x`;
  `pz` the same for the bool protection zone as uint8. Sent once per connection.
- `snap.ents`: the entities in `send_ids[v]` for this viewer, each `{id, kind, pos, hp, max_hp}` plus `name` for
  players (`kind` 0) and `item` for drops (`kind` >= 100; `item = kind - 100`); monsters carry `mkind = kind - 1`.
  `snap.gone`: list of ids. `snap.full` is true on the first snap after login (known was all False).
  `snap.you` keeps v1's `{pos, hp, max_hp, kills, deaths, inv, town, safe}`. `terrain` is gone from `snap`.
- The server keeps `known` (bool `[MAX_VIEWERS, E]`) and the previous `entity_table` on device; viewer row = slot.
  Login sets `known[slot] = False`; logout likewise. One `jax.jit` of `(step, entity_table, changed_mask,
  view_update)` per tick; **one host pull per tick** (`send_ids`, `gone_ids`, the entity table and `you` fields);
  JSON per client is built from those numpy rows.
- `net/client.py` and `play.py` move to v2: the client keeps an entity dict by id (upsert `ents`, delete `gone`)
  and a decoded terrain array. `bench/server_load.py` reports bytes/client/s without terrain; record the new
  numbers for 50 and 200 clients in your report (expected well under 10 KB/client/s).
- Tests: codec round trip for `welcome` (terrain decode equals the world's terrain), `snap` shape, `full` on first
  snap, gone after a monster dies, a second human sees the first appear and disappear, overrun count stays 0 in the
  existing load test, old v1 `terrain` key absent.

## Part C: client net logic, pure GDScript (worker **cnet**, Sonnet). Owns `client/game/net/*.gd` and
`client/tests/unit/test_net_*.gd`.

All `RefCounted` with `class_name`, typed, no Node, no wall clock, no `randf()`.

```gdscript
class_name NetLink            # interface: poll() -> Array[String] inbound texts since last poll; send(text: String) -> void
                              # connect_to(url: String) -> void; close() -> void; state() -> int (0 closed, 1 connecting, 2 open)
class_name WsLink extends NetLink      # WebSocketPeer wrapper (the only class here that touches the engine clock/net)
class_name FakeLink extends NetLink    # scripted inbound queue, optional lag in ticks via tick(); records outbound
class_name ReplayLink extends NetLink  # plays a FlightRecorder file back on tick()
class_name Codec              # static encode(msg: Dictionary) -> String; decode(text: String) -> Dictionary (empty on bad JSON / wrong v)
                              # static decode_map(b64: String, size: int) -> PackedByteArray (zlib inflate; empty on failure)
class_name EntityStore        # apply_snap(snap: Dictionary) -> void; get(id: int) -> Dictionary; ids() -> PackedInt32Array
                              # you: Dictionary; tick: int; snapshot() -> Dictionary; restore(d: Dictionary) -> void
class_name NetSession         # link: NetLink; store: EntityStore; map: MapData; login(name); send_input(move, attack, eat);
                              # walk_to(x, y); poll() -> Array[Dictionary] (decoded inbound, applied to store/map); signal-free: callers poll
class_name MapData            # size: int; terrain: PackedByteArray; pz: PackedByteArray; at(x, y) -> int; from_welcome(w: Dictionary) -> bool
class_name FlightRecorder     # start(path), stop(), record_in(frame, text), record_out(frame, text); JSONL lines {f, d: "in"/"out", m}
```
- Tests: codec round trips, bad input returns empty, map decode of a known zlib blob, store upsert/delete/full,
  snapshot/restore equality, FakeLink lag ordering, recorder write/read round trip, session login flow over FakeLink.

## Part D: client debug layer (worker **cdbg**, Sonnet). Owns `client/game/debug/tuning.gd` rows listed here,
`client/game/debug/tuning_panel.gd/.tscn`, `client/game/debug/debug_frame.gd`, `client/game/debug/overlay.gd/.tscn`,
`client/tests/lab/*`, `client/tests/unit/test_debug_*.gd`, `client/tests/sim/sim_lab_smoke.gd`.

- `Tuning` rows (add to `FIELDS` and vars): `tile_px` 32 (16..64 step 8), `interp_delay_ticks` 1.0 (0..3),
  `camera_zoom` 1.0 (0.5..3), `overlay_window_s` 2.0 (0.5..10), `name_label_px` 12 (8..24).
- `TuningPanel` (F2): one row per `FIELDS` entry (label, slider, value), live `set_value`, Reset, Save/Load preset
  to `user://tuning.json`.
- `DebugFrame` (`RefCounted`): `begin(tick)`, `note(key, value)`, `end()`; keeps a ring of the last 600 frames
  `{tick, frame, rx_bytes, ents, you_pos, latency_ms}`; `snapshot() -> Dictionary` and `restore(d)` that round-trip
  with `EntityStore.snapshot/restore` (Part C) so a state can be saved and put back.
- `Overlay` (F1, `CanvasLayer`): real frame time p50/p95/max over `overlay_window_s` measured with
  `Time.get_ticks_usec()` between frames in a lowest-priority node (never `Performance.TIME_PROCESS`), server tick,
  snaps/s, rx bytes/s, entity count, input-to-screen ms (time from a `walk_to` send to the first change of own pos),
  recorder state. Avoid pure `#00FF00`.
- `tests/lab/lab.tscn` + `lab.gd`: runs headless and windowed. Over `FakeLink` it feeds a synthetic 512x512 welcome
  and 300 snaps with 200 moving entities and measures: welcome-to-drawn ms (target < 1000), apply time per snap
  (target p95 < 2 ms), input-to-screen under 1 tick of fake lag (target p95 <= 150 ms at 120 ms ticks), and frame
  time p95 (target >= 55 fps, **WARN not FAIL when headless**). Prints `LAB <name> value=<n> target=<n> PASS|WARN|FAIL`
  per metric and `LAB summary pass=<n> warn=<n> fail=<n>`; exits non-zero on FAIL. `check_all.sh` picks it up.

## Part E: client view and input (worker **cview**, Sonnet). Owns `client/game/world/*` (map renderer, entity
sprites, camera), `client/game/ui/login.gd/.tscn`, `client/game/ui/hud.gd/.tscn`, `client/game/input/*`,
`client/tests/unit/test_view_*.gd`, `client/tests/sim/sim_login.gd`. Reports the lines for `main.gd`/`main.tscn`
and `project.godot` (input actions) to the lead; does not edit them.

- `MapRenderer` (`TileMapLayer`): a placeholder tile set built at runtime (one flat colour per terrain id 0..7,
  `tile_px` square, texture generated with `Image`), filled once from `MapData`; `Tuning.tile_px` changes rebuild it.
- `EntityLayer`: one `Node2D` per known id (pooled), a coloured quad per kind (players blue, own player white,
  monsters red by `mkind`, drops yellow), name label (`name_label_px`), HP bar for players and monsters;
  positions interpolated from the previous to the current tick pos over `interp_delay_ticks * tick_ms`.
- `Camera2D` on the own player, `camera_zoom`.
- Input: WASD/arrows -> `send_input(move, …)` on change, Space attack, E eat (tap sends once), left click ->
  `walk_to` of the clicked tile. Login scene: name field, host/port (defaults 127.0.0.1:8765), Connect; errors shown.
- HUD: HP bar, kills/deaths, inventory list (ids), notice line, `safe` badge.
- Sims: `sim_login.gd` over `FakeLink`: welcome -> map drawn (tile count equals MAP*MAP), 3 snaps -> entity nodes
  exist and move, `gone` removes a node, `--scripted` writes nothing.
- Render check: a windowed `-s` script that logs in over `FakeLink`, waits 10 frames, saves
  `get_viewport().get_texture().get_image()` to a PNG under the scratch folder, and looks at it (sample pixels: the
  own-player quad is white at screen centre). The user checks the real window.

## Part F: integration bot and dev tools (worker **cbot**, Sonnet, starts after B and C land). Owns
`client/tests/bot/bot.tscn`, `client/tests/bot/bot.gd`, `client/tools/dev_server.sh`, `client/tools/play_bot.sh`.
- `dev_server.sh` starts `python -m net.server --port <net_port>` from the repo root and prints the port.
- `bot.gd`: logs in over `WsLink`, walks to random reachable tiles for `--seconds=60` (default 30), attacks when a
  monster is adjacent, counts snaps, errors, stalls (no own-pos change for 50 ticks while a path is pending), and
  prints `BOT score=<snaps> errors=<n> stalls=<n>` plus `BOT <what> FAIL` lines; `--shard/--nshards` accepted.

## Lead-owned files
`game/config.py`, `game/state.py`, `game/step.py`, `client/project.godot`, `client/game/main.gd`,
`client/game/main.tscn`, `client/game/autoload/*`, `client/tests/runner.gd`, `client/tests/test_case.gd`,
`client/tests/sim/sim_suite.gd`, `docs/*`. Workers report lines to add.

## Done means
Your files' tests pass (`venv\Scripts\python.exe -m pytest tests/test_<module>.py -q` or, in `client/`,
`tools/run_tests.sh <name>` + `tools/lint.sh`, with the Windows env line from `client/CLAUDE.md`), one render looked
at if you render, and a report: files changed, summary lines, numbers before/after, integration lines, next 3 defects.
