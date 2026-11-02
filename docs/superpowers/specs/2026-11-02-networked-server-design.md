# Milestone 2: networked server + client (design)

Date: 2026-11-02. Status: approved in conversation, written for review.

## Goal
Several people play in one JAX world over the network. A player logs in by name, their character persists across logouts and server restarts, bots fill every slot without a human, and a mouse click walks the player to a tile.

## Decisions (from the brainstorm)
1. Name-only login; characters persist (pos, spawn, hp, inv, kills, deaths). No passwords yet.
2. Each tick the server sends every client a viewport snapshot: only what is on that client's screen.
3. WebSocket + JSON. `play.py` becomes a thin network client; its drawing code stays until milestone 5 replaces it.
4. Bots (`bots.policies.bot_inputs`) drive every slot without a human. Click-to-walk is server-side A*.

Out of scope: accounts/passwords, chat, multiple worlds, binary protocol, persisting the world (drops, monster timers) between restarts.

## Architecture
```
play.py (tkinter, client thread) --WebSocket/JSON--> net/server.py (asyncio, one world, jit step)
                                 <-- snapshots ----              |
                                                                 v  every SAVE_EVERY_TICKS and on logout
                                                           characters/<name>.npz + roster.json
```
New package `net/`: `server.py`, `protocol.py`, `characters.py`, `pathing.py`. Existing `game/` and `bots/` are untouched except for any constants added to `game/config.py`.

## Protocol (`net/protocol.py`)
Every message is a JSON object with `"v": 1` and `"t": <type>`. Unknown `t` is ignored. Malformed JSON, or any message before `login`, gets `error` and the socket is closed.

Client -> server:
- `login {name}`: first message, once. Reply `welcome`, or `error` then close (bad name, already logged in, server full).
- `input {move: 0..4, attack: bool, eat: bool}`: sent on change. The server keeps the latest per slot and applies it every tick until replaced. Cancels an active `walk_to` path.
- `walk_to {x, y}`: left click. Replaces any current path. Out of map, beyond `PATH_RADIUS`, or unreachable: ignored with a `notice`.

Server -> client:
- `welcome {slot, map: MAP, tick_ms: TICK_MS, view_w, view_h}`: view size is fixed per connection, odd, chosen by the server (`VIEW_W = 61`, `VIEW_H = 33`).
- `snap {tick, you, ents, terrain, notice}` every tick:
  - `you`: `{pos, hp, max_hp, kills, deaths, inv (INV_SLOTS ids), town, safe}`.
  - `ents`: entities inside the view window centred on `you.pos`: living players `{kind: "p", name, pos, hp, max_hp}`, active monsters `{kind: "m", mkind, pos, hp, max_hp}`, active drops `{kind: "d", item, pos}`. Slot numbers never appear.
  - `terrain`: `{x0, y0, rows: [[terrain ids] * view_h] * view_w}` for the whole window when the window moved since the last snapshot sent to this client (step, respawn, login); otherwise `null`. Window clipped at map edges: out-of-map cells carry `T_ROCK`.
  - `notice`: short text or `null` ("too far", "you died", "disconnected" is client-side).
- `error {msg}` then close.

Validation on the server: `move` in 0..4, booleans are booleans, `walk_to` ints within `[0, MAP)`, name matches `[a-z0-9_]{3,16}` after lower-casing.

`protocol.py` exposes pure functions: `encode(msg) -> str`, `decode(text) -> dict | None`, `snapshot(host, slot, names, moved, notice) -> dict` where `host` is a plain-numpy copy of the arrays the snapshot needs (pulled once per tick for all clients), and `view_window(pos) -> (x0, y0)`.

## Server (`net/server.py`)
- CLI: `--host 0.0.0.0 --port 8765 --seed 0 --chars characters/ --tick-ms 120`.
- Startup: `make_world_batch(seed, 1)` unbatched (bots and monsters populated), jit `step` and `bot_inputs`, load `roster.json`.
- Slot table: `owner[slot] in {None, "bot", name}`; `latest_input[slot]`; `path[slot]: deque of move codes`; `last_window[slot]` for the terrain-changed check; `conn[slot]`.
- Tick loop (one coroutine, monotonic deadline, no drift):
  1. `inputs = bot_inputs(world, key)`; overwrite human slots: queued path move if any (a blocked move retries; after `PATH_STUCK_TICKS = 8` ticks without progress the path is dropped with a notice), else `latest_input`.
  2. `world = step(world, inputs)`.
  3. Pull host copies once (`np.asarray` of the player, monster and drop arrays); build each client's snapshot; send concurrently. A client whose previous send is still pending skips this tick (at most one in-flight snapshot per client; slow clients lag, the tick never waits).
  4. Every `SAVE_EVERY_TICKS = 2400` save all human characters. Log tick duration and overruns.
- Login: validate name -> refuse if already online -> load or create the character (new characters get the next home town round-robin and spawn on that town's temple tile chosen like the runner does: a free temple tile of the town) -> pick a slot: a bot slot with `slot % NT == town`, else any bot slot, else `error: server full` -> write the character into the slot (`pos, spawn, hp, inv, kills, deaths, active=True, dead=False, attack_cd=0`), `owner[slot] = name`.
- Logout (socket closed, or shutdown): save the character from the slot's fields; re-seed the slot as a bot on its town's temple (full HP, empty inventory, kills/deaths 0), `owner[slot] = "bot"`.
- Death does not log out; `notice: "you died"` on the tick `dead` becomes True.
- Errors: each connection handler is wrapped; a send failure closes that client and the loop continues; save failures are logged and retried at the next interval; Ctrl-C saves everyone then exits.

## Characters (`net/characters.py`)
- `valid_name(name) -> str | None` (lower-cased canonical form or None).
- `save(dir, name, fields: dict)` -> `dir/<name>.npz` (keys: pos, spawn, hp, inv, kills, deaths, town). `load(dir, name) -> dict | None`. Round-trip exact.
- `roster(dir)`: `roster.json` with `{name: town}`; `next_town(roster) -> int` round-robin by count.
- `new_character(town, temple_tile) -> fields` with `PLAYER_MAX_HP`, empty inv, zero counts.

## Pathing (`net/pathing.py`)
- `find_path(walls: np.ndarray, start, goal, max_radius=PATH_RADIUS) -> list[int] | None`: A* on the 4-neighbour grid where `walls == 0`, Manhattan heuristic, nodes restricted to the box `start +- max_radius`. Returns move codes (1 N, 2 S, 3 W, 4 E) that replay from `start` to `goal`, or `None`. Other entities are not obstacles; collisions are handled by the tick (retry, then drop after `PATH_STUCK_TICKS`).
- `PATH_RADIUS = 32`.

## Client (`play.py`)
- Keeps the fullscreen tkinter window, drawing and HUD; drops the local world and bots. `--host --port --name` CLI.
- A background thread runs the asyncio WebSocket client; snapshots go to the UI through a `queue.Queue`; the UI draws the newest and discards older ones each `tick_ms`.
- Keys: WASD/arrows send `input` on press (move code) and release (move 0); Space sets `attack` while held; E sends `eat` once. Left click on the map sends `walk_to` for that tile. Esc quits.
- HUD: hp, kills/deaths, town, SAFE, the current notice, "disconnected" when the socket drops (window stays open).
- Draws other players' names above them.

## Constants (`game/config.py`, new)
`TICK_MS = 120`, `VIEW_W = 61`, `VIEW_H = 33`, `PATH_RADIUS = 32`, `PATH_STUCK_TICKS = 8`, `SAVE_EVERY_TICKS = 2400`, `PROTOCOL_VERSION = 1`.

## Testing
- `tests/test_protocol.py`: snapshot window clipping at map edges (rock fill), terrain only when moved, entities filtered to the window, names not slots, `you` fields; encode/decode round-trip; decode rejects malformed text and wrong version.
- `tests/test_characters.py`: save/load exact; new-character defaults; `valid_name`; round-robin towns.
- `tests/test_pathing.py`: straight line; around a wall; unreachable -> None; outside radius -> None; replaying the codes reaches the goal.
- `tests/test_server.py`: in-process server on a free port, `websockets` client, fast tick: login -> welcome; two clients see each other's names in `ents`; `input` moves a player over ticks; `walk_to` reaches the tile; logout saves the file and the slot returns to "bot"; duplicate name refused; server full refused with `MAX_PLAYERS`-sized slot table monkeypatched small. Whole file under 30 s.
- `bench/server_load.py`: N headless clients that log in and send random inputs; prints tick overruns and bytes per client per second. Run by hand for the "hundreds of players" check.

## Dependency
`websockets` (pure Python) installed in `venv`.

## Scale notes
Snapshot cost is per connected client, not per slot: hundreds of clients at 8 ticks/s is hundreds of small JSON encodes per tick; the host-side numpy pull happens once per tick. If the tick overruns at scale, the first moves are (1) terrain as a packed bytes field, (2) snapshot building in a thread pool. Neither changes the protocol shape.
