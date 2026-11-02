# Networked Server + Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Several people play one JAX world over WebSockets; characters persist by name; bots fill the other slots; a click walks the player to a tile.

**Architecture:** A new `net/` package: pure protocol + snapshot functions, character files, bounded A*, an asyncio server that owns one world and runs the jit `step` on a fixed timer, and a threaded client used by `play.py` (which keeps its drawing and drops its local world). The game engine (`game/`, `bots/`) is not changed except for new constants in `game/config.py`.

**Tech Stack:** Python 3.15 venv (`venv\Scripts\python.exe`), JAX CPU, numpy, `websockets` 17.4 (installed), asyncio, tkinter, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-networked-server-design.md`

## Global Constraints
- Pure JAX rules in `game/` are untouched; `net/` is plain Python/numpy and may hold coordinate literals only in tests.
- Every protocol message is a JSON object with `"v": 1` and `"t": <type>`; unknown `t` ignored; malformed or pre-login message -> `error` then close.
- Names: lower-cased, must match `[a-z0-9_]{3,16}`; a name already online is refused.
- Constants (add to `game/config.py`): `TICK_MS = 120`, `VIEW_W = 61`, `VIEW_H = 33`, `PATH_RADIUS = 32`, `PATH_STUCK_TICKS = 8`, `SAVE_EVERY_TICKS = 2400`, `PROTOCOL_VERSION = 1`. Code reads them from config, never repeats the numbers.
- Move codes: 0 stay, 1 N (y-1), 2 S (y+1), 3 W (x-1), 4 E (x+1) (`DIR_DELTAS`).
- Out-of-map cells in a terrain window are `T_ROCK`.
- The tick never awaits a client: at most one in-flight snapshot send per client; a client still sending skips this tick.
- Tests: plain pytest, each file under 30 s on CPU. Run from the project folder with `venv\Scripts\python.exe -m pytest tests/<file> -q`.
- Commit after each task on `main` (local repo, no remote). Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Do not run GPU work; everything here is CPU.

## Review Focus
1. A client that disconnects mid-tick (socket closed while its snapshot is being sent) must not stop the tick loop or leave its slot marked human. Test in Task 4 (`test_disconnect_during_play_frees_slot`).
2. A `walk_to` whose goal is the player's own tile, or a tile adjacent to the player, must not crash or loop; the path is empty or one move. Test in Task 2 (`test_path_to_self_and_adjacent`).
3. Two humans whose home towns share a slot stripe log in at the same time: slot assignment must never hand the same slot to both. Login runs on the event loop thread (no await between pick and mark), pinned by a test in Task 4 (`test_two_logins_get_distinct_slots`).
4. A character file that is missing or corrupt on login must produce a fresh character (missing) or an `error` (corrupt), never a server crash. Test in Task 1 (`test_load_corrupt_file_returns_none`) and Task 4 (`test_login_corrupt_character_errors`).
5. The terrain window at the top-left corner of the map (negative `x0`) must produce `VIEW_W x VIEW_H` rows with rock padding, not an index error. Test in Task 3 (`test_terrain_window_clips_at_origin`).

---

### Task 1: Config constants and character files

**Files:**
- Modify: `game/config.py` (append constants)
- Create: `net/__init__.py` (empty), `net/characters.py`
- Test: `tests/test_characters.py`

**Interfaces:**
- Produces: `valid_name(name: str) -> str | None`; `new_character(town: int, tile: tuple[int, int]) -> dict`; `save(chars_dir: Path, name: str, fields: dict) -> None`; `load(chars_dir: Path, name: str) -> dict | None`; `load_roster(chars_dir) -> dict[str, int]`; `save_roster(chars_dir, roster) -> None`; `next_town(roster: dict[str, int]) -> int`.
- A character dict has exactly: `town` (int), `pos` and `spawn` (np.int32 `[2]`), `hp` (int), `inv` (np.int32 `[INV_SLOTS]`), `kills` (int), `deaths` (int).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from game.config import INV_SLOTS, PLAYER_MAX_HP, ITEM_MEAT
from game.world_layout import TOWNS
from net.characters import (load, load_roster, new_character, next_town, save, save_roster, valid_name)

def test_valid_name():
    assert valid_name("Curtis_1") == "curtis_1"
    for bad in ("ab", "a" * 17, "bad name", "bad-name", "", "ünïcode"):
        assert valid_name(bad) is None

def test_new_character_defaults():
    c = new_character(2, (10, 20))
    assert c["town"] == 2 and c["hp"] == PLAYER_MAX_HP and c["kills"] == 0 and c["deaths"] == 0
    assert c["pos"].tolist() == [10, 20] and c["spawn"].tolist() == [10, 20]
    assert c["inv"].shape == (INV_SLOTS,) and c["inv"].dtype == np.int32 and not c["inv"].any()

def test_save_load_exact(tmp_path):
    c = new_character(1, (5, 6)); c["hp"] = 77; c["inv"][3] = ITEM_MEAT; c["kills"] = 9
    save(tmp_path, "bob", c)
    r = load(tmp_path, "bob")
    assert set(r) == set(c)
    for k in c:
        np.testing.assert_array_equal(np.asarray(r[k]), np.asarray(c[k]))
    assert isinstance(r["town"], int) and isinstance(r["hp"], int)
    assert load(tmp_path, "nobody") is None

def test_load_corrupt_file_returns_none(tmp_path):
    (tmp_path / "eve.npz").write_bytes(b"not an npz")
    assert load(tmp_path, "eve") is None

def test_roster_round_robin(tmp_path):
    assert load_roster(tmp_path) == {}
    r = {}
    towns = [next_town(r) for _ in range(len(TOWNS) + 1)]
    for i, t in enumerate(towns):
        r[f"p{i}"] = t
    assert towns == list(range(len(TOWNS))) + [0]
    save_roster(tmp_path, r)
    assert load_roster(tmp_path) == r
```

- [ ] **Step 2: Run to verify failure**

Run: `venv\Scripts\python.exe -m pytest tests/test_characters.py -q`
Expected: FAIL with `ModuleNotFoundError: net`

- [ ] **Step 3: Append the constants to `game/config.py`** under a `# Networking (milestone 2)` comment, values from Global Constraints.

- [ ] **Step 4: Implement `net/characters.py`**

`save` writes `np.savez(dir / f"{name}.npz", **fields)`; `load` returns `None` when the file is missing or `np.load` raises (`OSError`, `ValueError`, `zipfile.BadZipFile`), and converts 0-d arrays back to Python ints. Roster is `roster.json`; `next_town` returns the town with the fewest names, lowest index on ties.

- [ ] **Step 5: Run to verify pass**

Run: `venv\Scripts\python.exe -m pytest tests/test_characters.py -q`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add game/config.py net/__init__.py net/characters.py tests/test_characters.py
git commit -m "feat(net): character files and networking constants"
```

---

### Task 2: Bounded A* for click-to-walk

**Files:**
- Create: `net/pathing.py`
- Test: `tests/test_pathing.py`

**Interfaces:**
- Produces: `find_path(walls: np.ndarray, start: tuple[int, int], goal: tuple[int, int], max_radius: int = PATH_RADIUS) -> list[int] | None` returning move codes 1..4 that replay from `start` to `goal`; `[]` when `start == goal`; `None` when unreachable, when `goal` is a wall or out of the map, or when the goal lies outside the box `start ± max_radius`.
- Consumes: `DIR_DELTAS`, `PATH_RADIUS` from `game.config`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from game.config import DIR_DELTAS, MAP, PATH_RADIUS
from net.pathing import find_path

def replay(start, moves):
    x, y = start
    for m in moves:
        dx, dy = DIR_DELTAS[m]; x, y = x + dx, y + dy
    return x, y

def test_straight_line():
    walls = np.zeros((MAP, MAP), np.int8)
    p = find_path(walls, (10, 10), (10, 15))
    assert p == [2] * 5

def test_around_wall():
    walls = np.zeros((MAP, MAP), np.int8); walls[12, 5:16] = 1
    p = find_path(walls, (10, 10), (14, 10))
    assert p is not None and replay((10, 10), p) == (14, 10)
    assert all(walls[xy] == 0 for xy in [replay((10, 10), p[:i]) for i in range(len(p) + 1)])
    assert len(p) == 4 + 2 * 6  # detour around the wall end at y=16

def test_unreachable_and_invalid_goal():
    walls = np.zeros((MAP, MAP), np.int8); walls[8:13, 8:13] = 1; walls[10, 10] = 0
    assert find_path(walls, (10, 10), (20, 10)) is None      # boxed in
    assert find_path(walls, (20, 10), (9, 9)) is None        # goal is a wall
    assert find_path(walls, (20, 10), (MAP, 10)) is None     # out of map

def test_radius_cutoff():
    walls = np.zeros((MAP, MAP), np.int8)
    assert find_path(walls, (100, 100), (100, 100 + PATH_RADIUS)) is not None
    assert find_path(walls, (100, 100), (100, 101 + PATH_RADIUS)) is None

def test_path_to_self_and_adjacent():
    walls = np.zeros((MAP, MAP), np.int8)
    assert find_path(walls, (10, 10), (10, 10)) == []
    assert find_path(walls, (10, 10), (11, 10)) == [4]
```

- [ ] **Step 2: Run to verify failure** — `venv\Scripts\python.exe -m pytest tests/test_pathing.py -q` → `ModuleNotFoundError: net.pathing`

- [ ] **Step 3: Implement `find_path`** — A* with `heapq`, Manhattan heuristic, 4 neighbours in `DIR_DELTAS[1:]` order, nodes restricted to the clipped box; tie-break on insertion order so paths are deterministic.

- [ ] **Step 4: Run to verify pass** — `5 passed`

- [ ] **Step 5: Commit** — `git add net/pathing.py tests/test_pathing.py && git commit -m "feat(net): bounded A* for click-to-walk"`

---

### Task 3: Protocol: messages, validation, host view, snapshots

**Files:**
- Create: `net/protocol.py`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Produces:
  - `encode(msg: dict) -> str` (adds `"v": PROTOCOL_VERSION`); `decode(text: str) -> dict | None` (None unless valid JSON object with `v == PROTOCOL_VERSION` and a string `t`).
  - `validate_client(msg: dict) -> dict | None`: for `t == "login"` requires `name` str; `"input"` requires `move` int in 0..4 and `attack`, `eat` bools; `"walk_to"` requires `x`, `y` ints in `[0, MAP)`; anything else returns `None`. Returns the message with only the allowed keys.
  - `view_window(pos) -> tuple[int, int]`: `(x - VIEW_W // 2, y - VIEW_H // 2)`.
  - `terrain_window(terrain: np.ndarray, x0: int, y0: int) -> list[list[int]]`: `VIEW_W` rows (one per x) of `VIEW_H` ints, `T_ROCK` outside the map.
  - `host_view(world) -> dict`: numpy copies of `terrain`, `pz`, `players{pos,hp,active,dead,inv,kills,deaths}`, `monsters{pos,hp,kind,active}`, `drops{pos,item,active}` (`np.asarray` on each leaf, once per tick).
  - `snapshot(hv: dict, slot: int, names: dict[int, str], send_terrain: bool, notice: str | None, tick: int) -> dict` with the spec's `snap` shape: `you = {pos, hp, max_hp, kills, deaths, inv, town, safe}` (`town = slot % len(TOWNS)`, `safe = pz[pos]`), `ents` filtered to the window, `terrain` only when `send_terrain`.
- Consumes: constants from `game.config`; `MONSTER_HP`, `PLAYER_MAX_HP`; `TOWNS`.

- [ ] **Step 1: Write the failing tests**

```python
import jax, numpy as np
from game.config import (INV_SLOTS, MAP, MONSTER_HP, PLAYER_MAX_HP, PROTOCOL_VERSION, T_GRASS, T_ROCK,
                         VIEW_H, VIEW_W)
from game.state import make_empty_world
from net.protocol import decode, encode, host_view, snapshot, terrain_window, validate_client, view_window

def test_encode_decode_round_trip_and_rejects():
    m = decode(encode({"t": "input", "move": 2, "attack": False, "eat": True}))
    assert m == {"v": PROTOCOL_VERSION, "t": "input", "move": 2, "attack": False, "eat": True}
    for bad in ("not json", "[1,2]", '{"t": "x"}', '{"v": 99, "t": "x"}', '{"v": 1}', '{"v": 1, "t": 5}'):
        assert decode(bad) is None

def test_validate_client():
    assert validate_client({"t": "input", "move": 4, "attack": True, "eat": False, "junk": 1}) == \
        {"t": "input", "move": 4, "attack": True, "eat": False}
    assert validate_client({"t": "input", "move": 5, "attack": True, "eat": False}) is None
    assert validate_client({"t": "input", "move": 1, "attack": 1, "eat": False}) is None
    assert validate_client({"t": "walk_to", "x": MAP, "y": 0}) is None
    assert validate_client({"t": "walk_to", "x": 3, "y": 4}) == {"t": "walk_to", "x": 3, "y": 4}
    assert validate_client({"t": "login", "name": "bob"}) == {"t": "login", "name": "bob"}
    assert validate_client({"t": "bogus"}) is None

def test_terrain_window_clips_at_origin():
    t = np.full((MAP, MAP), T_GRASS, np.int8)
    x0, y0 = view_window((0, 0))
    rows = terrain_window(t, x0, y0)
    assert len(rows) == VIEW_W and all(len(r) == VIEW_H for r in rows)
    assert rows[0][0] == T_ROCK and rows[-1][-1] == T_GRASS
    assert rows[VIEW_W // 2][VIEW_H // 2] == T_GRASS  # the player's own tile

def _world():
    w = make_empty_world(jax.random.PRNGKey(0))
    p, m, d = w["players"], w["monsters"], w["drops"]
    p = {**p, "active": p["active"].at[0].set(True).at[7].set(True).at[9].set(True),
         "pos": p["pos"].at[0].set(np.array([100, 100])).at[7].set(np.array([103, 99])).at[9].set(np.array([300, 300])),
         "dead": p["dead"].at[7].set(False), "hp": p["hp"].at[0].set(50), "kills": p["kills"].at[0].set(3)}
    m = {**m, "active": m["active"].at[0].set(True), "pos": m["pos"].at[0].set(np.array([102, 102])),
         "kind": m["kind"].at[0].set(1), "hp": m["hp"].at[0].set(MONSTER_HP[1])}
    d = {**d, "active": d["active"].at[0].set(True), "pos": d["pos"].at[0].set(np.array([100, 101])), "item": d["item"].at[0].set(2)}
    w = {**w, "players": p, "monsters": m, "drops": d, "pz": w["pz"].at[100, 100].set(True)}
    return w

def test_snapshot_contents():
    hv = host_view(_world())
    s = snapshot(hv, 0, {0: "me", 7: "bob", 9: "far"}, send_terrain=True, notice="hi", tick=5)
    assert s["t"] == "snap" and s["tick"] == 5 and s["notice"] == "hi"
    you = s["you"]
    assert you["pos"] == [100, 100] and you["hp"] == 50 and you["max_hp"] == PLAYER_MAX_HP
    assert you["kills"] == 3 and you["town"] == 0 and you["safe"] is True and len(you["inv"]) == INV_SLOTS
    kinds = sorted((e["kind"], tuple(e["pos"])) for e in s["ents"])
    assert kinds == [("d", (100, 101)), ("m", (102, 102)), ("p", (100, 100)), ("p", (103, 99))]
    names = {e["name"] for e in s["ents"] if e["kind"] == "p"}
    assert names == {"me", "bob"} and not any("slot" in e for e in s["ents"])
    orc = next(e for e in s["ents"] if e["kind"] == "m")
    assert orc["mkind"] == 1 and orc["max_hp"] == MONSTER_HP[1]
    assert s["terrain"]["x0"] == 100 - VIEW_W // 2 and len(s["terrain"]["rows"]) == VIEW_W
    assert snapshot(hv, 0, {}, send_terrain=False, notice=None, tick=6)["terrain"] is None

def test_snapshot_is_json_serialisable():
    import json
    json.dumps(snapshot(host_view(_world()), 0, {0: "me"}, True, None, 1))
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: net.protocol`

- [ ] **Step 3: Implement `net/protocol.py`** — entity filtering with numpy masks on the window box (`x0 <= x < x0 + VIEW_W`, same for y), then `.tolist()` so every value is a Python int/bool (JSON-safe). Dead players are excluded from `ents`.

- [ ] **Step 4: Run to verify pass** — `5 passed`

- [ ] **Step 5: Commit** — `git add net/protocol.py tests/test_protocol.py && git commit -m "feat(net): protocol, validation and viewport snapshots"`

---

### Task 4: The server

**Files:**
- Create: `net/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: Task 1 (`characters.*`), Task 2 (`find_path`), Task 3 (`decode/encode/validate_client/host_view/snapshot/view_window`), `bots.runner.make_world_batch`, `bots.policies.bot_inputs`, `game.step.step`, `game.state.next_key`, `game.world_layout.temple_tiles, TOWNS`.
- Produces:
  - `class GameServer(seed: int, chars_dir: Path, tick_ms: int = TICK_MS, max_clients: int = MAX_PLAYERS)`.
  - `async start(host: str, port: int) -> None` (listening and ticking when it returns; `port=0` picks a free port, read back from `server.port`); `async stop() -> None` (saves every human, closes sockets, cancels the tick task).
  - `login(name: str) -> tuple[int | None, str | None]`: `(slot, None)` or `(None, "bad name" | "already online" | "server full" | "corrupt character")`. Synchronous (no await between slot pick and mark).
  - `logout(slot: int) -> None`: saves, re-seeds the slot as a bot on a free temple tile of its town, `owner[slot] = "bot"`.
  - `tick_once() -> None`: one tick as in the spec (inputs, step, host view, snapshots queued for sending, save interval, death notices).
  - Attributes read by tests: `owner: list[str | None]` (`None` inactive, `"bot"`, or a name), `tick: int`, `world`, `port`.
  - `main()` CLI: `--host 0.0.0.0 --port 8765 --seed 0 --chars characters --tick-ms TICK_MS`.

Decisions the spec leaves open, fixed here:
- Slot pick order: inactive slot with `slot % NT == town`, then a bot slot in that stripe, then any inactive, then any bot, else `"server full"`. `max_clients` caps concurrent humans (tests set it to 1).
- Free temple tile: first tile of `temple_tiles()[town]` not holding a living active player in the host view.
- Death notice: `prev_dead` per slot; `"you died"` on the tick `dead` turns True.
- Per client: `send_task: asyncio.Task | None`; a new snapshot is only sent when the previous task is done.
- `walk_to`: run `find_path` on the host `walls` copy from `start()`; `None` → `notice = "too far"` (also used for unreachable); path progress is tracked by comparing the player's pos to the expected tile, with `PATH_STUCK_TICKS` consecutive no-progress ticks dropping the path and setting `notice = "blocked"`.

- [ ] **Step 1: Write the failing tests**

```python
import asyncio, json, numpy as np, pytest
import websockets
from game.config import PLAYER_MAX_HP, PROTOCOL_VERSION
from net.server import GameServer

def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 25))

async def connect(server, name):
    ws = await websockets.connect(f"ws://127.0.0.1:{server.port}")
    await ws.send(json.dumps({"v": PROTOCOL_VERSION, "t": "login", "name": name}))
    return ws, json.loads(await ws.recv())

async def next_snap(ws):
    while True:
        m = json.loads(await ws.recv())
        if m["t"] == "snap":
            return m

def test_login_welcome_and_snapshots(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20)
        await s.start("127.0.0.1", 0)
        try:
            ws, w = await connect(s, "Alice")
            assert w["t"] == "welcome" and s.owner[w["slot"]] == "alice"
            snap = await next_snap(ws)
            assert snap["terrain"] is not None and snap["you"]["hp"] == PLAYER_MAX_HP
            assert snap["you"]["safe"] is True  # new characters start on a temple
            await ws.close()
        finally:
            await s.stop()
    run(body())

def test_input_moves_and_two_clients_see_each_other(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20)
        await s.start("127.0.0.1", 0)
        try:
            a, wa = await connect(s, "alice"); b, wb = await connect(s, "bob")
            # bob is placed on alice's town so they share a view: force via the roster order (both new -> towns 0 and 1)
            # so instead move alice's character onto bob's spawn neighbour through the world directly.
            sa, sb = wa["slot"], wb["slot"]
            pb = np.asarray(s.world["players"]["pos"][sb])
            s.world["players"]["pos"] = s.world["players"]["pos"].at[sa].set(pb + np.array([2, 0]))
            snap = await next_snap(b)
            assert "alice" in {e.get("name") for e in snap["ents"]}
            before = (await next_snap(a))["you"]["pos"]
            await a.send(json.dumps({"v": 1, "t": "input", "move": 2, "attack": False, "eat": False}))
            for _ in range(5):
                after = (await next_snap(a))["you"]["pos"]
            assert after[1] > before[1]
            await a.send(json.dumps({"v": 1, "t": "input", "move": 0, "attack": False, "eat": False}))
            await a.close(); await b.close()
        finally:
            await s.stop()
    run(body())

def test_walk_to_reaches_tile(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20)
        await s.start("127.0.0.1", 0)
        try:
            ws, w = await connect(s, "walker")
            you = (await next_snap(ws))["you"]["pos"]
            goal = [you[0] + 3, you[1] + 2]  # inside the temple square, always walkable
            await ws.send(json.dumps({"v": 1, "t": "walk_to", "x": goal[0], "y": goal[1]}))
            pos = None
            for _ in range(15):
                pos = (await next_snap(ws))["you"]["pos"]
                if pos == goal:
                    break
            assert pos == goal
            await ws.close()
        finally:
            await s.stop()
    run(body())

def test_logout_saves_and_slot_returns_to_bot(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20)
        await s.start("127.0.0.1", 0)
        try:
            ws, w = await connect(s, "carol"); slot = w["slot"]
            await next_snap(ws); await ws.close()
            for _ in range(20):
                if s.owner[slot] == "bot": break
                await asyncio.sleep(0.02)
            assert s.owner[slot] == "bot" and (tmp_path / "carol.npz").exists()
            ws2, w2 = await connect(s, "carol")
            assert w2["t"] == "welcome"
            await ws2.close()
        finally:
            await s.stop()
    run(body())

def test_refusals(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20, max_clients=1)
        await s.start("127.0.0.1", 0)
        try:
            a, wa = await connect(s, "dave"); assert wa["t"] == "welcome"
            b, wb = await connect(s, "dave"); assert wb["t"] == "error" and "online" in wb["msg"]
            c, wc = await connect(s, "erin"); assert wc["t"] == "error" and "full" in wc["msg"]
            d, wd = await connect(s, "x"); assert wd["t"] == "error" and "name" in wd["msg"]
            (tmp_path / "frank.npz").write_bytes(b"junk")
            await a.close()
            await asyncio.sleep(0.1)
            e, we = await connect(s, "frank"); assert we["t"] == "error" and "corrupt" in we["msg"]
            ws = await websockets.connect(f"ws://127.0.0.1:{s.port}")
            await ws.send("garbage")
            assert json.loads(await ws.recv())["t"] == "error"
        finally:
            await s.stop()
    run(body())

def test_two_logins_get_distinct_slots(tmp_path):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20)
    s.prepare()  # builds the world without starting sockets
    a, _ = s.login("gina"); b, _ = s.login("hank")
    assert a is not None and b is not None and a != b
    assert s.owner[a] == "gina" and s.owner[b] == "hank"

def test_disconnect_during_play_frees_slot(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20)
        await s.start("127.0.0.1", 0)
        try:
            ws, w = await connect(s, "ivan"); slot = w["slot"]
            await ws.send(json.dumps({"v": 1, "t": "input", "move": 1, "attack": True, "eat": False}))
            await next_snap(ws)
            await ws.transport.close() if hasattr(ws, "transport") else ws.close()
            t0 = s.tick
            for _ in range(30):
                await asyncio.sleep(0.02)
                if s.owner[slot] == "bot": break
            assert s.owner[slot] == "bot" and s.tick > t0  # the loop kept ticking
        finally:
            await s.stop()
    run(body())
```

Note for the implementer: `prepare()` is a public synchronous method that builds the world, jits, and loads the roster; `start()` calls it when it has not run.

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: net.server`

- [ ] **Step 3: Implement `net/server.py`** — `websockets.serve` handler per connection; the tick loop is `asyncio.create_task(self._tick_loop())` using `loop.time()` deadlines (`deadline += tick_ms / 1000`; if behind by more than one tick, skip ahead and count an overrun). `tick_once` is synchronous and runs on the loop thread (the JAX call for one CPU world is a few ms). Snapshot sends are `asyncio.create_task(ws.send(encode(snap)))` stored per client. Logging via `logging` at INFO: logins, logouts, overruns, saves.

- [ ] **Step 4: Run to verify pass** — `venv\Scripts\python.exe -m pytest tests/test_server.py -q` → `7 passed` in under 30 s.

- [ ] **Step 5: Run the whole suite** — `venv\Scripts\python.exe -m pytest tests -q` → all pass.

- [ ] **Step 6: Commit** — `git add net/server.py tests/test_server.py && git commit -m "feat(net): asyncio game server with bots, persistence and click-to-walk"`

---

### Task 5: Threaded client and `play.py` as a network client

**Files:**
- Create: `net/client.py`
- Modify: `play.py` (remove the local world and bots; keep drawing)
- Test: `tests/test_client.py`

**Interfaces:**
- Produces `class NetClient(host: str, port: int, name: str)` (a `threading.Thread`, daemon): `start()`; `welcome: dict | None`; `error: str | None`; `connected: bool`; `latest() -> dict | None` (newest `snap`, older ones discarded); `send_input(move: int, attack: bool, eat: bool) -> None`; `walk_to(x: int, y: int) -> None`; `close() -> None`. Sends go through a thread-safe queue drained by the asyncio loop in the thread.
- Consumes: Task 3 `encode/decode`, Task 4 `GameServer` (tests only).

- [ ] **Step 1: Write the failing tests**

```python
import asyncio, threading, time
from net.client import NetClient
from net.server import GameServer

def _serve(tmp_path):
    ready = threading.Event(); holder = {}
    def run():
        async def body():
            s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20)
            await s.start("127.0.0.1", 0); holder["s"] = s; ready.set()
            await holder["stop"]
            await s.stop()
        loop = asyncio.new_event_loop(); holder["loop"] = loop
        holder["stop"] = loop.create_future()
        loop.run_until_complete(body())
    t = threading.Thread(target=run, daemon=True); t.start(); ready.wait(10)
    return holder, t

def _wait(pred, secs=5):
    t0 = time.time()
    while time.time() - t0 < secs:
        if pred(): return True
        time.sleep(0.02)
    return False

def test_client_logs_in_and_moves(tmp_path):
    h, t = _serve(tmp_path)
    try:
        c = NetClient("127.0.0.1", h["s"].port, "zed"); c.start()
        assert _wait(lambda: c.welcome is not None) and c.connected
        assert _wait(lambda: c.latest() is not None)
        y0 = c.latest()["you"]["pos"][1]
        c.send_input(2, False, False)
        assert _wait(lambda: c.latest()["you"]["pos"][1] > y0)
        c.send_input(0, False, False)
        c.close()
        assert _wait(lambda: not c.connected)
    finally:
        h["loop"].call_soon_threadsafe(h["stop"].set_result, None); t.join(10)

def test_client_reports_error(tmp_path):
    h, t = _serve(tmp_path)
    try:
        c = NetClient("127.0.0.1", h["s"].port, "x"); c.start()
        assert _wait(lambda: c.error is not None) and "name" in c.error
    finally:
        h["loop"].call_soon_threadsafe(h["stop"].set_result, None); t.join(10)
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: net.client`

- [ ] **Step 3: Implement `net/client.py`** — `asyncio.run(self._main())` inside `run()`; `_main` connects, sends `login`, awaits the first reply (`welcome` → set, `error` → set and return), then runs a receive loop and a send loop (`asyncio.Queue` fed via `loop.call_soon_threadsafe`) until closed; `latest()` guarded by a lock holding one dict; `connected` False once the socket closes.

- [ ] **Step 4: Run to verify pass** — `2 passed`

- [ ] **Step 5: Rewrite `play.py` on top of `NetClient`** — keep `TILE`, `HUD_H`, every `draw_*` function and the canvas setup; delete `make_world_batch`, `bot_inputs`, `step`, `reset`, `toggle_pause`, `R`/`P` keys and the JAX imports. `Game.__init__(root, client)`: the frame timer `root.after(welcome["tick_ms"], self.frame)` draws `client.latest()`; terrain drawn from the last non-null `snap["terrain"]` (cached) using `x0, y0` from it, entities from `ents` (`"p"` with `name` drawn in 9-pt text above the tile, `"m"` by `mkind`, `"d"` by `item`), the human from `you.pos` at the window centre; HUD line 1 from `you` (`HP x/max  kills  deaths  items  town N  SAFE`) plus `notice`; HUD line 2 shows controls, or `DISCONNECTED` when `client.connected` is False. Keys: on press/release recompute `(move, attack, eat)` from `held`/`attacking`/`eating` and call `send_input` only when the tuple changed; `<Button-1>` converts pixels to a tile via `x0 + (px - off_x) // TILE` and calls `walk_to`. Argparse: `--host 127.0.0.1 --port 8765 --name` (default `getpass.getuser()` lower-cased). The window's `view_w/view_h` are taken from `welcome` (the server's `VIEW_W/VIEW_H`), with `off_x` centring the strip; tiles outside the window are not drawn.

- [ ] **Step 6: Smoke run (headless check only)** — `venv\Scripts\python.exe -c "import ast,sys; ast.parse(open('play.py').read())"` passes, and `venv\Scripts\python.exe -m pytest tests -q` all pass. The lead cannot see the window: say so in the report; the user checks it.

- [ ] **Step 7: Commit** — `git add net/client.py tests/test_client.py play.py && git commit -m "feat(net): threaded client; play.py becomes a network client"`

---

### Task 6: Load bench and docs

**Files:**
- Create: `bench/server_load.py`
- Modify: `docs/handoff.md`, `docs/brief.md`

**Interfaces:**
- Consumes: `NetClient` (Task 5), `GameServer` (Task 4).

- [ ] **Step 1: Write `bench/server_load.py`** — argparse `--clients 100 --seconds 20 --host 127.0.0.1 --port 8765` (connects to a running server), or `--embedded` to start a `GameServer` in-process on a free port. Each client is a `NetClient` named `load_<i>` that every 0.5 s sends a random `input`. At the end print: connected clients, snapshots received per client per second (mean/min), bytes received per client per second, and, in embedded mode, the server's overrun count (expose `GameServer.overruns: int` in Task 4 if missing — it is an attribute the tick loop increments).

- [ ] **Step 2: Run it embedded** — `venv\Scripts\python.exe bench/server_load.py --embedded --clients 50 --seconds 10` prints the summary without exceptions. Record the numbers in the report.

- [ ] **Step 3: Update docs** — `docs/brief.md`: a `## Networking (net/)` section listing the four modules and their public functions as in the Interfaces blocks above, and the protocol's message types in one table. `docs/handoff.md`: State (test count, how to run server and client: `venv\Scripts\python.exe -m net.server` and `venv\Scripts\python.exe play.py --name you`), a "Milestone 2" section (what persists, slot order, what is not done: passwords, chat, world persistence), the load-bench numbers, and the open-work list with milestone 3 next.

- [ ] **Step 4: Full suite** — `venv\Scripts\python.exe -m pytest tests -q` → all pass.

- [ ] **Step 5: Commit** — `git add bench/server_load.py docs/handoff.md docs/brief.md && git commit -m "bench(net): load client; docs for milestone 2"`
