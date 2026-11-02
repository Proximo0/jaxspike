"""Asyncio WebSocket game server: one shared JAX world, human players in bot slots.

Run from the project folder:  venv\\Scripts\\python.exe -m net.server --port 8765
"""
import argparse
import asyncio
import logging
import os
import signal
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import websockets  # noqa: E402

from bots.policies import bot_inputs  # noqa: E402
from bots.runner import make_world_batch  # noqa: E402
from game.config import (DIR_DELTAS, INV_SLOTS, MAP, MAX_PLAYERS, PATH_STUCK_TICKS, PLAYER_MAX_HP,  # noqa: E402
                         SAVE_EVERY_TICKS, TICK_MS, VIEW_H, VIEW_W)
from game.state import next_key  # noqa: E402
from game.step import step  # noqa: E402
from game.world_layout import TOWNS, temple_tiles  # noqa: E402
from net import characters  # noqa: E402
from net.pathing import find_path  # noqa: E402
from net.protocol import decode, encode, host_view, snapshot, validate_client, view_window  # noqa: E402

log = logging.getLogger("net.server")

NT = len(TOWNS)
_bot_inputs_jit = jax.jit(bot_inputs)
_step_jit = jax.jit(step)
_world_cache: dict[int, dict] = {}  # seed -> populated world; jax arrays are immutable so sharing is safe


def _initial_world(seed: int) -> dict:
    """Build (once per process and seed) the populated unbatched world."""
    if seed not in _world_cache:
        batched = make_world_batch(seed, 1)
        _world_cache[seed] = jax.tree_util.tree_map(lambda x: x[0], batched)
    return _world_cache[seed]


class Client:
    """One connected human."""

    def __init__(self, ws):
        self.ws = ws
        self.send_task: asyncio.Task | None = None


class GameServer:
    def __init__(self, seed: int, chars_dir: Path, tick_ms: int = TICK_MS, max_clients: int = MAX_PLAYERS):
        self.seed = seed
        self.chars_dir = Path(chars_dir)
        self.tick_ms = tick_ms
        self.max_clients = max_clients
        self.tick = 0
        self.overruns = 0
        self.port = 0
        self.world = None
        self.walls = None
        self.roster: dict[str, int] = {}
        self.owner: list[str | None] = [None] * MAX_PLAYERS
        self.conn: dict[int, Client] = {}
        self.latest_input = {}
        self.path: dict[int, deque] = {}
        self.pending: dict[int, tuple | None] = {}
        self.stuck: dict[int, int] = {}
        self.last_window: dict[int, tuple] = {}
        self.notice: dict[int, str] = {}
        self.prev_dead: dict[int, bool] = {}
        self._server = None
        self._tick_task: asyncio.Task | None = None
        self._prepared = False

    # ----- setup -----

    def prepare(self) -> None:
        """Synchronous: build the world, warm the jitted functions, load the roster."""
        if self._prepared:
            return
        self.world = _initial_world(self.seed)
        self.walls = np.asarray(self.world["walls"])
        active = np.asarray(self.world["players"]["active"])
        self.owner = ["bot" if a else None for a in active]
        inputs, _ = self._bot_inputs()  # also compiles; the advanced rng is dropped so the world is untouched
        _step_jit(self.world, inputs)
        self.chars_dir.mkdir(parents=True, exist_ok=True)
        self.roster = characters.load_roster(self.chars_dir)
        self._prepared = True

    async def start(self, host: str, port: int) -> None:
        self.prepare()
        self._server = await websockets.serve(self._handler, host, port)
        self.port = self._server.sockets[0].getsockname()[1]
        self._tick_task = asyncio.create_task(self._tick_loop())
        log.info("listening on %s:%d", host, self.port)

    async def stop(self) -> None:
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None
        for slot in list(self.conn):
            self.logout(slot)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ----- slots and characters -----

    def _humans(self) -> list[int]:
        return [s for s, o in enumerate(self.owner) if o not in (None, "bot")]

    def _pick_slot(self, town: int) -> int | None:
        stripe = range(town, MAX_PLAYERS, NT)
        for pool, want in ((stripe, None), (stripe, "bot"), (range(MAX_PLAYERS), None), (range(MAX_PLAYERS), "bot")):
            for s in pool:
                if self.owner[s] == want:
                    return s
        return None

    def _free_temple_tile(self, town: int) -> tuple[int, int]:
        p = self.world["players"]
        pos, live = np.asarray(p["pos"]), np.asarray(p["active"] & ~p["dead"])
        taken = {(int(x), int(y)) for (x, y) in pos[live]}
        tiles = np.asarray(temple_tiles())[town]
        for x, y in tiles:
            if (int(x), int(y)) not in taken:
                return int(x), int(y)
        return int(tiles[0][0]), int(tiles[0][1])

    def _write_slot(self, slot: int, pos, spawn, hp: int, inv, kills: int, deaths: int) -> None:
        p = self.world["players"]
        upd = {"pos": pos, "spawn": spawn, "hp": hp, "inv": inv, "kills": kills, "deaths": deaths,
               "active": True, "dead": False, "attack_cd": 0}
        newp = {**p, **{k: p[k].at[slot].set(jnp.asarray(v, p[k].dtype)) for k, v in upd.items()}}
        self.world = {**self.world, "players": newp}

    def login(self, name: str) -> tuple[int | None, str | None]:
        """Pick and mark a slot for `name`. Synchronous: nothing can interleave between pick and mark."""
        key = characters.valid_name(name)
        if key is None:
            return None, "bad name"
        if key in self.owner:
            return None, "already online"
        if len(self._humans()) >= self.max_clients:
            return None, "server full"
        char = characters.load(self.chars_dir, key)
        if char is None and (self.chars_dir / f"{key}.npz").exists():
            return None, "corrupt character"
        town = char["town"] if char is not None else characters.next_town(self.roster)
        slot = self._pick_slot(town)
        if slot is None:
            return None, "server full"
        if char is None:
            char = characters.new_character(town, self._free_temple_tile(town))
            self.roster[key] = town
            try:
                characters.save_roster(self.chars_dir, self.roster)
            except OSError:
                log.exception("roster save failed")
        self._write_slot(slot, char["pos"], char["spawn"], char["hp"], char["inv"], char["kills"], char["deaths"])
        self.owner[slot] = key
        self.latest_input[slot] = (0, False, False)
        self.path[slot] = deque()
        self.pending[slot] = None
        self.stuck[slot] = 0
        self.last_window.pop(slot, None)
        self.notice.pop(slot, None)
        self.prev_dead[slot] = False
        log.info("login %s slot %d town %d", key, slot, town)
        return slot, None

    def _fields(self, slot: int) -> dict:
        p = self.world["players"]
        dead = bool(p["dead"][slot])
        spawn = np.asarray(p["spawn"][slot], np.int32)
        return {"town": slot % NT, "pos": spawn.copy() if dead else np.asarray(p["pos"][slot], np.int32),
                "spawn": spawn, "hp": PLAYER_MAX_HP if dead else int(p["hp"][slot]),
                "inv": np.asarray(p["inv"][slot], np.int32)[:INV_SLOTS],
                "kills": int(p["kills"][slot]), "deaths": int(p["deaths"][slot])}

    def _save_slot(self, slot: int) -> None:
        name = self.owner[slot]
        try:
            characters.save(self.chars_dir, name, self._fields(slot))
        except Exception:
            log.exception("save failed for %s", name)

    def logout(self, slot: int) -> None:
        """Save the human, hand the slot back to the bots, drop its connection state."""
        name = self.owner[slot]
        if name in (None, "bot"):
            return
        self._save_slot(slot)
        town = slot % NT
        tile = self._free_temple_tile(town)
        self._write_slot(slot, tile, tile, PLAYER_MAX_HP, np.zeros(INV_SLOTS, np.int32), 0, 0)
        self.owner[slot] = "bot"
        for d in (self.latest_input, self.path, self.pending, self.stuck, self.last_window, self.notice,
                  self.prev_dead):
            d.pop(slot, None)
        c = self.conn.pop(slot, None)
        if c is not None:
            asyncio.ensure_future(self._close(c.ws))
        log.info("logout %s slot %d", name, slot)

    @staticmethod
    async def _close(ws) -> None:
        try:
            await ws.close()
        except Exception:
            pass

    # ----- tick -----

    def _bot_inputs(self) -> tuple[dict, dict]:
        """Bot inputs for the next tick and the world with its rng advanced."""
        key, advanced = next_key(self.world)
        return _bot_inputs_jit(advanced, key), advanced

    def tick_once(self) -> None:
        """One game tick: inputs, step, host view, snapshots, periodic save, death notices."""
        inputs, self.world = self._bot_inputs()
        move, attack, eat = (np.array(inputs[k]) for k in ("move", "attack", "eat"))
        pos = np.asarray(self.world["players"]["pos"])
        for slot in self._humans():
            m, a, e = self.latest_input[slot]
            attack[slot], eat[slot] = a, e
            move[slot] = self._path_move(slot, tuple(int(v) for v in pos[slot]), m)
        inputs = {"move": jnp.asarray(move, jnp.int32), "attack": jnp.asarray(attack), "eat": jnp.asarray(eat)}
        self.world = _step_jit(self.world, inputs)
        self.tick += 1
        hv = host_view(self.world)
        names = {s: self.owner[s] for s in self._humans()}
        for slot in list(self.conn):
            self._queue_snapshot(slot, hv, names)
        if self.tick % SAVE_EVERY_TICKS == 0:
            for slot in self._humans():
                self._save_slot(slot)
            log.info("saved %d characters at tick %d", len(self._humans()), self.tick)

    def _path_move(self, slot: int, pos: tuple, default: int) -> int:
        """Move code for a human this tick: the next path step if walking, else the held input."""
        q = self.path[slot]
        if not q:
            return default
        pend = self.pending[slot]
        if pend is not None:
            if pos == pend:
                q.popleft()
                self.stuck[slot] = 0
            else:
                self.stuck[slot] += 1
                if self.stuck[slot] >= PATH_STUCK_TICKS:
                    q.clear()
                    self.notice[slot] = "blocked"
        if not q:
            self.pending[slot] = None
            return default
        dx, dy = DIR_DELTAS[q[0]]
        self.pending[slot] = (pos[0] + dx, pos[1] + dy)
        return q[0]

    def _queue_snapshot(self, slot: int, hv: dict, names: dict) -> None:
        dead = bool(hv["players"]["dead"][slot])
        if dead and not self.prev_dead.get(slot, False):
            self.notice[slot] = "you died"
        self.prev_dead[slot] = dead
        c = self.conn[slot]
        if c.send_task is not None and not c.send_task.done():
            return  # slow client: skip this tick, never wait
        window = view_window(hv["players"]["pos"][slot])
        snap = snapshot(hv, slot, names, self.last_window.get(slot) != window, self.notice.pop(slot, None),
                        self.tick)
        self.last_window[slot] = window
        c.send_task = asyncio.create_task(self._send(c, encode(snap)))

    async def _send(self, c: Client, text: str) -> None:
        try:
            await c.ws.send(text)
        except websockets.ConnectionClosed:
            pass  # the handler sees the close and logs the slot out
        except Exception:
            log.exception("send failed")

    async def _tick_loop(self) -> None:
        loop = asyncio.get_running_loop()
        dt = self.tick_ms / 1000
        deadline = loop.time() + dt
        while True:
            delay = deadline - loop.time()
            await asyncio.sleep(max(delay, 0))
            try:
                self.tick_once()
            except Exception:
                log.exception("tick failed")
            deadline += dt
            now = loop.time()
            if now > deadline + dt:
                self.overruns += 1
                log.warning("tick %d overrun, %.0f ms behind", self.tick, (now - deadline) * 1000)
                deadline = now + dt

    # ----- connections -----

    async def _refuse(self, ws, msg: str) -> None:
        try:
            await ws.send(encode({"t": "error", "msg": msg}))
        except websockets.ConnectionClosed:
            pass
        await self._close(ws)

    async def _handler(self, ws) -> None:
        slot = None
        try:
            try:
                raw = await ws.recv()
            except websockets.ConnectionClosed:
                return
            first = decode(raw) if isinstance(raw, str) else None
            first = validate_client(first) if first is not None else None
            if first is None or first["t"] != "login":
                await self._refuse(ws, "expected login")
                return
            slot, err = self.login(first["name"])
            if err is not None:
                await self._refuse(ws, err)
                return
            c = Client(ws)
            self.conn[slot] = c
            welcome = {"t": "welcome", "slot": slot, "map": MAP, "tick_ms": self.tick_ms,
                       "view_w": VIEW_W, "view_h": VIEW_H}
            c.send_task = asyncio.create_task(self._send(c, encode(welcome)))
            async for raw in ws:
                msg = decode(raw) if isinstance(raw, str) else None
                if msg is None:
                    await self._refuse(ws, "malformed message")
                    return
                if msg["t"] in ("input", "walk_to"):
                    msg = validate_client(msg)
                    if msg is None:
                        await self._refuse(ws, "malformed message")
                        return
                    self._on_message(slot, msg)
        except websockets.ConnectionClosed:
            pass
        except Exception:
            log.exception("handler failed")
        finally:
            if slot is not None and self.conn.get(slot) is not None and self.conn[slot].ws is ws:
                self.logout(slot)

    def _on_message(self, slot: int, msg: dict) -> None:
        if msg["t"] == "input":
            self.latest_input[slot] = (msg["move"], msg["attack"], msg["eat"])
            self.path[slot].clear()
            self.pending[slot] = None
        elif msg["t"] == "walk_to":
            p = self.world["players"]["pos"][slot]
            start = (int(p[0]), int(p[1]))
            path = find_path(self.walls, start, (msg["x"], msg["y"]))
            if path is None:
                self.notice[slot] = "too far"
                return
            self.path[slot] = deque(path)
            self.pending[slot] = None
            self.stuck[slot] = 0


def main() -> None:
    ap = argparse.ArgumentParser(description="JAX tile MMO game server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chars", default="characters")
    ap.add_argument("--tick-ms", type=int, default=TICK_MS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    async def run() -> None:
        server = GameServer(args.seed, Path(args.chars), args.tick_ms)
        await server.start(args.host, args.port)
        stop = asyncio.Event()
        try:
            asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stop.set)
        except NotImplementedError:
            pass  # Windows: KeyboardInterrupt cancels the main task instead
        try:
            await stop.wait()
        finally:
            await server.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
