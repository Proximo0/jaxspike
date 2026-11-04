"""Asyncio WebSocket game server: one shared JAX world, human players in bot slots.

Run from the project folder:  venv\\Scripts\\python.exe -m net.server --port 8765
GPU (WSL):  ~/jaxgpu/bin/python -m net.server --platform gpu   (sets JAX_PLATFORMS and the 80% memory cap)
"""
import argparse
import asyncio
import gc
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path



def _argv_platform(argv: list[str]) -> str:
    """The --platform value (cpu or gpu) from the command line; read before jax is imported."""
    for i, a in enumerate(argv):
        if a == "--platform" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--platform="):
            return a.split("=", 1)[1]
    return "cpu"


if __name__ == "__main__" and _argv_platform(sys.argv) == "gpu":
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")  # the project's GPU cap: never above 80%
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
try:
    import websockets  # noqa: E402
except ImportError:  # --bench-ticks opens no socket: it runs in a venv without websockets (the WSL GPU venv)
    websockets = None

from bots.ro_policies import ro_bot_inputs as bot_inputs  # noqa: E402
from bots.runner import make_world_batch  # noqa: E402
from game import ro_anim, ro_craft, ro_items, ro_mvp, ro_party, ro_skills, ro_stats  # noqa: E402
from game.config import (DIR_DELTAS, INV_SLOTS, MAX_PLAYERS, MAX_VIEWERS, PATH_STUCK_TICKS,  # noqa: E402
                         PLAYER_MAX_HP, SAVE_EVERY_TICKS, TICK_MS)
from game.data import items as item_data  # noqa: E402
from game.ro_move import LEGACY_TO_RO  # noqa: E402
from game.state import empty_inputs, next_key  # noqa: E402
from game.step import step  # noqa: E402
from game.view import changed_mask, entity_table, view_update  # noqa: E402
from game.world_layout import TOWNS, temple_tiles, town_boxes  # noqa: E402
from net import characters  # noqa: E402
from net.accounts import AccountStore  # noqa: E402
from net.chat import ChatRouter  # noqa: E402
from net.kafra import new_storage, transfer  # noqa: E402
from net.pathing import find_path  # noqa: E402
from net.protocol import (build_snaps, build_welcome_static, decode, encode, encode_map,  # noqa: E402
                          validate_client, welcome_message, you_delta, you_full)

try:  # the persist worker's RO-block (de)serialiser; without it saves keep the legacy shape
    from net.char_fields import RO_KEYS, STAT_SCALARS, character_fields, fresh_ro_fields
except ImportError:  # pragma: no cover
    RO_KEYS = STAT_SCALARS = character_fields = fresh_ro_fields = None

log = logging.getLogger("net.server")

NT = len(TOWNS)
_PLAYER_PULL = ("pos", "spawn", "hp", "dead", "active", "inv", "kills", "deaths")
# RO block rows in the one per-tick host pull (docs/*_wire.md). The key sets are what net.char_fields.character_fields
# reads, so the same dict feeds both the snapshots and the periodic save.
_STATS_PULL = ("str", "agi", "vit", "int", "dex", "luk", "base_level", "job_level", "base_exp", "job_exp", "job",
               "stat_points", "skill_points", "sp", "max_sp", "was_dead")
_SKILLS_PULL = ("learned", "sp", "cooldown", "buffs", "buff_level", "casting", "cast_left", "cast_target",
                "after_delay")
_ITEMS_PULL = ("inv_count", "inv_refine", "inv_cards", "equip", "zeny")
_CRAFT_PULL = ("level", "exp", "gathers", "crafts")
_PARTY_PULL = ("party_id", "leader")
# Per-slot rows reset when a slot changes hands (login, logout): block -> keys (None = every key of the block).
_SLOT_ROWS = {"stats": None, "skills": None, "items": None, "craft": ("level", "exp", "gathers", "crafts"),
              "party": ("party_id", "leader", "invite_from", "invite_ttl")}
# One-shot pending human inputs (everything the RO systems read except move/attack/eat, which have their own path).
_RO_INPUT_KEYS = tuple(k for k in empty_inputs() if k not in ("move", "attack", "eat"))
MAX_CHAT_PER_TICK = 256
MAX_OUTBOX = 128  # queued extra messages (chat, mvp) per client while a snapshot send is in flight


def _ro_blocks(world) -> dict:
    """The RO block rows the host reads each tick (snapshots, chat routing, periodic save): one dict, one pull."""
    return {"stats": {k: world["stats"][k] for k in _STATS_PULL},
            "skills": {k: world["skills"][k] for k in _SKILLS_PULL},
            "items": {**{k: world["items"][k] for k in _ITEMS_PULL}, "range": ro_items.weapon_range(world)},
            "craft": {k: world["craft"][k] for k in _CRAFT_PULL},
            "party": {k: world["party"][k] for k in _PARTY_PULL}}


def _tick_fn(world, hmask, hmove, hattack, heat, hro, prev_tab, known, viewer_slot):
    """One tick on device: bot inputs, human overrides, step, entity table, delta mask, per-viewer view update.

    `hro` holds the humans' one-shot RO inputs (allot, job, skill, learn, target, use_item, equip, refine, drop,
    gather, craft, party, save) as arrays over all slots; they replace the bot policy's values on human slots only.
    Returns (world, tab, known_next, pull); `pull` is the only thing the host reads back."""
    key, world = next_key(world)
    bots = bot_inputs(world, key)
    none = empty_inputs()
    inputs = {k: jnp.where(hmask, hro.get(k, none[k]), bots[k]) for k in bots}
    inputs.update({"move": jnp.where(hmask, hmove, bots["move"]), "attack": jnp.where(hmask, hattack, bots["attack"]),
                   "eat": jnp.where(hmask, heat, bots["eat"])})
    z = jnp.zeros((MAX_PLAYERS,), jnp.bool_)
    inputs["sit"] = jnp.where(hmask, hro.get("sit", z), bots.get("sit", z))
    m1 = jnp.full((MAX_PLAYERS,), -1, jnp.int32)  # ro3-world3: kafra teleport destination town (-1 = none)
    inputs["teleport"] = jnp.where(hmask, hro.get("teleport", m1), bots.get("teleport", m1))
    pos_before = world["players"]["pos"]
    world = step(world, inputs)
    tab = entity_table(world)
    send, gone, known_next = view_update(tab, changed_mask(prev_tab, tab), viewer_slot, known)
    p = world["players"]
    d = ro_stats.derived(world)
    pull = {"send": send, "gone": gone, "tab": tab, "players": {k: p[k] for k in _PLAYER_PULL},
            "safe": world["pz"][p["pos"][:, 0], p["pos"][:, 1]],
            "warped": jnp.max(jnp.abs(p["pos"] - pos_before), axis=1) > 1,  # ro3-wire: a warp/teleport/respawn moved the player this tick (snap.you.warped)
            "ro": {**_ro_blocks(world), "derived": {"max_hp": d["max_hp"], "max_sp": d["max_sp"]}},
            "anim": {"amotion_ms": ro_anim.amotion_ms(world, d)[:MAX_PLAYERS]},  # snap.you (docs/anim_wire.md)
            "mvp": {"event": world["mvp"]["event"], "event_head": world["mvp"]["event_head"]}}
    return world, tab, known_next, pull


_tick_jit = jax.jit(_tick_fn)


def _reset_rows_fn(world, fresh, slot, kills, deaths):
    """Give `slot` fresh RO rows (a new Novice, empty inventory, no party); the stats ledgers start at kills/deaths so
    the first tick sees a zero delta (docs/stats_wire.md)."""
    for blk, keys in _SLOT_ROWS.items():
        cur = world[blk]
        new = {k: cur[k].at[slot].set(fresh[blk][k][slot]) for k in (keys or cur)}
        world = {**world, blk: {**cur, **new}}
    st = world["stats"]
    st = {**st, "kills_seen": st["kills_seen"].at[slot].set(kills), "deaths_seen": st["deaths_seen"].at[slot].set(deaths)}
    return {**world, "stats": st}


_reset_rows = jax.jit(_reset_rows_fn)


def _login_rows_fn(world, fresh, slot, f):
    """Everything a login writes into `slot` in one device call: fresh RO rows, then the character's fields
    (net.char_fields key names), then the live flags. `slot` is traced, so there is one compile for all slots."""
    world = _reset_rows_fn(world, fresh, slot, f["kills"], f["deaths"])

    def put(world, blk, key, val):
        old = world[blk][key]
        return {**world, blk: {**world[blk], key: old.at[slot].set(jnp.asarray(val, old.dtype))}}

    for k in ("pos", "spawn", "hp", "inv", "kills", "deaths"):
        world = put(world, "players", k, f[k])
    for k, v in (("active", True), ("dead", False), ("attack_cd", 0)):
        world = put(world, "players", k, v)
    for k in STAT_SCALARS:
        world = put(world, "stats", k, f[f"stats_{k}"])
    world = put(world, "skills", "learned", f["skills_learned"])
    world = put(world, "skills", "sp", f["skills_sp"])
    for k in ("inv_count", "inv_refine", "inv_cards", "equip", "zeny"):
        world = put(world, "items", k, f[f"items_{k}"])
    for k in ("level", "exp", "gathers", "crafts"):
        world = put(world, "craft", k, f[f"craft_{k}"])
    return world


_login_rows = jax.jit(_login_rows_fn) if RO_KEYS is not None else None
_NONE_INPUTS = {k: np.asarray(v) for k, v in empty_inputs().items()}
if "sit" not in _NONE_INPUTS:  # ro3-wire: until game/state.py carries the key
    _NONE_INPUTS["sit"] = np.zeros(MAX_PLAYERS, bool)
if "teleport" not in _NONE_INPUTS:  # ro3-world3's kafra teleport input, same deal
    _NONE_INPUTS["teleport"] = np.full(MAX_PLAYERS, -1, np.int32)
_RO_INPUT_KEYS = tuple(_RO_INPUT_KEYS) + tuple(k for k in ("sit", "teleport") if k not in _RO_INPUT_KEYS)
_TOWN_BOXES = np.asarray(town_boxes())
_RO_MESSAGES = frozenset(("allot", "job", "skill", "learn", "use_item", "equip", "refine", "drop", "gather", "craft",
                          "party", "chat", "storage", "save", "buy", "sell", "sit", "teleport"))
PARTY_OP = {"create": 1, "leave": 2, "invite": 3, "accept": 4}


def _fresh_blocks() -> dict:
    return {"stats": ro_stats.make_state(), "skills": ro_skills.make_state(), "items": ro_items.make_state(),
            "craft": ro_craft.make_state(), "party": ro_party.make_state()}
_world_cache: dict[int, dict] = {}  # seed -> populated world; jax arrays are immutable so sharing is safe


def _initial_world(seed: int) -> dict:
    """Build (once per process and seed) the populated unbatched world."""
    if seed not in _world_cache:
        batched = make_world_batch(seed, 1)
        _world_cache[seed] = jax.tree_util.tree_map(lambda x: x[0], batched)
    return jax.tree_util.tree_map(lambda x: x, _world_cache[seed])  # fresh containers per server


class Client:
    """One connected human."""

    def __init__(self, ws, fmt: str = "json"):
        self.ws = ws
        self.fmt = fmt  # snap encoding chosen at login: "json" (text frames) or "bin" (binary frames)
        self.send_task: asyncio.Task | None = None
        self.outbox: list[str] = []  # extra encoded messages (chat, mvp) sent ahead of the next snapshot
        self.last_you: dict = {}  # `you` values last sent (net.protocol.you_delta)


class GameServer:
    def __init__(self, seed: int, chars_dir: Path, tick_ms: int = TICK_MS, max_clients: int = MAX_PLAYERS,
                 require_password: bool = False, stats_every: int = 100, pipeline: bool = True):
        self.seed = seed
        self.pipeline = pipeline  # _tick_loop overlaps the device tick with the previous tick's snapshots
        self._inflight: tuple | None = None  # (pull on device, viewer, chat) of the dispatched, uncollected tick
        self.stats_every = stats_every  # ticks between "max tick duration" log lines
        self.require_password = require_password  # False: a login without password still works for name-only characters
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
        self.tap: dict[int, list] = {}  # slot -> [move, attack, eat] set by a message, consumed by the next tick
        self.maps = None  # (terrain_b64, pz_b64) for welcome, built once
        self.prev_tab = None  # previous entity table (device)
        self.known = None  # bool [MAX_VIEWERS, N_ENTITIES] (device), row = slot
        self.pos_host = None  # numpy copy of player positions after the last tick (for click-to-walk bookkeeping)
        self.need_full: dict[int, bool] = {}
        self._walked: set = set()
        self._max_tick_s = 0.0
        self._saving: set = set()
        self._phase_max = [0.0, 0.0, 0.0]  # pipelined loop: max collect wait, dispatch, publish (s) per stats line
        self.save_gen: dict[str, int] = {}  # name -> generation of the newest initiated save (event loop writes only)
        self.path: dict[int, deque] = {}
        self.pending: dict[int, tuple | None] = {}
        self.stuck: dict[int, int] = {}
        self.notice: dict[int, str] = {}
        self.prev_dead: dict[int, bool] = {}
        self.town: dict[int, int] = {}  # each human's own home town (may differ from slot % NT)
        self._closing: set = set()
        self._sending: set = set()
        self._server = None
        self._tick_task: asyncio.Task | None = None
        self._prepared = False
        # RO plumbing (docs/*_wire.md)
        self.accounts: AccountStore | None = None
        self.welcome_static: dict | None = None
        self.pend = {k: _NONE_INPUTS[k].copy() for k in _RO_INPUT_KEYS}  # one-shot human inputs, cleared per tick
        self.chat_msgs: list[dict] = []
        self.router = ChatRouter()
        self.storage_ops: list[tuple] = []
        self.storage: dict[int, np.ndarray] = {}  # slot -> kafra storage item ids
        self.last_members: dict[int, list] = {}
        self.last_storage: dict[int, bytes] = {}
        self.last_event_head = 0
        self._fresh = None
        self._you_prev: list | None = None  # last tick's `you` source arrays (_dirty_slots)

    # ----- setup -----

    def prepare(self) -> None:
        """Synchronous: build the world, warm the jitted functions, load the roster."""
        if self._prepared:
            return
        self.world = _initial_world(self.seed)
        self.walls = np.asarray(self.world["walls"])
        self.maps = (encode_map(self.world["terrain"], np.int8), encode_map(self.world["pz"], np.uint8))
        self.prev_tab = entity_table(self.world)
        self.known = jnp.zeros((MAX_VIEWERS, self.prev_tab["pos"].shape[0]), bool)
        self.pos_host = np.array(self.world["players"]["pos"])
        active = np.asarray(self.world["players"]["active"])
        self.owner = ["bot" if a else None for a in active]
        z = np.zeros(MAX_PLAYERS)
        _tick_jit(self.world, z.astype(bool), z.astype(np.int32), z.astype(bool), z.astype(bool),
                  {k: v.copy() for k, v in self.pend.items()}, self.prev_tab, self.known,
                  np.full(MAX_VIEWERS, -1, np.int32))  # compiles; results dropped so the world is untouched
        self._fresh = _fresh_blocks()
        _reset_rows(self.world, self._fresh, 0, 0, 0)  # compile
        self._warm_login()
        self.welcome_static = build_welcome_static()
        self.last_event_head = int(self.world["mvp"]["event_head"])
        self.chars_dir.mkdir(parents=True, exist_ok=True)
        self.accounts = AccountStore(self.chars_dir / "accounts")
        self.roster = characters.load_roster(self.chars_dir)
        self._prepared = True
        gc.freeze()  # the world, tables and jit caches are long-lived: keep them out of every gc pass

    async def start(self, host: str, port: int) -> None:
        if websockets is None:
            raise RuntimeError("the websockets package is missing: pip install websockets (requirements.txt)")
        self.prepare()
        # compression=None: permessage-deflate cost 27% of the loop at 480 clients (zlib per message per client), and
        # the Godot WebSocketPeer does not negotiate it, so every client now gets what Godot clients already got
        self._server = await websockets.serve(self._handler, host, port, backlog=1024, compression=None)
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
        for t in list(self._sending):
            t.cancel()
        await asyncio.gather(*self._closing, *self._sending, *self._saving, return_exceptions=True)

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

    def _tile_taken(self, pos) -> bool:
        p = self.world["players"]
        live = np.asarray(p["active"] & ~p["dead"])
        ppos = np.asarray(p["pos"])[live]
        return bool(((ppos[:, 0] == int(pos[0])) & (ppos[:, 1] == int(pos[1]))).any())

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
        if self.pos_host is not None:
            self.pos_host[slot] = np.asarray(pos, np.int32)

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
        pos = char["pos"]
        if self._tile_taken(pos):
            pos = np.array(self._free_temple_tile(town), np.int32)
        if _login_rows is not None:
            f = {**fresh_ro_fields(char["inv"]), **{k: char[k] for k in char if k in RO_KEYS}, **{
                k: char[k] for k in ("spawn", "hp", "inv", "kills", "deaths")}, "pos": pos}
            self.world = _login_rows(self.world, self._fresh, slot, {k: np.asarray(v) for k, v in f.items()})
            self.pos_host[slot] = np.asarray(pos, np.int32)
        else:
            self._reset_ro(slot, char["kills"], char["deaths"])
            self._write_slot(slot, pos, char["spawn"], char["hp"], char["inv"], char["kills"], char["deaths"])
        st = char.get("storage")  # characters.load validates it
        self.storage[slot] = np.array(st, np.int32) if st is not None else new_storage()
        self._clear_pending(slot)
        self.owner[slot] = key
        self.latest_input[slot] = (0, False, False)
        self.tap[slot] = [0, False, False]
        self.path[slot] = deque()
        self.pending[slot] = None
        self.stuck[slot] = 0
        self.need_full[slot] = True
        self.known = self.known.at[slot].set(False)
        self.notice.pop(slot, None)
        self.prev_dead[slot] = False
        self.town[slot] = town
        log.info("login %s slot %d town %d", key, slot, town)
        return slot, None

    def _warm_login(self) -> None:
        """Compile the login writer (result dropped)."""
        if _login_rows is None:
            return
        c = characters.new_character(0, (1, 1))
        f = {**fresh_ro_fields(c["inv"]), **{k: c[k] for k in ("pos", "spawn", "hp", "inv", "kills", "deaths")}}
        _login_rows(self.world, self._fresh, 0, {k: np.asarray(v) for k, v in f.items()})

    def _reset_ro(self, slot: int, kills: int, deaths: int) -> None:
        """Fresh RO rows for `slot` (new Novice, empty inventory, no party) in one jitted call."""
        self.world = _reset_rows(self.world, self._fresh, slot, kills, deaths)

    def _clear_pending(self, slot: int) -> None:
        for k, v in self.pend.items():
            v[slot] = _NONE_INPUTS[k][slot]

    def _ro_host(self) -> dict:
        """Numpy copies of the RO block rows, read from the world (used when no tick pull is at hand)."""
        return jax.device_get(_ro_blocks(self.world))

    def _player_arrays(self) -> dict:
        """Numpy copies of the player fields a save needs, read from the world."""
        p = self.world["players"]
        return {k: np.asarray(p[k]) for k in _PLAYER_PULL}

    def _fields(self, slot: int, pull: dict | None = None) -> dict:
        """Character fields for `slot`, from the tick's pull (or the world when none given).

        With net.char_fields present the RO blocks are included (stats, skills, items, craft, storage)."""
        pl = self._player_arrays() if pull is None else pull["players"]
        ro = None if pull is None else pull.get("ro")
        dead = bool(pl["dead"][slot])
        spawn = np.array(pl["spawn"][slot], np.int32)
        max_hp = PLAYER_MAX_HP
        if dead and character_fields is not None:
            dv = (ro or {}).get("derived")
            max_hp = int(dv["max_hp"][slot]) if dv else int(ro_stats.derived(self.world)["max_hp"][slot])
        base = {"town": self.town[slot], "pos": spawn.copy() if dead else np.array(pl["pos"][slot], np.int32),
                "spawn": spawn, "hp": max_hp if dead else int(pl["hp"][slot]),
                "inv": np.array(pl["inv"][slot], np.int32)[:INV_SLOTS],
                "kills": int(pl["kills"][slot]), "deaths": int(pl["deaths"][slot])}
        if character_fields is None:
            return base
        ro = ro or self._ro_host()
        out = character_fields({"players": pl, **{b: ro[b] for b in ("stats", "skills", "items", "craft")}}, slot,
                               town=self.town[slot], storage=self.storage.get(slot))
        out.update(base)
        return out

    def _save_slot(self, slot: int) -> None:
        name = self.owner[slot]
        self.save_gen[name] = self.save_gen.get(name, 0) + 1  # before the write: any older batch entry is now stale
        try:
            characters.save(self.chars_dir, name, self._fields(slot))
        except Exception:
            log.exception("save failed for %s", name)

    def _write_batch(self, batch: list) -> None:
        """Write (name, generation, fields) entries; an entry is skipped if a newer save for that name started.

        Runs in a thread: only reads save_gen (the event loop is the sole writer), and the check happens inside
        the characters file lock so a logout save cannot slip between the check and the write."""
        for name, gen, fields in batch:
            try:
                characters.save(self.chars_dir, name, fields, guard=lambda n=name, g=gen: self.save_gen.get(n, 0) <= g)
            except Exception:
                log.exception("save failed for %s", name)

    def _collect_batch(self, pull: dict) -> list:
        """Snapshot every human's fields and bump their save generation (event loop only)."""
        batch = []
        for s in self._humans():
            name = self.owner[s]
            self.save_gen[name] = self.save_gen.get(name, 0) + 1
            batch.append((name, self.save_gen[name], self._fields(s, pull)))
        return batch

    def _periodic_save(self, pull: dict) -> None:
        """Collect every human's fields now, write the files off the tick path."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and self._saving:
            log.warning("previous periodic save still running; skipping tick %d", self.tick)
            return  # before collecting: bumping generations here would invalidate the in-flight batch
        batch = self._collect_batch(pull)
        if not batch:
            return
        if loop is None:
            self._write_batch(batch)  # no event loop (direct tick_once call): write inline
            return
        t = loop.create_task(asyncio.to_thread(self._write_batch, batch))
        self._saving.add(t)
        t.add_done_callback(self._saving.discard)
        log.info("saving %d characters at tick %d", len(batch), self.tick)

    def logout(self, slot: int) -> None:
        """Save the human, hand the slot back to the bots, drop its connection state."""
        name = self.owner[slot]
        if name in (None, "bot"):
            return
        self._save_slot(slot)
        town = slot % NT
        tile = self._free_temple_tile(town)
        self._write_slot(slot, tile, tile, PLAYER_MAX_HP, np.zeros(INV_SLOTS, np.int32), 0, 0)
        self._reset_ro(slot, 0, 0)  # the bot that takes the slot over starts as a fresh Novice with no party
        self.owner[slot] = "bot"
        self._walked.discard(slot)
        self.known = self.known.at[slot].set(False)
        self._clear_pending(slot)
        for d in (self.latest_input, self.path, self.pending, self.stuck, self.notice,
                  self.prev_dead, self.town, self.tap, self.need_full, self.storage, self.last_members,
                  self.last_storage):
            d.pop(slot, None)
        self.chat_msgs = [m for m in self.chat_msgs if m["slot"] != slot]
        self.storage_ops = [o for o in self.storage_ops if o[0] != slot]
        c = self.conn.pop(slot, None)
        if c is not None:
            t = asyncio.ensure_future(self._close(c.ws))
            self._closing.add(t)
            t.add_done_callback(self._closing.discard)
            if c.send_task is not None and not c.send_task.done():
                self._sending.add(c.send_task)
                c.send_task.add_done_callback(self._sending.discard)
        log.info("logout %s slot %d", name, slot)

    @staticmethod
    async def _close(ws) -> None:
        try:
            await ws.close()
        except Exception:
            pass

    # ----- tick -----

    def tick_once(self) -> None:
        """One game tick, synchronously: one jit (inputs, step, entity table, deltas), one host pull, snapshots,
        periodic save. The tick loop runs the same three steps pipelined (see _tick_loop)."""
        self._dispatch()
        self._publish(*self._collect())

    def _dispatch(self) -> None:
        """Gather the humans' inputs and call the jitted tick. Returns at once: XLA computes the tick on its own
        threads (no GIL) until _collect pulls the results."""
        self._process_storage()
        pos = self.pos_host
        hmask = np.zeros(MAX_PLAYERS, bool)
        hmove = np.zeros(MAX_PLAYERS, np.int32)
        hattack = np.zeros(MAX_PLAYERS, bool)
        heat = np.zeros(MAX_PLAYERS, bool)
        for slot in self._humans():
            m, a, e = self.latest_input[slot]
            tm, ta, te = self.tap[slot]
            m, a, e = m or tm, a or ta, e or te  # a tap released before this tick still counts for one tick
            self.tap[slot] = [0, False, False]
            if e:
                self.latest_input[slot] = (self.latest_input[slot][0], self.latest_input[slot][1], False)  # one eat per message
            hmask[slot], hattack[slot], heat[slot] = True, a, e
            hmove[slot] = self._path_move(slot, tuple(int(v) for v in pos[slot]), m)
        # a client whose previous send is still in flight is skipped this tick: its known row is reset (viewer -1)
        # so the next snap it receives is a full resync
        viewer = np.full(MAX_VIEWERS, -1, np.int32)
        for slot, c in self.conn.items():
            if c.send_task is None or c.send_task.done():
                viewer[slot] = slot
        hro = {k: v.copy() for k, v in self.pend.items()}
        for k, v in self.pend.items():  # one-shot: the tick consumes them
            v[:] = _NONE_INPUTS[k]
        self.world, self.prev_tab, self.known, pull = _tick_jit(
            self.world, hmask, hmove, hattack, heat, hro, self.prev_tab, self.known, viewer)
        chat, self.chat_msgs = self.chat_msgs, []  # routed with this tick's positions, as in the unpipelined order
        self._inflight = (pull, viewer, chat)

    def _collect(self) -> tuple:
        """Wait for the dispatched tick and make the one host pull of the tick; returns (pull, viewer, chat)."""
        pull, viewer, chat = self._inflight
        self._inflight = None
        pull = jax.device_get(pull)
        self.pos_host = np.array(pull["players"]["pos"])
        self.tick += 1
        self._walked.clear()
        return pull, viewer, chat

    def _publish(self, pull: dict, viewer: np.ndarray, chat: list) -> None:
        """Host work of a collected tick: chat, MVP events, every snapshot, the periodic save."""
        names = {s: self.owner[s] for s in self._humans()}
        self._route_chat(pull, names, [m for m in chat if m["slot"] in names])  # senders still online
        self._mvp_events(pull, names)
        self._send_snapshots(pull, names, viewer)
        if self.tick % SAVE_EVERY_TICKS == 0:
            self._periodic_save(pull)

    def _enqueue(self, slot: int, text: str) -> None:
        """Queue an encoded extra message for a connected client; it goes out ahead of the next snapshot."""
        c = self.conn.get(slot)
        if c is not None and len(c.outbox) < MAX_OUTBOX:
            c.outbox.append(text)

    def _route_chat(self, pull: dict, names: dict, msgs: list) -> None:
        if not msgs:
            return
        routed = self.router.route(self.tick, msgs, pull["players"]["pos"], pull["players"]["active"],
                                   pull["ro"]["party"]["party_id"], names)
        for s, out in routed.items():
            for m in out:
                self._enqueue(s, encode(m))

    def _mvp_events(self, pull: dict, names: dict) -> None:
        """Broadcast one `mvp` message per new row of the MVP event ring (docs/party_wire.md)."""
        head = int(pull["mvp"]["event_head"])
        if head > self.last_event_head:
            ev = pull["mvp"]["event"]
            n = min(head - self.last_event_head, ro_mvp.EVENT_RING)
            for i in range(head - n, head):
                kind, killer, tick = (int(v) for v in ev[i % ro_mvp.EVENT_RING])
                who = None if killer < 0 else names.get(killer, "bot")
                text = encode({"t": "mvp", "kind": kind, "killer": who, "tick": tick})
                for s in self.conn:
                    self._enqueue(s, text)
        self.last_event_head = head

    def _in_town(self, pos) -> bool:
        x, y = int(pos[0]), int(pos[1])
        b = _TOWN_BOXES
        return bool(((b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3])).any())

    def _process_storage(self) -> None:
        """Run the queued kafra storage moves (town check, then net.kafra.transfer) before the tick."""
        ops, self.storage_ops = self.storage_ops, []
        for slot, op, idx in ops:
            if slot not in self.storage or self.owner[slot] in (None, "bot"):
                continue
            if not self._in_town(self.pos_host[slot]):
                self.notice[slot] = "no kafra here"
                continue
            try:
                self._storage_move(slot, op, idx)
            except Exception:
                log.exception("storage move failed for slot %d", slot)

    def _storage_move(self, slot: int, op: str, idx: int) -> None:
        """Move one unit between inventory slot / storage slot `idx`. A refused move changes nothing.

        Storage keeps item ids only, so gear with a refine level or cards is refused (its data would be lost) and a
        stack moves one unit at a time."""
        p, it = self.world["players"], self.world["items"]
        inv, cnt = np.array(p["inv"][slot]), np.array(it["inv_count"][slot])
        ref, cards = np.array(it["inv_refine"][slot]), np.array(it["inv_cards"][slot])
        if op == "in" and 0 <= idx < INV_SLOTS and (ref[idx] > 0 or cards[idx].any()):
            return
        new_inv, new_st, ok = transfer(inv, self.storage[slot], idx, op)
        if not ok:
            return
        if op == "in":
            if cnt[idx] > 1:  # the rest of the stack stays
                new_inv[idx], cnt[idx] = inv[idx], cnt[idx] - 1
            else:
                cnt[idx], ref[idx], cards[idx] = 0, 0, 0
        else:
            dst = int(np.flatnonzero(new_inv != inv)[0])
            cnt[dst], ref[dst], cards[dst] = 0, 0, 0  # the items phase counts it as a pickup (1) and merges stacks
        self.storage[slot] = new_st
        dt = p["inv"].dtype
        self.world = {**self.world,
                      "players": {**p, "inv": p["inv"].at[slot].set(jnp.asarray(new_inv, dt))},
                      "items": {**it, "inv_count": it["inv_count"].at[slot].set(jnp.asarray(cnt, jnp.int32)),
                                "inv_refine": it["inv_refine"].at[slot].set(jnp.asarray(ref, jnp.int32)),
                                "inv_cards": it["inv_cards"].at[slot].set(jnp.asarray(cards, jnp.int32))}}

    def _members(self, pull: dict, slot: int, names: dict) -> list[str]:
        pid = pull["ro"]["party"]["party_id"]
        if pid[slot] < 0:
            return []
        ids = np.flatnonzero((pid == pid[slot]) & pull["players"]["active"])
        return [names.get(int(i), "bot") for i in ids]

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

    def _dirty_slots(self, pull: dict) -> np.ndarray:
        """Bool [MAX_PLAYERS]: slots whose `you` source rows changed since the previous tick (one vectorised compare
        per pulled array). A slot that is not dirty and not on a full snap gets `you` = {pos, hp} only."""
        tab = pull["tab"]
        leaves = jax.tree_util.tree_leaves({"players": pull["players"], "safe": pull["safe"], "ro": pull["ro"],
                                            "anim": pull.get("anim", {}),
                                            "tab": {k: tab[k][:MAX_PLAYERS] for k in ("speed_ms",) if k in tab}})
        prev, self._you_prev = self._you_prev, leaves
        if prev is None or len(prev) != len(leaves):
            return np.ones(MAX_PLAYERS, bool)
        dirty = np.zeros(MAX_PLAYERS, bool)
        for a, b in zip(leaves, prev):
            dirty |= (a != b).reshape(MAX_PLAYERS, -1).any(axis=1)
        return dirty

    def _snapshot_view(self, slot: int, pull: dict, names: dict, viewed: bool, dirty: np.ndarray) -> tuple | None:
        """Per-slot bookkeeping and the (slot, you, notice, full, fmt) entry for build_snaps; None = no snap."""
        dead = bool(pull["players"]["dead"][slot])
        if dead and not self.prev_dead.get(slot, False):
            self.notice[slot] = "you died"
            self.path[slot].clear()
            self.pending[slot] = None
            self.stuck[slot] = 0
        self.prev_dead[slot] = dead
        c = self.conn[slot]
        if not viewed:
            self.need_full[slot] = True  # known was reset for this row; resync on the next snap
            return None  # slow client: skip this tick, never wait
        full = self.need_full.get(slot, True)
        self.need_full[slot] = False
        if full or dirty[slot]:
            you = you_delta(you_full(pull, slot, self.town[slot]), c.last_you, full)
        else:
            p = pull["players"]
            you = {"pos": p["pos"][slot].tolist(), "hp": int(p["hp"][slot])}
        if bool(pull["warped"][slot]):  # ro3-wire: sent on that tick only, never stored in c.last_you
            you["warped"] = True
        members = self._members(pull, slot, names)  # change-detected: sent on full snaps and when they differ
        if full or members != self.last_members.get(slot):
            you["members"] = self.last_members[slot] = members
        st = self.storage.get(slot)
        if st is not None and (full or st.tobytes() != self.last_storage.get(slot)):
            you["storage"] = st.tolist()
            self.last_storage[slot] = st.tobytes()
        return slot, you, self.notice.pop(slot, None), full, c.fmt

    def _send_snapshots(self, pull: dict, names: dict, viewer: np.ndarray) -> None:
        """Every connected client's snap for this tick, encoded in one build_snaps pass, then one send task each."""
        dirty = self._dirty_slots(pull)
        views = []
        for slot in list(self.conn):
            try:
                v = self._snapshot_view(slot, pull, names, bool(viewer[slot] >= 0), dirty)
            except Exception:
                log.exception("snapshot failed for slot %d; logging it out", slot)
                self.logout(slot)
                continue
            if v is not None:
                views.append(v)
        for (slot, *_), msg in zip(views, build_snaps(pull, views, self.tick, names)):
            c = self.conn[slot]
            texts, c.outbox = c.outbox + [msg], []
            # pipelined loop: this tick's viewer row was fixed before the previous send was queued, so that send
            # can still be in flight; chain after it to keep per-client tick order
            prev = c.send_task if c.send_task is not None and not c.send_task.done() else None
            c.send_task = asyncio.create_task(self._send(c, texts, prev))

    async def _send(self, c: Client, texts: list, after: asyncio.Task | None = None) -> None:
        try:
            if after is not None:
                await asyncio.wait([after])
            for text in texts:
                await c.ws.send(text)
        except websockets.ConnectionClosed:
            pass  # the handler sees the close and logs the slot out
        except Exception:
            log.exception("send failed")

    async def _tick_loop(self) -> None:
        """Pipelined when self.pipeline: each period collects the tick dispatched one period earlier (done by
        then), dispatches the next tick, and only then builds and sends the collected tick's snapshots, so the
        snapshot work overlaps the device tick. Cost: snaps leave one device-tick time later (~50 ms on CPU)."""
        loop = asyncio.get_running_loop()
        dt = self.tick_ms / 1000
        deadline = loop.time() + dt
        while True:
            delay = deadline - loop.time()
            await asyncio.sleep(max(delay, 0))
            t0 = time.perf_counter()
            try:
                if not self.pipeline:
                    self.tick_once()
                else:
                    done = self._collect() if self._inflight is not None else None
                    t1 = time.perf_counter()
                    self._dispatch()
                    t2 = time.perf_counter()
                    if done is not None:
                        self._publish(*done)
                    t3 = time.perf_counter()
                    self._phase_max = [max(a, b) for a, b in zip(self._phase_max, (t1 - t0, t2 - t1, t3 - t2))]
            except Exception:
                log.exception("tick failed")
            took = time.perf_counter() - t0
            self._max_tick_s = max(self._max_tick_s, took)
            if took > dt:
                self.overruns += 1
            if self.tick % self.stats_every == 0:  # bench/server_load.py --spawn parses this line
                log.info("tick %d: max tick duration over the last %d ticks: %.1f ms, overruns %d "
                         "(max collect %.1f, dispatch %.1f, publish %.1f ms)", self.tick, self.stats_every,
                         self._max_tick_s * 1000, self.overruns, *(1000 * x for x in self._phase_max))
                self._max_tick_s = 0.0
                self._phase_max = [0.0, 0.0, 0.0]
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

    async def _authenticate(self, first: dict) -> str | None:
        """Check the login's password (or create the account) in the default executor; None when it passes.

        Compat: a login without `password` still works for names that have no account (existing name-only
        characters), unless `require_password` is set. A name with an account always needs its password."""
        store, name = self.accounts, first["name"]
        key = characters.valid_name(name)
        if key is None:
            return "bad name"
        pw, create = first.get("password"), first.get("create", False)
        if pw is None and not create:
            return "bad password" if self.require_password or store.exists(key) else None
        if store.locked_out(key):
            return "locked out"
        loop = asyncio.get_running_loop()
        if create:
            return await loop.run_in_executor(None, store.create, name, pw or "")
        if not await loop.run_in_executor(None, store.verify, name, pw):
            return "locked out" if store.locked_out(key) else "bad password"
        return None

    def _account_login(self, first: dict) -> None:
        """Record the login on the account (v1: account name == character name). Off the loop, never fatal."""
        if first.get("password") is None and not first.get("create"):
            return
        key = characters.valid_name(first["name"])

        def work() -> None:
            try:
                self.accounts.touch_login(key)
                self.accounts.add_character(key, key)
            except Exception:
                log.exception("account bookkeeping failed for %s", key)

        asyncio.get_running_loop().run_in_executor(None, work)

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
            err = await self._authenticate(first)  # may hash off-loop; the slot pick below stays synchronous
            if err is not None:
                await self._refuse(ws, err)
                return
            if getattr(ws, "close_code", None) is not None:  # the client left while the password was hashing
                return
            slot, err = self.login(first["name"])
            if err is not None:
                await self._refuse(ws, err)
                return
            self._account_login(first)
            c = Client(ws, first.get("format", "json"))
            self.conn[slot] = c
            c.send_task = asyncio.create_task(self._send(
                c, [encode(welcome_message(slot, self.tick_ms, self.maps, self.welcome_static))]))
            async for raw in ws:
                msg = decode(raw) if isinstance(raw, str) else None
                if msg is None:
                    await self._refuse(ws, "malformed message")
                    return
                if msg["t"] in ("input", "walk_to"):
                    if msg["t"] == "walk_to" and all(
                            isinstance(msg.get(k), int) and not isinstance(msg.get(k), bool) for k in ("x", "y")):
                        msg = {"t": "walk_to", "x": msg["x"], "y": msg["y"]}  # out-of-map ints: "too far", not malformed
                    else:
                        msg = validate_client(msg)
                    if msg is None:
                        await self._refuse(ws, "malformed message")
                        return
                    self._on_message(slot, msg)
                elif msg["t"] in _RO_MESSAGES:
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
            move = LEGACY_TO_RO[msg["move"]] if msg["move"] < len(LEGACY_TO_RO) else msg["move"]  # 4-way or RO 1..8
            self.latest_input[slot] = (move, msg["attack"], msg["eat"])
            tap = self.tap[slot]
            if move:
                tap[0] = move
            if msg["attack"]:
                tap[1] = True
            if msg["eat"]:
                tap[2] = True
            self.path[slot].clear()
            self.pending[slot] = None
        elif msg["t"] == "walk_to":
            if slot in self._walked:
                return  # at most one A* per slot per tick
            self._walked.add(slot)
            p = self.world["players"]["pos"][slot]
            start = (int(p[0]), int(p[1]))
            path = find_path(self.walls, start, (msg["x"], msg["y"]))
            if path is None:
                self.notice[slot] = "too far"
                return
            self.path[slot] = deque(path)
            self.pending[slot] = None
            self.stuck[slot] = 0
        else:
            self._on_ro_message(slot, msg)

    def _slot_of(self, name) -> int:
        """Slot of the online human called `name`, -1 when null or unknown."""
        key = characters.valid_name(name) if isinstance(name, str) else None
        return self.owner.index(key) if key is not None and key in self.owner else -1

    def _on_ro_message(self, slot: int, msg: dict) -> None:
        """Store the RO request as a one-shot pending input for exactly the next tick (last message wins)."""
        t, pend = msg["t"], self.pend
        if t == "allot":
            pend["allot"][slot] = msg["stat"]
        elif t == "job":
            pend["job"][slot] = msg["job"]
        elif t == "skill":
            pend["skill"][slot], pend["target"][slot] = msg["id"], msg["target"]
        elif t == "learn":
            pend["learn"][slot] = msg["id"]
        elif t in ("use_item", "equip", "refine", "drop", "sell"):
            v = msg["slot"]
            pend[t][slot] = v if -1 <= v < 2 ** 31 else -1  # the engine ignores anything outside 0..INV_SLOTS-1
        elif t == "buy":
            pend["buy"][slot] = msg["item"] if 0 < msg["item"] < 2 ** 31 else 0
        elif t == "gather":
            pend["gather"][slot] = True
        elif t == "craft":
            pend["craft"][slot] = msg["recipe"]
        elif t == "party":
            pend["party"][slot] = PARTY_OP[msg["op"]]
            pend["target"][slot] = self._slot_of(msg["target"])
            if msg["op"] == "invite" and pend["target"][slot] < 0:
                self.notice[slot] = "no such player"
        elif t == "chat":
            if len(self.chat_msgs) < MAX_CHAT_PER_TICK:
                self.chat_msgs.append({"slot": slot, "kind": msg["kind"], "text": msg["text"], "to": msg.get("to")})
        elif t == "storage":
            if len(self.storage_ops) < MAX_CHAT_PER_TICK:
                self.storage_ops.append((slot, msg["op"], msg["slot"]))
        elif t == "save":
            pend["save"][slot] = True
        elif t == "sit":
            pend["sit"][slot] = True
        elif t == "teleport":  # docs/world_wire.md: kafra teleport to a town index
            pend["teleport"][slot] = msg["town"]


def bench_ticks(seed: int, n: int) -> None:
    """Time n ticks (device tick + host pull) at MAX_VIEWERS viewers on the current jax backend."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        srv = GameServer(seed, Path(tmp))
        srv.prepare()
        z = np.zeros(MAX_PLAYERS)
        hro = {k: v.copy() for k, v in srv.pend.items()}
        viewer = np.arange(MAX_VIEWERS, dtype=np.int32)
        world, tab, known, ts = srv.world, srv.prev_tab, srv.known, []
        for i in range(n + 5):
            t0 = time.perf_counter()
            world, tab, known, pull = _tick_jit(world, z.astype(bool), z.astype(np.int32), z.astype(bool),
                                                z.astype(bool), hro, tab, known, viewer)
            jax.device_get(pull)
            if i >= 5:
                ts.append((time.perf_counter() - t0) * 1000)
        print(f"backend={jax.default_backend()} devices={jax.devices()} viewers={MAX_VIEWERS} "
              f"E={tab['kind'].shape[0]} ticks={n}: p50 {np.percentile(ts, 50):.1f} ms, "
              f"p95 {np.percentile(ts, 95):.1f} ms, min {min(ts):.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="JAX tile MMO game server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chars", default="characters")
    ap.add_argument("--tick-ms", type=int, default=TICK_MS)
    ap.add_argument("--require-password", action="store_true", help="refuse logins that carry no password")
    ap.add_argument("--platform", choices=("cpu", "gpu"), default="cpu",
                    help="jax backend; gpu sets JAX_PLATFORMS=cuda and the 80%% memory cap (run in WSL)")
    ap.add_argument("--stats-every", type=int, default=100, help="ticks between tick-duration log lines")
    ap.add_argument("--no-pipeline", action="store_true", help="run each tick start to end (no device overlap)")
    ap.add_argument("--bench-ticks", type=int, default=0,
                    help="no sockets: time N ticks with every slot a viewer, print p50/p95 ms, exit")
    args = ap.parse_args()
    if args.bench_ticks:
        bench_ticks(args.seed, args.bench_ticks)
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    async def run() -> None:
        server = GameServer(args.seed, Path(args.chars), args.tick_ms, require_password=args.require_password,
                            stats_every=args.stats_every, pipeline=not args.no_pipeline)
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
