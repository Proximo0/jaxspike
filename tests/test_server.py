import asyncio, json, numpy as np, pytest
import jax.numpy as jnp
import websockets
from game.config import ITEM_MEAT, PLAYER_MAX_HP, PROTOCOL_VERSION
from game.world_layout import TOWNS, temple_tiles
from net.characters import load, load_roster, new_character, save
from net.protocol import decode_map
from net.server import Client, GameServer

def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 120))  # the first test pays the jit compile; slow on a loaded box

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
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        await s.start("127.0.0.1", 0)
        try:
            ws, w = await connect(s, "Alice")
            assert w["t"] == "welcome" and s.owner[w["slot"]] == "alice" and w["v"] == 2
            assert (decode_map(w["terrain"], np.int8) == np.asarray(s.world["terrain"])).all()
            assert (decode_map(w["pz"], np.uint8).astype(bool) == np.asarray(s.world["pz"])).all()
            msgs = [json.loads(await ws.recv()) for _ in range(3)]
            assert [m["t"] for m in msgs] == ["snap", "snap", "snap"]  # welcome comes first
            snap = msgs[0]
            assert "terrain" not in snap and "terrain" not in msgs[1]  # v1 key is gone
            assert snap["full"] is True and msgs[1]["full"] is False and msgs[2]["full"] is False
            assert snap["you"]["hp"] == PLAYER_MAX_HP and snap["gone"] == [] and snap["v"] == 2
            ids = {e["id"] for e in snap["ents"]}
            assert w["slot"] in ids  # a viewer always sees itself
            me = next(e for e in snap["ents"] if e["id"] == w["slot"])
            assert me["kind"] == 0 and me["name"] == "alice" and me["pos"] == snap["you"]["pos"]
            assert snap["you"]["safe"] is True  # new characters start on a temple
            assert len(msgs[2]["ents"]) < len(snap["ents"])  # later snaps carry only deltas
            await ws.close()
        finally:
            await s.stop()
    run(body())

def test_input_moves_and_two_clients_see_each_other(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        await s.start("127.0.0.1", 0)
        try:
            a, wa = await connect(s, "alice"); b, wb = await connect(s, "bob")
            # bob is placed on alice's town so they share a view: force via the roster order (both new -> towns 0 and 1)
            # so instead move alice's character onto bob's spawn neighbour through the world directly.
            sa, sb = wa["slot"], wb["slot"]
            pb = np.asarray(s.world["players"]["pos"][sb])
            s.world["players"]["pos"] = s.world["players"]["pos"].at[sa].set(jnp.asarray(pb + np.array([2, 0]), jnp.int32))
            ids_seen = {}
            for _ in range(5):
                snap = await next_snap(b)
                ids_seen.update({e["id"]: e for e in snap["ents"]})
            assert ids_seen[sa]["name"] == "alice"  # bob sees alice appear
            before = (await next_snap(a))["you"]["pos"]
            while before != [int(pb[0]) + 2, int(pb[1])]:  # drain snaps queued before the teleport
                before = (await next_snap(a))["you"]["pos"]
            await a.send(json.dumps({"v": PROTOCOL_VERSION, "t": "input", "move": 2, "attack": False, "eat": False}))
            for _ in range(5):
                after = (await next_snap(a))["you"]["pos"]
            assert after[1] > before[1]
            await a.send(json.dumps({"v": PROTOCOL_VERSION, "t": "input", "move": 0, "attack": False, "eat": False}))
            # alice leaves the window: bob gets her id in gone
            far = np.asarray(s.world["players"]["pos"][sb]) + np.array([40, 0])
            s.world["players"]["pos"] = s.world["players"]["pos"].at[sa].set(jnp.asarray(far, jnp.int32))
            gone = []
            for _ in range(6):
                gone += (await next_snap(b))["gone"]
            assert sa in gone
            await a.close(); await b.close()
        finally:
            await s.stop()
    run(body())

def test_walk_to_reaches_tile(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        await s.start("127.0.0.1", 0)
        try:
            ws, w = await connect(s, "walker")
            you = (await next_snap(ws))["you"]["pos"]
            goal = [you[0] + 3, you[1] + 2]  # inside the temple square, always walkable
            await ws.send(json.dumps({"v": PROTOCOL_VERSION, "t": "walk_to", "x": goal[0], "y": goal[1]}))
            pos = None
            for _ in range(15):
                pos = (await next_snap(ws))["you"]["pos"]
                if pos == goal:
                    break
            assert pos == goal
            await ws.send(json.dumps({"v": PROTOCOL_VERSION, "t": "walk_to", "x": -1, "y": 5}))
            notices = [(await next_snap(ws))["notice"] for _ in range(5)]
            assert "too far" in notices
            await ws.close()
        finally:
            await s.stop()
    run(body())

def test_logout_saves_and_slot_returns_to_bot(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        await s.start("127.0.0.1", 0)
        try:
            ws, w = await connect(s, "carol"); slot = w["slot"]
            last = await next_snap(ws)
            await ws.close()
            for _ in range(100):
                if s.owner[slot] == "bot": break
                await asyncio.sleep(0.05)
            assert s.owner[slot] == "bot" and (tmp_path / "carol.npz").exists()
            saved = load(tmp_path, "carol")
            assert saved["pos"].tolist() == last["you"]["pos"] and saved["hp"] == last["you"]["hp"]
            bot_pos = tuple(int(v) for v in np.asarray(s.world["players"]["pos"][slot]))
            assert bot_pos in {tuple(int(v) for v in t) for t in np.asarray(temple_tiles())[slot % len(TOWNS)]}
            ws2, w2 = await connect(s, "carol")
            assert w2["t"] == "welcome"
            await ws2.close()
        finally:
            await s.stop()
    run(body())

def test_refusals(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50, max_clients=1)
        await s.start("127.0.0.1", 0)
        try:
            a, wa = await connect(s, "dave"); assert wa["t"] == "welcome"
            b, wb = await connect(s, "dave"); assert wb["t"] == "error" and wb["msg"] == "already online"
            c, wc = await connect(s, "erin"); assert wc["t"] == "error" and wc["msg"] == "server full"
            d, wd = await connect(s, "x"); assert wd["t"] == "error" and wd["msg"] == "bad name"
            (tmp_path / "frank.npz").write_bytes(b"junk")
            await a.close()
            await asyncio.sleep(0.1)
            e, we = await connect(s, "frank"); assert we["t"] == "error" and we["msg"] == "corrupt character"
            ws = await websockets.connect(f"ws://127.0.0.1:{s.port}")
            await ws.send("garbage")
            m = json.loads(await ws.recv())
            assert m["t"] == "error" and m["msg"] == "expected login"
        finally:
            await s.stop()
    run(body())

def test_two_logins_get_distinct_slots(tmp_path):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
    s.prepare()  # builds the world without starting sockets
    a, _ = s.login("gina"); b, _ = s.login("hank")
    assert a is not None and b is not None and a != b
    assert s.owner[a] == "gina" and s.owner[b] == "hank"

def test_disconnect_during_play_frees_slot(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        await s.start("127.0.0.1", 0)
        try:
            ws, w = await connect(s, "ivan"); slot = w["slot"]
            await ws.send(json.dumps({"v": PROTOCOL_VERSION, "t": "input", "move": 1, "attack": True, "eat": False}))
            await next_snap(ws)
            await ws.close()
            t0 = s.tick
            for _ in range(30):
                await asyncio.sleep(0.05)
                if s.owner[slot] == "bot": break
            assert s.owner[slot] == "bot" and s.tick > t0  # the loop kept ticking
        finally:
            await s.stop()
    run(body())

def test_slot_pick_prefers_own_stripe(tmp_path):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
    s.prepare()
    save(tmp_path, "zed", new_character(0, (1, 1)))
    slot, err = s.login("zed")
    assert err is None and slot % len(TOWNS) == 0

def test_fallback_slot_keeps_home_town(tmp_path):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
    s.prepare()
    nt = len(TOWNS)
    save(tmp_path, "yan", new_character(0, tuple(int(v) for v in np.asarray(temple_tiles())[0][0])))
    for i in range(0, len(s.owner), nt):  # fill town 0's whole stripe with humans
        if s.owner[i] in (None, "bot"):
            s.owner[i] = f"filler{i}"
    slot, err = s.login("yan")
    assert err is None and slot % nt != 0
    s.logout(slot)
    assert load(tmp_path, "yan")["town"] == 0


class FakeWS:
    def __init__(self):
        self.sent = []
    async def send(self, text):
        self.sent.append(json.loads(text))
    async def close(self):
        pass

def _attach(s, slot):
    ws = FakeWS()
    s.conn[slot] = Client(ws)
    return ws

def test_eat_is_applied_once_per_message(tmp_path):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
    s.prepare()
    slot, _ = s.login("eater")
    inv = s.world["players"]["inv"]
    s.world["players"]["inv"] = inv.at[slot, 0].set(ITEM_MEAT).at[slot, 1].set(ITEM_MEAT)
    s._on_message(slot, {"t": "input", "move": 0, "attack": False, "eat": True})
    for _ in range(5):
        s.tick_once()
    left = int((np.asarray(s.world["players"]["inv"][slot]) == ITEM_MEAT).sum())
    assert left == 1

def test_eat_tap_shorter_than_a_tick_still_eats(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        await s.start("127.0.0.1", 0)
        try:
            ws, w = await connect(s, "nibbler"); slot = w["slot"]
            await next_snap(ws)
            p = s.world["players"]
            s.world["players"] = {**p, "inv": p["inv"].at[slot, 0].set(ITEM_MEAT).at[slot, 1].set(ITEM_MEAT),
                                  "hp": p["hp"].at[slot].set(10)}
            for eat in (True, False):  # press and release with no await between them: no tick can run in between
                await ws.send(json.dumps({"v": PROTOCOL_VERSION, "t": "input", "move": 0, "attack": False, "eat": eat}))
            snaps = [await next_snap(ws) for _ in range(5)]
            left = int((np.asarray(s.world["players"]["inv"][slot]) == ITEM_MEAT).sum())
            assert left == 1 and snaps[-1]["you"]["hp"] > 10
            await ws.close()
        finally:
            await s.stop()
    run(body())

def test_stale_periodic_batch_does_not_overwrite_logout_save(tmp_path):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
    s.prepare()
    slot, _ = s.login("gen")
    batch = s._collect_batch({'players': s._player_arrays()})  # collected, not yet written
    full = int(s.world["players"]["hp"][slot])
    p = s.world["players"]
    s.world["players"] = {**p, "hp": p["hp"].at[slot].set(7)}
    s.logout(slot)
    assert load(tmp_path, "gen")["hp"] == 7
    s._write_batch(batch)  # the stale batch carries the older hp and must be skipped
    assert full != 7 and load(tmp_path, "gen")["hp"] == 7
    batch2 = [(n, g + 1, f) for n, g, f in batch]  # sanity: a newer-generation entry does write
    s._write_batch(batch2)
    assert load(tmp_path, "gen")["hp"] == full

def test_skipped_periodic_save_leaves_generations_alone(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        s.prepare()
        s.login("busy")
        before = dict(s.save_gen)
        fut = asyncio.get_running_loop().create_future()  # stands in for an in-flight batch
        s._saving.add(fut)
        s._periodic_save({'players': s._player_arrays()})
        assert s.save_gen == before
        s._saving.discard(fut)
        s._periodic_save({'players': s._player_arrays()})
        assert s.save_gen["busy"] == before.get("busy", 0) + 1
        await asyncio.gather(*s._saving)
    run(body())

def test_sub_tick_tap_moves_exactly_one_tile(tmp_path):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
    s.prepare()
    slot, _ = s.login("tapper")
    start = np.asarray(s.world["players"]["pos"][slot]).copy()
    s._on_message(slot, {"t": "input", "move": 2, "attack": False, "eat": False})
    s._on_message(slot, {"t": "input", "move": 0, "attack": False, "eat": False})
    s.tick_once()
    one = np.asarray(s.world["players"]["pos"][slot])
    assert one.tolist() == [start[0], start[1] + 1]
    for _ in range(3):
        s.tick_once()
    assert np.asarray(s.world["players"]["pos"][slot]).tolist() == one.tolist()

def test_path_cleared_on_death(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        s.prepare()
        slot, _ = s.login("mortal")
        ws = _attach(s, slot)
        start = np.asarray(s.world["players"]["pos"][slot]).copy()
        s._on_message(slot, {"t": "walk_to", "x": int(start[0]) + 3, "y": int(start[1]) + 2})
        s.tick_once(); s.tick_once()
        assert len(s.path[slot]) > 0
        p = s.world["players"]
        spawn = np.asarray(p["spawn"][slot])
        # keep the body from respawning by parking a monster on the spawn tile
        m = s.world["monsters"]
        s.world["monsters"] = {**m, "active": m["active"].at[0].set(True), "pos": m["pos"].at[0].set(jnp_arr(spawn)),
                               "hp": m["hp"].at[0].set(10 ** 6)}
        s.world["players"] = {**p, "dead": p["dead"].at[slot].set(True)}
        s.tick_once()
        await asyncio.sleep(0)
        assert not s.path[slot] and s.stuck[slot] == 0 and s.pending[slot] is None
        m = s.world["monsters"]
        s.world["monsters"] = {**m, "active": m["active"].at[0].set(False)}
        for _ in range(50):
            s.tick_once(); await asyncio.sleep(0)
            if not bool(s.world["players"]["dead"][slot]):
                break
        assert not bool(s.world["players"]["dead"][slot])
        pos = np.asarray(s.world["players"]["pos"][slot]).tolist()
        assert pos == spawn.tolist()
        for _ in range(5):
            s.tick_once(); await asyncio.sleep(0)
            assert np.asarray(s.world["players"]["pos"][slot]).tolist() == pos
        assert "you died" in [m_["notice"] for m_ in ws.sent if m_["t"] == "snap"]
    run(body())

def jnp_arr(v):
    import jax.numpy as jnp
    return jnp.asarray(v, jnp.int32)

def _snaps(ws):
    return [m for m in ws.sent if m["t"] == "snap"]

def test_gone_after_monster_dies_and_new_monster_appears(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        s.prepare()
        slot, _ = s.login("hunter")
        ws = _attach(s, slot)
        pos = np.asarray(s.world["players"]["pos"][slot])
        m = s.world["monsters"]
        s.world["monsters"] = {**m, "active": m["active"].at[0].set(True), "hp": m["hp"].at[0].set(40),
                               "pos": m["pos"].at[0].set(jnp_arr(pos + np.array([3, 0])))}
        s.tick_once(); await asyncio.sleep(0)
        mid = 512 + 0
        first = _snaps(ws)[0]
        assert first["full"] is True and any(e["id"] == mid for e in first["ents"])
        m = s.world["monsters"]
        s.world["monsters"] = {**m, "active": m["active"].at[0].set(False)}  # the monster dies / despawns
        for _ in range(3):
            s.tick_once(); await asyncio.sleep(0.01)
        snaps = _snaps(ws)
        assert [x["full"] for x in snaps[1:]] == [False] * (len(snaps) - 1)
        assert any(mid in x["gone"] for x in snaps[1:])
        assert all("terrain" not in x for x in snaps)
    run(body())

def test_skipped_slow_client_gets_a_full_resync(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        s.prepare()
        slot, _ = s.login("slowpoke")
        ws = _attach(s, slot)
        s.tick_once(); await asyncio.sleep(0.01)
        s.tick_once(); await asyncio.sleep(0.01)
        assert [x["full"] for x in _snaps(ws)] == [True, False]
        gate = asyncio.get_running_loop().create_future()  # a send that never completes: the client is slow
        s.conn[slot].send_task = asyncio.ensure_future(gate)
        n = len(_snaps(ws))
        s.tick_once(); s.tick_once(); await asyncio.sleep(0.01)
        assert len(_snaps(ws)) == n  # the tick never waited for it and sent nothing
        gate.set_result(None); await asyncio.sleep(0.01)
        s.tick_once(); await asyncio.sleep(0.01)
        last = _snaps(ws)[-1]
        assert last["full"] is True and any(e["id"] == slot for e in last["ents"])
    run(body())

