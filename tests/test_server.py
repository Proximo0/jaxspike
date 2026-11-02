import asyncio, json, numpy as np, pytest
import websockets
from game.config import PLAYER_MAX_HP, PROTOCOL_VERSION
from game.world_layout import TOWNS, temple_tiles
from net.characters import load, load_roster, new_character, save
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
            last = await next_snap(ws)
            await ws.close()
            for _ in range(20):
                if s.owner[slot] == "bot": break
                await asyncio.sleep(0.02)
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
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20, max_clients=1)
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
            await ws.close()
            t0 = s.tick
            for _ in range(30):
                await asyncio.sleep(0.02)
                if s.owner[slot] == "bot": break
            assert s.owner[slot] == "bot" and s.tick > t0  # the loop kept ticking
        finally:
            await s.stop()
    run(body())

def test_slot_pick_prefers_own_stripe(tmp_path):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20)
    s.prepare()
    save(tmp_path, "zed", new_character(0, (1, 1)))
    slot, err = s.login("zed")
    assert err is None and slot % len(TOWNS) == 0

def test_fallback_slot_keeps_home_town(tmp_path):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=20)
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
