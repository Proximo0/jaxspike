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
            assert snap["you"]["hp"] == snap["you"]["max_hp"] and snap["gone"] == [] and snap["v"] == 2
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
            for _ in range(20):  # tick-count wait: the pipelined tick loop adds one tick of input latency
                after = (await next_snap(a))["you"]["pos"]
                if after[1] > before[1]:
                    break
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
            notices = []
            for _ in range(40):
                notices.append((await next_snap(ws))["notice"])
                if "too far" in notices:
                    break
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
            bot_pos = tuple(int(v) for v in np.asarray(s.world["players"]["spawn"][slot]))  # pos may have moved on
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
    one = [start[0], start[1] + 1]
    for _ in range(6):  # the temple is crowded with bots: a lower slot can win the tile for a tick or two
        s.tick_once()
        if np.asarray(s.world["players"]["pos"][slot]).tolist() == one:
            break
    assert np.asarray(s.world["players"]["pos"][slot]).tolist() == one
    for _ in range(3):
        s.tick_once()
    assert np.asarray(s.world["players"]["pos"][slot]).tolist() == one  # the tap was consumed by one step

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

def _you(snaps):
    """`you` merged over the snaps in order, as the clients do (keys come on full snaps and on change only)."""
    out = {}
    for x in snaps:
        out.update(x["you"])
    return out

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



# ----- RO systems over the wire (docs/*_wire.md) -----

def _ro_server(tmp_path, **kw):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50, **kw)
    s.prepare()
    return s

def _human(s, name):
    slot, err = s.login(name)
    assert err is None
    return slot, _attach(s, slot)

async def tk(s):
    s.tick_once()
    await asyncio.sleep(0.005)  # let the send tasks of this tick run before the next one

def _flush():
    return asyncio.sleep(0.02)

NONE_KEYS = ("allot", "job", "skill", "learn", "target", "use_item", "equip", "refine", "drop", "gather", "craft",
             "party", "save")

def _pend_is_none(s, slot=None):
    from net.server import _NONE_INPUTS
    return all((s.pend[k] == _NONE_INPUTS[k]).all() if slot is None else s.pend[k][slot] == _NONE_INPUTS[k][slot]
               for k in NONE_KEYS)

def test_ro_messages_fill_pending_and_clear_after_one_tick(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, _ = _human(s, "pendera")
        b, _ = _human(s, "penderb")
        assert _pend_is_none(s)
        send = lambda t, **kw: s._on_ro_message(a, {"t": t, **kw})
        send("allot", stat=3); send("job", job=2); send("skill", id=4, target=515); send("learn", id=7)
        send("use_item", slot=1); send("equip", slot=2); send("refine", slot=3); send("drop", slot=4)
        send("gather"); send("craft", recipe=5); send("save")
        p = s.pend
        assert (p["allot"][a], p["job"][a], p["skill"][a], p["target"][a], p["learn"][a]) == (3, 2, 4, 515, 7)
        assert (p["use_item"][a], p["equip"][a], p["refine"][a], p["drop"][a]) == (1, 2, 3, 4)
        assert bool(p["gather"][a]) and p["craft"][a] == 5 and bool(p["save"][a])
        assert _pend_is_none(s, b)  # only the sender's slot
        send("skill", id=9, target=-1)  # a later message of the same kind overwrites (last wins)
        assert (p["skill"][a], p["target"][a]) == (9, -1)
        await tk(s)
        assert _pend_is_none(s)  # consumed by exactly one tick
        await tk(s)
        assert _pend_is_none(s)
    run(body())

def test_sit_message_sets_pending_and_toggles_the_sit_action(tmp_path):
    from game import ro_anim
    async def body():
        s = _ro_server(tmp_path)
        a, _ = _human(s, "sitter")
        s._on_ro_message(a, {"t": "sit"})
        assert bool(s.pend["sit"][a])
        await tk(s)
        assert not s.pend["sit"].any()  # one-shot
        act = lambda: int(s.world["anim"]["action"][a])
        assert act() == ro_anim.SIT
        s._on_ro_message(a, {"t": "sit"})
        await tk(s)
        assert act() == ro_anim.IDLE
    run(body())

def test_snapshot_you_carries_warped_only_on_that_tick(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, _ = _human(s, "warper")
        seen = []
        orig = s._snapshot_view
        s._snapshot_view = lambda slot, pull, *r: (seen.append(pull), orig(slot, pull, *r))[1]
        await tk(s); await tk(s)
        pull = seen[-1]
        names = {a: "warper"}
        dirty = np.zeros(len(pull["warped"]), bool)
        pull["warped"] = np.asarray(pull["warped"]).copy()
        pull["warped"][a] = True
        you = s._snapshot_view(a, pull, names, True, dirty)[1]
        assert you["warped"] is True and "warped" not in s.conn[a].last_you
        pull["warped"][a] = False
        assert "warped" not in s._snapshot_view(a, pull, names, True, dirty)[1]
    run(body())

def test_party_message_maps_target_slot_and_notice(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, _ = _human(s, "partya")
        b, _ = _human(s, "partyb")
        s._on_ro_message(a, {"t": "party", "op": "invite", "target": "PartyB"})  # names are case-insensitive
        assert s.pend["party"][a] == 3 and s.pend["target"][a] == b and a not in s.notice
        s._on_ro_message(a, {"t": "party", "op": "invite", "target": "nobodyhere"})
        assert s.pend["party"][a] == 3 and s.pend["target"][a] == -1 and s.notice[a] == "no such player"
        s._on_ro_message(a, {"t": "party", "op": "create", "target": None})
        assert s.pend["party"][a] == 1 and s.pend["target"][a] == -1
    run(body())

def test_allot_applies_once_and_only_to_the_sender(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, _ = _human(s, "allota")
        b, _ = _human(s, "allotb")
        st = lambda k, i: int(s.world["stats"][k][i])
        pts = st("stat_points", a)
        s._on_ro_message(a, {"t": "allot", "stat": 0})
        await tk(s); await tk(s); await tk(s)
        assert st("str", a) == 2 and st("stat_points", a) == pts - 2  # one message = one point, STAT_COST[1] == 2
        assert st("str", b) == 1 and st("stat_points", b) == pts
    run(body())

def test_tick_overrides_only_human_slots(tmp_path):
    """Bots keep their policy values: the human override arrays only replace the masked slots."""
    import jax
    from net.server import _tick_jit, _NONE_INPUTS
    s = _ro_server(tmp_path)
    z = np.zeros(512)
    viewer = np.full(512, -1, np.int32)
    hmask = z.astype(bool); hmask[3] = True
    base = {k: v.copy() for k, v in _NONE_INPUTS.items() if k in NONE_KEYS}
    junk = {k: v.copy() for k, v in base.items()}
    for k in ("allot", "job", "skill", "learn", "use_item", "equip", "refine", "drop", "craft"):
        junk[k][:] = 1                      # every slot wants something; only slot 3 is a human
    junk["gather"][:] = True; junk["party"][:] = 1; junk["save"][:] = True
    def run_tick(h):
        w, *_ = _tick_jit(s.world, hmask, z.astype(np.int32), z.astype(bool), z.astype(bool), h, s.prev_tab, s.known, viewer)
        return jax.device_get(w["stats"])
    a, b = run_tick(base), run_tick(junk)
    others = np.arange(512) != 3
    for k in ("str", "agi", "stat_points", "skill_points", "job", "base_level"):
        assert (a[k][others] == b[k][others]).all(), k   # bot slots are identical whatever the human arrays hold
    assert not all((a[k] == b[k]).all() for k in ("str", "agi", "vit", "int", "dex", "luk", "stat_points"))  # slot 3 changed

def test_snap_you_has_the_ro_fields(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, ws = _human(s, "youfields")
        await tk(s); await tk(s); await _flush()
        first, second = _snaps(ws)[:2]
        you = first["you"]
        assert you["level"] == 1 and you["job"] == 0 and you["job_level"] == 1 and len(you["stats"]) == 6
        assert you["stat_points"] > 0 and you["skill_points"] == 0 and you["max_sp"] > 0 and you["sp"] > 0
        assert you["max_hp"] > 0 and you["base_exp_next"] > 0 and you["job_exp_next"] > 0
        assert you["buffs"] == [] and you["cooldowns"] == [] and you["learned"] == []
        assert you["casting"] == 0 and you["zeny"] == 0 and len(you["equip"]) == 8
        assert len(you["inv"]) == 16 and set(you["inv"][0]) == {"item", "count", "refine", "cards"} and len(you["inv"][0]["cards"]) == 4
        assert len(you["craft"]["level"]) == 6 and len(you["craft"]["exp"]) == 6
        assert you["party"] == -1 and you["leader"] is False
        assert you["members"] == [] and len(you["storage"]) == 100  # full snap carries the change-detected fields
        assert "members" not in second["you"] and "storage" not in second["you"]  # unchanged: omitted
        assert "pos" in second["you"] and "hp" in second["you"]  # always sent
        assert not {"inv", "equip", "craft", "learned", "max_sp"} & set(second["you"])  # unchanged: omitted
        assert _you([first, second])["max_hp"] == you["max_hp"]
    run(body())

def test_learn_then_skill_in_one_tick(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, ws = _human(s, "caster")
        st = s.world["stats"]
        s.world["stats"] = {**st, "skill_points": st["skill_points"].at[a].set(2)}
        s._on_ro_message(a, {"t": "learn", "id": 2})                       # First Aid (novice, self target)
        s._on_ro_message(a, {"t": "skill", "id": 2, "target": -1})
        await tk(s); await tk(s); await _flush()
        you = _snaps(ws)[0]["you"]
        assert you["learned"] == [[2, 1]] and you["skill_points"] == 1
        assert you["after_delay"] > 0  # Bash: after-cast delay, no cooldown (docs/ro_fidelity.md)
        assert _pend_is_none(s)
    run(body())

def test_chat_say_party_whisper_delivery(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, wa = _human(s, "chata")
        b, wb = _human(s, "chatb")
        c, wc = _human(s, "chatc")
        pos = s.world["players"]["pos"]
        near = np.asarray(pos[a]) + np.array([2, 1])
        far = np.asarray(pos[a]) + np.array([60, 0])
        s.world["players"] = {**s.world["players"], "pos": pos.at[b].set(jnp_arr(near)).at[c].set(jnp_arr(far))}
        chats = lambda ws: [m for m in ws.sent if m["t"] == "chat"]
        await tk(s)
        s._on_ro_message(a, {"t": "chat", "kind": "say", "text": "  hello world  ", "to": None})
        await tk(s); await _flush()
        assert [m["text"] for m in chats(wa)] == ["hello world"] and [m["text"] for m in chats(wb)] == ["hello world"]
        assert chats(wc) == []  # out of view range
        m = chats(wb)[0]
        assert m["kind"] == "say" and m["from"] == "chata" and m["v"] == 2
        i = wb.sent.index(m)
        assert wb.sent[i + 1]["t"] == "snap" and wb.sent[i + 1]["tick"] == m["tick"]  # chat goes out ahead of its tick's snap
        # whisper reaches the named player only, wherever they are; unknown names answer the sender
        s._on_ro_message(a, {"t": "chat", "kind": "whisper", "text": "psst", "to": "ChatC"})
        s._on_ro_message(a, {"t": "chat", "kind": "whisper", "text": "who?", "to": "ghost"})
        await tk(s); await _flush()
        assert [m["text"] for m in chats(wc)] == ["psst"] and chats(wc)[0]["kind"] == "whisper"
        assert [m["text"] for m in chats(wb)] == ["hello world"]
        assert chats(wa)[-1] == {"t": "chat", "kind": "system", "text": "no such player", "v": 2}
        # party chat needs a party
        s._on_ro_message(a, {"t": "chat", "kind": "party", "text": "team?", "to": None})
        await tk(s); await _flush()
        assert all(m["text"] != "team?" for ws in (wa, wb, wc) for m in chats(ws))
        s._on_ro_message(a, {"t": "party", "op": "create", "target": None}); await tk(s)
        s._on_ro_message(a, {"t": "party", "op": "invite", "target": "chatc"}); await tk(s)
        s._on_ro_message(c, {"t": "party", "op": "accept", "target": None}); await tk(s)
        s._on_ro_message(c, {"t": "chat", "kind": "party", "text": "team!", "to": None})
        await tk(s); await _flush()
        assert [m["text"] for m in chats(wa) if m["kind"] == "party"] == ["team!"]
        assert [m["text"] for m in chats(wc) if m["kind"] == "party"] == ["team!"]
        assert not [m for m in chats(wb) if m["kind"] == "party"]  # b is not in the party
    run(body())

def test_party_fields_and_members_change_detection(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, wa = _human(s, "leadera")
        b, wb = _human(s, "joinerb")
        await tk(s)
        s._on_ro_message(a, {"t": "party", "op": "create", "target": None}); await tk(s)
        s._on_ro_message(a, {"t": "party", "op": "invite", "target": "joinerb"}); await tk(s)
        s._on_ro_message(b, {"t": "party", "op": "accept", "target": None}); await tk(s)
        await tk(s); await _flush()
        sa, sb = _snaps(wa), _snaps(wb)
        assert sa[0]["you"]["party"] == -1 and sa[0]["you"]["members"] == []
        created = next(x for x in sa if x["you"].get("party", -1) >= 0)
        assert created["you"]["leader"] is True and created["you"]["members"] == ["leadera"]
        last_a, last_b = _you(sa), _you(sb)
        assert last_a["party"] == last_b["party"] >= 0 and last_a["leader"] is True and last_b["leader"] is False
        joined_b = next(x for x in sb if x["you"].get("party", -1) >= 0)
        assert joined_b["you"]["members"] == ["leadera", "joinerb"]
        assert "members" not in sb[-1]["you"]  # only on change
        assert sum("members" in x["you"] for x in sa) == 3  # full snap, party created, b joined
        # leaving empties the list on the next snap
        s._on_ro_message(b, {"t": "party", "op": "leave", "target": None}); await tk(s); await tk(s); await _flush()
        gone = next(x for x in _snaps(wb)[::-1] if "members" in x["you"])
        assert gone["you"]["members"] == [] and _you(_snaps(wb))["party"] == -1
    run(body())

def test_mvp_broadcast_when_event_head_advances(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, wa = _human(s, "mvpa")
        b, wb = _human(s, "mvpb")
        await tk(s)
        mvps = lambda ws: [m for m in ws.sent if m["t"] == "mvp"]
        await tk(s); await _flush()
        assert mvps(wa) == []  # nothing before the ring advances
        mv = s.world["mvp"]
        row = jnp_arr([18, a, 7])
        s.world["mvp"] = {**mv, "event": mv["event"].at[0].set(row), "event_head": mv["event_head"] + 1}
        await tk(s); await tk(s); await _flush()
        want = {"t": "mvp", "kind": 18, "killer": "mvpa", "tick": 7, "v": 2}
        assert mvps(wa) == [want] and mvps(wb) == [want]  # every client, once
        mv = s.world["mvp"]
        s.world["mvp"] = {**mv, "event": mv["event"].at[1].set(jnp_arr([19, -1, 9])), "event_head": mv["event_head"] + 1}
        await tk(s); await _flush()
        assert mvps(wb)[-1]["killer"] is None and mvps(wb)[-1]["kind"] == 19 and len(mvps(wb)) == 2
        mv = s.world["mvp"]
        s.world["mvp"] = {**mv, "event_head": mv["event_head"] + 20}  # more than the ring holds: only the last 8
        await tk(s); await _flush()
        assert len(mvps(wb)) == 2 + 8
    run(body())

def test_slow_client_keeps_chat_in_its_outbox(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, wa = _human(s, "slowchat")
        await tk(s); await _flush()
        gate = asyncio.get_running_loop().create_future()
        s.conn[a].send_task = asyncio.ensure_future(gate)
        s._on_ro_message(a, {"t": "chat", "kind": "say", "text": "queued", "to": None})
        await tk(s); await tk(s)
        assert len(s.conn[a].outbox) == 1  # the tick never waited; the line stays queued
        gate.set_result(None); await _flush()
        await tk(s); await _flush()
        texts = [m["t"] for m in wa.sent]
        assert s.conn[a].outbox == [] and texts.index("chat") < len(texts) - 1 and texts[-1] == "snap"
    run(body())

def test_storage_in_out_and_no_kafra_outside_town(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, ws = _human(s, "boxer")
        stack_id = 4  # red potion: stackable (id 1 is the gold coin, which the items phase turns into zeny)
        p, it = s.world["players"], s.world["items"]
        s.world["players"] = {**p, "inv": p["inv"].at[a, 0].set(stack_id).at[a, 1].set(2)}
        s.world["items"] = {**it, "inv_count": it["inv_count"].at[a, 0].set(5).at[a, 1].set(1)}
        await tk(s); await tk(s); await _flush()
        assert len(_snaps(ws)[0]["you"]["storage"]) == 100
        s._on_ro_message(a, {"t": "storage", "op": "in", "slot": 0})   # a stack moves one unit
        s._on_ro_message(a, {"t": "storage", "op": "in", "slot": 1})
        await tk(s); await tk(s); await _flush()
        sn = _snaps(ws)
        st = next(x["you"]["storage"] for x in sn if "storage" in x["you"] and x["you"]["storage"][0] != 0)
        assert st[:2] == [stack_id, 2]
        assert _you(sn)["inv"][0]["count"] == 4 and _you(sn)["inv"][1]["item"] == 0
        assert sum("storage" in x["you"] for x in sn) == 2  # the full snap, then once on change
        s._on_ro_message(a, {"t": "storage", "op": "out", "slot": 1})
        s._on_ro_message(a, {"t": "storage", "op": "out", "slot": 50})  # empty storage slot: refused, changes nothing
        await tk(s); await tk(s); await _flush()
        last = _you(_snaps(ws))
        assert last["inv"][1]["item"] == 2 and last["inv"][1]["count"] == 1
        # a refined item is refused (the box keeps ids only)
        it = s.world["items"]
        s.world["items"] = {**it, "inv_refine": it["inv_refine"].at[a, 1].set(3)}
        s._on_ro_message(a, {"t": "storage", "op": "in", "slot": 1})
        await tk(s); await tk(s); await _flush()
        assert _you(_snaps(ws))["inv"][1]["item"] == 2
        # outside a town: notice, nothing moves
        far = jnp_arr([5, 5])
        p = s.world["players"]
        s.world["players"] = {**p, "pos": p["pos"].at[a].set(far)}
        s.pos_host[a] = [5, 5]
        s._on_ro_message(a, {"t": "storage", "op": "in", "slot": 0})
        await tk(s); await _flush()
        assert _snaps(ws)[-1]["notice"] == "no kafra here" and _you(_snaps(ws))["inv"][0]["count"] == 4
    run(body())

def test_save_message_sets_the_respawn_point(tmp_path):
    from game.world_layout import town_centers
    async def body():
        s = _ro_server(tmp_path)
        a, _ = _human(s, "saver")
        before = np.asarray(s.world["players"]["spawn"][a]).tolist()
        centre = town_centers()[s.town[a]].tolist()
        assert before != centre
        s._on_ro_message(a, {"t": "save"})
        await tk(s)
        assert np.asarray(s.world["players"]["spawn"][a]).tolist() == centre
    run(body())

def test_logout_gives_the_slot_back_as_a_fresh_bot_row(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, _ = _human(s, "leaver")
        st, it = s.world["stats"], s.world["items"]
        s.world["stats"] = {**st, "base_level": st["base_level"].at[a].set(9), "str": st["str"].at[a].set(30)}
        s.world["items"] = {**it, "zeny": it["zeny"].at[a].set(777)}
        s.logout(a)
        assert int(s.world["stats"]["base_level"][a]) == 1 and int(s.world["stats"]["str"][a]) == 1
        assert int(s.world["items"]["zeny"][a]) == 0 and int(s.world["party"]["party_id"][a]) == -1
        assert a not in s.storage and _pend_is_none(s)
    run(body())

def test_ro_state_round_trips_through_the_character_file(tmp_path):
    async def body():
        s = _ro_server(tmp_path)
        a, _ = _human(s, "persistx")
        st, it = s.world["stats"], s.world["items"]
        s.world["stats"] = {**st, "str": st["str"].at[a].set(12), "stat_points": st["stat_points"].at[a].set(5)}
        s.world["items"] = {**it, "zeny": it["zeny"].at[a].set(4321)}
        s.storage[a][3] = 2
        s.logout(a)
        b, err = s.login("persistx")
        assert err is None
        assert int(s.world["stats"]["str"][b]) == 12 and int(s.world["stats"]["stat_points"][b]) == 5
        assert int(s.world["items"]["zeny"][b]) == 4321
        assert int(s.world["stats"]["kills_seen"][b]) == int(s.world["players"]["kills"][b])
    run(body())


# ----- over real sockets: auth, malformed RO messages, welcome tables, chat -----

async def login_pw(server, name, password=None, create=None):
    ws = await websockets.connect(f"ws://127.0.0.1:{server.port}")
    msg = {"v": PROTOCOL_VERSION, "t": "login", "name": name}
    if password is not None:
        msg["password"] = password
    if create is not None:
        msg["create"] = create
    await ws.send(json.dumps(msg))
    return ws, json.loads(await ws.recv())

async def wait_ticks(s, n):
    t0 = s.tick
    for _ in range(400):
        if s.tick >= t0 + n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("ticks stopped")

async def wait_until(pred, ticks=200):
    for _ in range(ticks):
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met")

def test_password_create_login_and_refusals(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        await s.start("127.0.0.1", 0)
        try:
            ws, w = await login_pw(s, "Pwuser", "hunter2", True)
            assert w["t"] == "welcome" and s.owner[w["slot"]] == "pwuser"
            slot = w["slot"]
            assert s.accounts.exists("pwuser") and s.accounts.verify("pwuser", "hunter2")
            await ws.close(); await wait_until(lambda: s.owner[slot] == "bot")
            bad, m = await login_pw(s, "pwuser", "wrong"); assert m == {**m, "t": "error", "msg": "bad password"}
            none, m = await login_pw(s, "pwuser"); assert m["t"] == "error" and m["msg"] == "bad password"  # account: password required
            ghost, m = await login_pw(s, "nosuchacct", "pw"); assert m["t"] == "error" and m["msg"] == "bad password"
            taken, m = await login_pw(s, "pwuser", "other", True); assert m["t"] == "error" and m["msg"] == "name taken"
            empty, m = await login_pw(s, "newbie1", "", True); assert m["t"] == "error" and m["msg"] == "bad password"
            badname, m = await login_pw(s, "x", "pw", True); assert m["t"] == "error" and m["msg"] == "bad name"
            ok, m = await login_pw(s, "pwuser", "hunter2"); assert m["t"] == "welcome"
            await ok.close(); await wait_until(lambda: "pwuser" not in s.owner)
            assert s.accounts.characters("pwuser") == ["pwuser"]
            # compat: a name with no account still logs in without a password
            legacy, m = await login_pw(s, "legacyguy"); assert m["t"] == "welcome"
            await legacy.close()
        finally:
            await s.stop()
    run(body())

def test_password_lockout_and_require_password(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50, require_password=True)
        await s.start("127.0.0.1", 0)
        try:
            ws, m = await login_pw(s, "strict"); assert m["t"] == "error" and m["msg"] == "bad password"  # compat off
            ws, m = await login_pw(s, "strict", "pw1", True); assert m["t"] == "welcome"
            await ws.close(); await wait_until(lambda: "strict" not in s.owner)
            for _ in range(5):
                ws, m = await login_pw(s, "strict", "nope")
                assert m["t"] == "error" and m["msg"] in ("bad password", "locked out")
            ws, m = await login_pw(s, "strict", "pw1"); assert m["t"] == "error" and m["msg"] == "locked out"
        finally:
            await s.stop()
    run(body())

def test_malformed_ro_messages_close_the_connection(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        await s.start("127.0.0.1", 0)
        try:
            for i, bad in enumerate(({"t": "allot", "stat": 9}, {"t": "skill", "id": 99, "target": -1},
                                     {"t": "chat", "kind": "shout", "text": "x"}, {"t": "party", "op": "boom", "target": None},
                                     {"t": "storage", "op": "in", "slot": "1"}, {"t": "craft", "recipe": 0})):
                ws, w = await connect(s, f"malf{i}")
                assert w["t"] == "welcome"
                await ws.send(json.dumps({"v": PROTOCOL_VERSION, **bad}))
                got = None
                for _ in range(40):
                    m = json.loads(await ws.recv())
                    if m["t"] == "error":
                        got = m
                        break
                assert got is not None and got["msg"] == "malformed message", bad
            ws, w = await connect(s, "fine1")
            await ws.send(json.dumps({"v": PROTOCOL_VERSION, "t": "somethingnew"}))   # unknown t is ignored
            await next_snap(ws); await next_snap(ws)
            await ws.close()
        finally:
            await s.stop()
    run(body())

def test_socket_flow_welcome_tables_allot_and_chat(tmp_path):
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        await s.start("127.0.0.1", 0)
        try:
            a, wa = await connect(s, "sockera")
            b, wb = await connect(s, "sockerb")
            for k in ("world", "items", "drops", "skills", "elements", "buffs", "stat_names", "job_names", "stat_cost",
                      "mvp_kinds", "recipes", "nodes", "craft_skills", "max_job_level"):
                assert k in wa, k
            sa, sb = wa["slot"], wb["slot"]
            pb = np.asarray(s.world["players"]["pos"][sb])
            s.world["players"]["pos"] = s.world["players"]["pos"].at[sa].set(jnp_arr(pb + np.array([2, 0])))
            await a.send(json.dumps({"v": PROTOCOL_VERSION, "t": "allot", "stat": 1}))
            await a.send(json.dumps({"v": PROTOCOL_VERSION, "t": "chat", "kind": "say", "text": "over the wire"}))
            got_chat = got_stat = False
            for _ in range(80):
                m = json.loads(await b.recv())
                if m["t"] == "chat" and m["text"] == "over the wire":
                    got_chat = True
                    assert m["from"] == "sockera" and m["kind"] == "say"
                if got_chat:
                    break
            assert got_chat
            for _ in range(80):
                m = await next_snap(a)
                if m["you"].get("stats", [0, 0])[1] == 2:  # stats come on change only
                    got_stat = True
                    break
            assert got_stat
            await wait_ticks(s, 5)
            assert int(s.world["stats"]["agi"][sa]) == 2  # still 2 after more ticks: the allot was one-shot
            await a.close(); await b.close()
        finally:
            await s.stop()
    run(body())


def test_binary_format_login_gets_binary_snaps(tmp_path):
    from net.protocol import decode_snap_bin
    async def body():
        s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50)
        await s.start("127.0.0.1", 0)
        try:
            ws = await websockets.connect(f"ws://127.0.0.1:{s.port}")
            await ws.send(json.dumps({"v": PROTOCOL_VERSION, "t": "login", "name": "binny", "format": "bin"}))
            w = json.loads(await ws.recv())
            assert w["t"] == "welcome"  # welcome (and chat, mvp) stay JSON text frames
            frames = [await ws.recv() for _ in range(3)]
            assert all(isinstance(f, bytes) for f in frames)
            snaps = [decode_snap_bin(f) for f in frames]
            assert [x["full"] for x in snaps] == [True, False, False]
            me = next(e for e in snaps[0]["ents"] if e["id"] == w["slot"])
            assert me["kind"] == 0 and me["name"] == "binny" and me["pos"] == snaps[0]["you"]["pos"]
            assert "inv" in snaps[0]["you"] and "inv" not in snaps[2]["you"]  # you on change in binary too
            await ws.close()
        finally:
            await s.stop()
    run(body())
