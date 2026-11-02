import asyncio, threading, time
import numpy as np
from game.config import MAP
from net.client import NetClient, apply_snap
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
        c.join(5)
        assert not c.is_alive()
    finally:
        h["loop"].call_soon_threadsafe(h["stop"].set_result, None); t.join(10)

def test_client_reports_error(tmp_path):
    h, t = _serve(tmp_path)
    try:
        c = NetClient("127.0.0.1", h["s"].port, "x"); c.start()
        assert _wait(lambda: c.error is not None) and "name" in c.error
        assert c.welcome is None
    finally:
        h["loop"].call_soon_threadsafe(h["stop"].set_result, None); t.join(10)

def test_client_decodes_terrain_and_tracks_entities(tmp_path):
    h, t = _serve(tmp_path)
    try:
        c = NetClient("127.0.0.1", h["s"].port, "kim"); c.start()
        assert _wait(lambda: c.welcome is not None) and _wait(lambda: c.latest() is not None)
        assert c.terrain.shape == (MAP, MAP) and (c.terrain == np.asarray(h["s"].world["terrain"])).all()
        assert c.pz.dtype == bool and (c.pz == np.asarray(h["s"].world["pz"])).all()
        slot = c.welcome["slot"]
        assert _wait(lambda: slot in c.entities())
        me = c.entities()[slot]
        assert me["name"] == "kim" and "terrain" not in c.latest() and c.snap_count > 0 and c.snap_bytes > 0
        c.close(); c.join(5)
    finally:
        h["loop"].call_soon_threadsafe(h["stop"].set_result, None); t.join(10)

def test_apply_snap_upserts_deletes_and_full_resets():
    ents = {}
    apply_snap(ents, {"full": True, "gone": [], "ents": [{"id": 1, "hp": 5}, {"id": 2, "hp": 6}]})
    apply_snap(ents, {"full": False, "gone": [2], "ents": [{"id": 1, "hp": 4}, {"id": 3, "hp": 9}]})
    assert ents == {1: {"id": 1, "hp": 4}, 3: {"id": 3, "hp": 9}}
    apply_snap(ents, {"full": True, "gone": [], "ents": [{"id": 7, "hp": 1}]})
    assert list(ents) == [7]
