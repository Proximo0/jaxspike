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
