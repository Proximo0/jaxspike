"""Load bench: N headless NetClients send random inputs; prints snapshot rate and bandwidth.

Usage: venv/Scripts/python.exe bench/server_load.py --embedded --clients 50 --seconds 10
"""
import argparse
import asyncio
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from net.client import NetClient  # noqa: E402


class EmbeddedServer:
    """Runs a GameServer on its own event loop thread."""

    def __init__(self, seed, chars_dir):
        self.seed, self.chars_dir = seed, chars_dir
        self.server = None
        self.port = None
        self._loop = None
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        from net.server import GameServer
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self.server = GameServer(self.seed, Path(self.chars_dir))
            self._loop.run_until_complete(self.server.start("127.0.0.1", 0))
            self.port = self.server.port
        except Exception as e:  # noqa: BLE001
            self._error = e
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    def start(self):
        self._thread.start()
        self._ready.wait()
        if self._error:
            raise self._error

    def stop(self):
        fut = asyncio.run_coroutine_threadsafe(self.server.stop(), self._loop)
        fut.result(timeout=60)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--embedded", action="store_true")
    args = ap.parse_args()

    emb = None
    tmp = None
    host, port = args.host, args.port
    if args.embedded:
        tmp = tempfile.TemporaryDirectory()
        emb = EmbeddedServer(0, tmp.name)
        print("starting embedded server (jit warmup) ...", flush=True)
        emb.start()
        host, port = "127.0.0.1", emb.port

    clients = [NetClient(host, port, f"load_{i}") for i in range(args.clients)]
    for c in clients:
        c.start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not all(c.welcome or c.error for c in clients):
        time.sleep(0.1)

    live = [c for c in clients if c.welcome and c.connected]
    refused = [c for c in clients if c.error]
    if refused:
        print("refusal reasons:", sorted({c.error for c in refused}))
    timed_out = [c for c in clients if not c.welcome and not c.error]
    # Per-client tally from NetClient's own counters (snap messages and snap text length; terrain is in
    # welcome only, so it is not in these bytes).
    base_snaps = {id(c): c.snap_count for c in live}
    base_bytes = {id(c): c.snap_bytes for c in live}
    rng = random.Random(0)
    start = time.monotonic()
    next_input = start
    while time.monotonic() - start < args.seconds:
        now = time.monotonic()
        if now >= next_input:
            for c in live:
                c.send_input(rng.randint(0, 4), rng.random() < 0.3, rng.random() < 0.1)
            next_input = now + 0.5
        time.sleep(0.01)
    elapsed = time.monotonic() - start
    still = [c for c in live if c.connected]

    rates = [(c.snap_count - base_snaps[id(c)]) / elapsed for c in live] or [0.0]
    bps = [(c.snap_bytes - base_bytes[id(c)]) / elapsed for c in live] or [0.0]
    print(f"clients requested: {args.clients}, connected: {len(still)}, refused: {len(refused)}, timed out: {len(timed_out)}")
    print(f"snapshots/client/s: mean {sum(rates)/len(rates):.2f}, min {min(rates):.2f}")
    print(f"bytes/client/s (snaps, no terrain): mean {sum(bps)/len(bps):.0f}, max {max(bps):.0f}")
    if emb:
        print(f"server tick overruns: {emb.server.overruns} (ticks run: {emb.server.tick})")

    for c in clients:
        c.close()
    if emb:
        emb.stop()
        tmp.cleanup()


if __name__ == "__main__":
    main()
