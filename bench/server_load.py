"""Load bench: N headless clients send random inputs; prints snapshot rate, bandwidth and server overruns.

Usage: venv/Scripts/python.exe bench/server_load.py --spawn --clients 480 --tick-ms 100 --format bin
  --spawn     the server runs in its own process (as in production) and light asyncio clients run in worker
              processes, so client work does not compete with the server for its GIL; overruns are read from the
              server's tick log over the measured window only.
  --embedded  the older mode: a server thread plus one NetClient thread per client, all in this process.
"""
import argparse
import asyncio
import json
import multiprocessing
import os
import re
import subprocess
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

    def __init__(self, seed, chars_dir, **server_kw):
        self.seed, self.chars_dir, self.server_kw = seed, chars_dir, server_kw
        self.server = None
        self.port = None
        self._loop = None
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            from net.server import GameServer
            self.server = GameServer(self.seed, Path(self.chars_dir), **self.server_kw)
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


def _light_main(job):
    """Worker process: run this worker's clients on one event loop; returns [(snaps, bytes, error or None)]."""
    host, port, names, fmt, t_start, t_end = job

    async def one(name, rng):
        import websockets
        for attempt in range(20):  # the accept backlog can refuse part of a 480-client burst: retry
            try:
                ws = await websockets.connect(f"ws://{host}:{port}", max_size=None, open_timeout=120)
                break
            except OSError as e:
                if attempt == 19:
                    return 0, 0, f"connect: {type(e).__name__}"
                await asyncio.sleep(0.5 + rng.random())
            except Exception as e:  # noqa: BLE001
                return 0, 0, f"connect: {type(e).__name__}"
        login = {"v": 2, "t": "login", "name": name}
        if fmt != "json":
            login["format"] = fmt
        tally = [0, 0]
        try:
            await ws.send(json.dumps(login))
            first = json.loads(await ws.recv())
            if first.get("t") != "welcome":
                return 0, 0, first.get("msg", "refused")

            async def recv():
                async for m in ws:
                    if t_start <= time.time() < t_end and (isinstance(m, bytes) or m.startswith('{"t":"snap"')):
                        tally[0] += 1
                        tally[1] += len(m)

            async def send():
                while True:
                    await ws.send(json.dumps({"v": 2, "t": "input", "move": rng.randint(0, 4),
                                              "attack": rng.random() < 0.3, "eat": rng.random() < 0.1}))
                    await asyncio.sleep(0.5)

            tasks = [asyncio.ensure_future(recv()), asyncio.ensure_future(send())]
            await asyncio.wait(tasks, timeout=max(t_end - time.time(), 0), return_when=asyncio.FIRST_COMPLETED)
            for t in tasks:
                t.cancel()
            err = None if time.time() >= t_end - 0.5 else "dropped"
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
        finally:
            await ws.close()
        return tally[0], tally[1], err

    async def run():
        return await asyncio.gather(*(one(n, random.Random(n)) for n in names))

    return asyncio.run(run())


_STATS = re.compile(r"tick (\d+): max tick duration over the last \d+ ticks: ([\d.]+) ms, overruns (\d+)")


def _spawn(args):
    """Server subprocess + light clients in worker processes; prints the same summary lines as --embedded."""
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # the killed server may still hold a save file
    cmd = [sys.executable, *([args.server_script] if args.server_script else ["-m", "net.server"]), "--host", "127.0.0.1", "--port", "0", "--chars", tmp.name,
           "--stats-every", "10", "--platform", args.platform]
    if args.tick_ms:
        cmd += ["--tick-ms", str(args.tick_ms)]
    if args.no_pipeline:
        cmd += ["--no-pipeline"]
    root = str(Path(__file__).resolve().parent.parent)
    srv = subprocess.Popen(cmd, cwd=root, stderr=subprocess.PIPE, text=True,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
    records, port, ready = [], [None], threading.Event()

    def pump():
        for line in srv.stderr:
            m = re.search(r"listening on [\d.]+:(\d+)", line)
            if m:
                port[0] = int(m.group(1))
                ready.set()
            m = _STATS.search(line)
            if m:
                records.append((time.time(), int(m.group(1)), float(m.group(2)), int(m.group(3))))
            if args.server_log:
                sys.stderr.write(line)
        ready.set()

    threading.Thread(target=pump, daemon=True).start()
    print("starting server process (jit warmup) ...", flush=True)
    ready.wait(600)
    if port[0] is None:
        print("server did not start")
        srv.kill()
        return
    workers = max(1, min(args.workers, args.clients))
    t_start = time.time() + args.connect_s
    t_end = t_start + args.seconds
    names = [f"load_{i}" for i in range(args.clients)]
    jobs = [("127.0.0.1", port[0], names[w::workers], args.format, t_start, t_end) for w in range(workers)]
    with multiprocessing.Pool(workers) as pool:
        res = [r for part in pool.map(_light_main, jobs) for r in part]
    time.sleep(1.5)  # one more stats line after the window
    srv.terminate()
    srv.wait(30)
    ok = [r for r in res if r[2] is None]
    errs = sorted({r[2] for r in res if r[2] is not None})
    rates = [r[0] / args.seconds for r in ok] or [0.0]
    bps = [r[1] / args.seconds for r in ok] or [0.0]
    before = [r for r in records if r[0] <= t_start]
    inside = [r for r in records if t_start < r[0] <= t_end + 1.2]
    o0 = before[-1][3] if before else 0
    o1 = inside[-1][3] if inside else o0
    ticks = (inside[-1][1] - (before[-1][1] if before else 0)) if inside else 0
    worst = max((r[2] for r in inside), default=0.0)
    print(f"clients requested: {args.clients}, ok for the whole window: {len(ok)}, failed: {len(res) - len(ok)} {errs}")
    print(f"format {args.format}, tick_ms {args.tick_ms or 'default'}, pipeline {not args.no_pipeline}, "
          f"window {args.seconds:.0f} s")
    print(f"snapshots/client/s: mean {sum(rates)/len(rates):.2f}, min {min(rates):.2f}")
    print(f"bytes/client/s (snaps, no terrain): mean {sum(bps)/len(bps):.0f}, max {max(bps):.0f}")
    print(f"server tick overruns in window: {o1 - o0} (ticks in window: {ticks}), max tick {worst:.1f} ms")
    tmp.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--embedded", action="store_true")
    ap.add_argument("--spawn", action="store_true", help="server in its own process, light clients in workers")
    ap.add_argument("--tick-ms", type=int, default=None, help="server tick (default: game.config.TICK_MS)")
    ap.add_argument("--format", choices=("json", "bin"), default="json", help="snap format asked for at login")
    ap.add_argument("--workers", type=int, default=4, help="--spawn: client worker processes")
    ap.add_argument("--connect-s", type=float, default=30, help="--spawn: seconds allowed for logins before the window")
    ap.add_argument("--platform", choices=("cpu", "gpu"), default="cpu", help="--spawn: server jax backend")
    ap.add_argument("--server-log", action="store_true", help="--spawn: echo the server's log")
    ap.add_argument("--no-pipeline", action="store_true", help="--spawn: server runs each tick start to end")
    ap.add_argument("--server-script", help="--spawn: run this script (same flags) instead of -m net.server")
    args = ap.parse_args()
    if args.spawn:
        _spawn(args)
        return

    emb = None
    tmp = None
    host, port = args.host, args.port
    if args.embedded:
        tmp = tempfile.TemporaryDirectory()
        emb = EmbeddedServer(0, tmp.name, **({"tick_ms": args.tick_ms} if args.tick_ms else {}))
        print("starting embedded server (jit warmup) ...", flush=True)
        emb.start()
        host, port = "127.0.0.1", emb.port

    clients = [NetClient(host, port, f"load_{i}", fmt=args.format) for i in range(args.clients)]
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
