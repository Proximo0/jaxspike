"""Threaded WebSocket client: an asyncio loop in a daemon thread, polled by the UI thread."""
import asyncio
import threading

import websockets

from net.protocol import decode, encode


class NetClient(threading.Thread):
    def __init__(self, host: str, port: int, name: str):
        super().__init__(daemon=True)
        self.host, self.port, self.name_ = host, port, name
        self.welcome = None
        self.error = None
        self.connected = False
        self._snap = None
        self._lock = threading.Lock()
        self._loop = None
        self._queue = None
        self._pending = []   # messages sent before the loop is ready
        self._closing = False

    def latest(self):
        with self._lock:
            return self._snap

    def _send(self, msg):
        with self._lock:
            loop, q = self._loop, self._queue
            if loop is None:
                self._pending.append(msg)
                return
        try:
            loop.call_soon_threadsafe(q.put_nowait, msg)
        except RuntimeError:
            pass  # loop already closed

    def send_input(self, move: int, attack: bool, eat: bool) -> None:
        self._send({"t": "input", "move": int(move), "attack": bool(attack), "eat": bool(eat)})

    def walk_to(self, x: int, y: int) -> None:
        self._send({"t": "walk_to", "x": int(x), "y": int(y)})

    def close(self) -> None:
        self._closing = True
        self._send(None)

    def run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:  # connection refused, dropped, etc.
            if self.error is None and not self._closing:
                self.error = f"connection failed: {e}"
        finally:
            self.connected = False

    async def _main(self):
        q = asyncio.Queue()
        with self._lock:
            self._loop, self._queue = asyncio.get_running_loop(), q
            for m in self._pending:
                q.put_nowait(m)
            self._pending.clear()
        async with websockets.connect(f"ws://{self.host}:{self.port}", max_size=None) as ws:
            await ws.send(encode({"t": "login", "name": self.name_}))
            msg = decode(await ws.recv())
            if msg is None or msg["t"] == "error":
                self.error = (msg or {}).get("msg", "bad reply")
                return
            if msg["t"] != "welcome":
                self.error = "unexpected reply"
                return
            self.welcome = msg
            self.connected = True

            async def recv_loop():
                async for text in ws:
                    m = decode(text)
                    if m is None:
                        continue
                    if m["t"] == "snap":
                        with self._lock:
                            self._snap = m
                    elif m["t"] == "error":
                        self.error = m.get("msg", "error")

            async def send_loop():
                while True:
                    m = await q.get()
                    if m is None:
                        return
                    await ws.send(encode(m))

            tasks = [asyncio.ensure_future(recv_loop()), asyncio.ensure_future(send_loop())]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in tasks:
                    t.cancel()
                self.connected = False
