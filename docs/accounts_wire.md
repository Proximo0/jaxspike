# Accounts: wire change and server integration (report only, the lead places the edits)

Line numbers are from `ro/accounts` at f1b5d60. Nothing here is applied; this file is the patch description.

## 1. Login message

Client to server (protocol v2, `login` is still the first message):

```json
{"t": "login", "name": "alice", "password": "hunter2", "create": true}
```

`password` is required (str). `create` is optional (bool, default false). `create: true` registers the account and
then logs in; with an existing name the reply is `name taken`.

Error replies use the existing shape `{"t": "error", "msg": "<string>"}`, then the socket closes. Exact strings:

| msg | when |
|---|---|
| `bad name` | name fails `characters.valid_name` (existing; also returned by `create`) |
| `bad password` | `create` with empty or >128 char password; also a plain login with a wrong password or unknown name (does not reveal which, so names cannot be enumerated) |
| `name taken` | `create: true` for a name that already has an account file |
| `locked out` | 5 failed verifies for that name inside the 60 s lockout; checked before any hashing |
| `already online` | existing (from `GameServer.login`) |
| `server full` | existing |
| `corrupt character` | existing |

## 2. net/protocol.py `validate_client` (existing lines 35-36)

Existing:

```python
35:    if t == "login" and isinstance(msg.get("name"), str):
36:        return {"t": t, "name": msg["name"]}
```

Replace with:

```python
    if t == "login" and isinstance(msg.get("name"), str) and isinstance(msg.get("password"), str) \
            and isinstance(msg.get("create", False), bool):
        return {"t": t, "name": msg["name"], "password": msg["password"], "create": msg.get("create", False)}
```

A login without `password` becomes invalid, so the handler answers `expected login` (existing path). Existing tests and
`net/client.py` callers that omit `password` must be updated together.

## 3. net/server.py

Import, next to existing line 31 `from net import characters  # noqa: E402`:

```python
from net.accounts import AccountStore  # noqa: E402
```

`GameServer.__init__`, after existing line 81 `self.chars_dir = Path(chars_dir)`:

```python
        self.accounts = AccountStore(self.chars_dir / "accounts")
```

`AccountStore.__init__` does its own `mkdir(parents=True)`, so the `prepare()` mkdir at existing line 132
(`self.chars_dir.mkdir(parents=True, exist_ok=True)`) needs no change (keep it for the roster).

`_handler`, existing lines 470-473:

```python
470:            slot, err = self.login(first["name"])
471:            if err is not None:
472:                await self._refuse(ws, err)
473:                return
```

Replace with (verify or create off the loop first, then the synchronous slot pick):

```python
            name, password = first["name"], first["password"]
            store = self.accounts
            loop = asyncio.get_running_loop()
            key = name.lower()
            if store.locked_out(key):
                await self._refuse(ws, "locked out")
                return
            if first["create"]:
                err = await loop.run_in_executor(None, store.create, name, password)
                if err is not None:
                    await self._refuse(ws, err)  # "bad name" | "name taken" | "bad password"
                    return
            elif not await loop.run_in_executor(None, store.verify, name, password):
                await self._refuse(ws, "locked out" if store.locked_out(key) else "bad password")
                return
            if ws.close_code is not None:  # the client left while we were hashing
                return
            slot, err = self.login(name)  # synchronous: slot pick and mark stay atomic on the loop
            if err is not None:
                await self._refuse(ws, err)
                return
            store.touch_login(key)
            store.add_character(key, key)  # v1: account name == character name
```

Notes for the lead:
- `add_character` returns `"too many characters"` only above 3 distinct names; with account == character it never
  trips. Ignore the return value in v1. If the store is not idempotent for a repeated name, guard with
  `if key not in store.characters(key):`.
- `touch_login` is a small atomic JSON write, cheap, no hashing; it may run on the loop. Move it into
  `run_in_executor` if disk latency matters.
- `create` also logs in: no second hash is paid, the single scrypt cost is inside `store.create`.
- The `key = name.lower()` line mirrors the store's own lowering; `locked_out` is then keyed the same as `verify`.

## 4. Client side (report only)

`net/client.py` existing line 87:

```python
87:            await ws.send(encode({"t": "login", "name": self.name_}))
```

becomes:

```python
            await ws.send(encode({"t": "login", "name": self.name_, "password": self.password, "create": self.create}))
```

with `password: str = ""` and `create: bool = False` added to the client `__init__` and its CLI flags
(`--password`, `--create`). The bot clients and `bench/server_load.py` need a password too (a fixed test password,
`create: true` on first connect).

Godot (`client/`, report only): the login scene gets a masked `LineEdit` (`secret = true`) for the password, a
"Create account" check box, and the `login` message dictionary gains `"password": pw, "create": create`. Show the
error strings above in the existing login-error label. The password must never reach logs, the debug layer or the
recorder.

## 5. Why scrypt must not run on the event loop

`hashlib.scrypt(n=2**14, r=8, p=1)` is deliberately slow and memory-hard: about 16 MiB per call and a measured
73.7 ms mean (p95 102.8 ms) on this CPU (`bench/account_bench.py --n 200 --workers 4`; four pool threads bring it to
25.3 ms of wall time per verify). The game tick and every websocket send and
receive share one asyncio loop with `TICK_MS = 120`. A verify run inline stalls every connected player for its whole
duration (74 ms is more than half a tick), and a burst of logins (200 clients reconnecting) stacks into whole missed
ticks and `overruns`. `hashlib.scrypt` releases the GIL, so `loop.run_in_executor(None, ...)` lets the tick and the
hashing overlap on other cores. The slot pick (`GameServer.login`) stays synchronous and runs after the awaits so
nothing can interleave between "pick" and "mark". The lockout check is a dict lookup, done first on the loop, so locked
names never reach the thread pool (a flood of guesses costs no hashing). The default executor has `min(32, cpu+4)`
threads, which also bounds concurrent scrypt memory.
