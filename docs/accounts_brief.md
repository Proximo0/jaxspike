# Accounts and world persistence: worker contract (branch `ro/accounts`)

Sub-orchestrator-owned (2026-11-02). Three Sonnet workers, disjoint files, **new files only**. Nobody edits an existing
file under `net/`, `game/`, `bench/`, `tests/` or `docs/`: integration lines are *reported* in the wire docs.
Python: `C:\Users\crp27\jaxspike\venv\Scripts\python.exe` (absolute; this worktree has no venv). Run
`-m pytest tests -q` before hand-back. Accounts and persistence are legitimately Python (irregular-shaped I/O);
nothing here touches the JAX tick.

## Worker A: `net/accounts.py` + `tests/test_accounts.py`

```python
class AccountStore:
    MAX_CHARACTERS = 3
    LOCKOUT_FAILS = 5          # failed verifies before lockout
    LOCKOUT_SECONDS = 60.0
    SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
    SALT_BYTES = 32

    def __init__(self, dir, clock=time.monotonic): ...   # dir created if missing (mkdir parents)
    def exists(self, name) -> bool
    def create(self, name, password) -> str | None      # None = ok; else reason: "bad name" | "name taken" | "bad password"
    def verify(self, name, password) -> bool             # False for unknown name, wrong password, or locked out
    def locked_out(self, name) -> bool
    def add_character(self, account, char_name) -> str | None  # None ok; "too many characters" (>3) | "no such account" | "bad name"
    def characters(self, account) -> list[str]           # [] for unknown account
    def touch_login(self, name) -> None                  # sets last_login = time.time(); atomic write
```

- Names go through `net.characters.valid_name` (lower-cased `[a-z0-9_]{3,16}`); `create`/`verify` use the lowered key.
  `"bad password"` = empty or longer than 128 characters (no other complexity rule).
- Hash: `hashlib.scrypt(password.encode("utf-8"), salt=os.urandom(32), n=2**14, r=8, p=1, dklen=32)`.
  Compare with `hmac.compare_digest`. A thread-safe `threading.Lock` guards the file writes (same pattern as
  `net/characters.py`'s `_save_lock`) so `verify` may run in a thread pool while `create`/`touch_login` run on the loop.
- File: `<dir>/<name>.json` with keys `{"v": 1, "salt": hex, "hash": hex, "n": 16384, "r": 8, "p": 1,
  "created": float, "last_login": float | null, "characters": [str, ...]}`. Writes are atomic: `<name>.json.tmp` then
  `os.replace`. A corrupt or incomplete file (not JSON, missing keys, non-hex salt/hash, wrong length) makes
  `verify` return False and `exists` return True (so `create` answers `"name taken"`, never overwrites).
- Rate limit, in memory per store instance, keyed by name: a failed `verify` (wrong password OR unknown name)
  increments a counter; at 5 failures the name is locked until `clock() + 60`; while locked `verify` returns False
  without hashing; a successful verify resets the counter. `clock` is injectable (default `time.monotonic`).
  The server will treat "locked out" as its own error string, so expose `locked_out(name)` for it to check first.
- `verify` for an unknown name must still take roughly as long as a real verify (hash against a fixed dummy salt) so
  names cannot be enumerated by timing.
- Tests (`tests/test_accounts.py`, `tmp_path`, fake clock as a list-backed counter): create then verify ok; wrong
  password False; second create `"name taken"`; bad names; 5 wrong -> `locked_out` True, correct password still
  False, advance clock 61 s -> verify True and counter reset; corrupt file (`b"{"`, `b""`, missing keys) -> verify
  False, exists True, create `"name taken"`; no `*.tmp` left after create/touch_login; add_character three ok,
  fourth `"too many characters"`; `characters()` round-trips through a fresh `AccountStore(dir)`.
  To keep tests fast, build the store with `AccountStore(tmp_path, clock=fake)` and monkeypatch
  `AccountStore.SCRYPT_N = 2 ** 8` in a fixture (store must read the class attributes at call time, and must write
  the `n/r/p` it used into the file and read them back for verify, so files hashed with other params still verify).

## Worker B: `game/world_persist.py` + `tests/test_world_persist.py`

```python
WORLD_FORMAT_VERSION = 1

class WorldFileError(ValueError): ...   # message names the key and both shapes/dtypes, or the version mismatch

def save_world(world, path) -> None     # atomic: <path>.tmp then os.replace; built on game.persistence._flatten
def load_world(path) -> dict            # nested dict of jnp arrays; raises WorldFileError
def world_matches(world, template) -> str | None   # None if every leaf has the same shape and dtype, else the first mismatch message
```

- Build on `game/persistence.py` (flattened `a/b` keys). Add two extra leaves at save: `__version__` (int32 scalar)
  and `__layout__` (an int32 array `[MAP, MAX_PLAYERS, MAX_MONSTERS, MAX_DROPS, INV_SLOTS, len(TOWNS)]`). On load,
  refuse when `__version__ != WORLD_FORMAT_VERSION` or `__layout__` differs from the current `game.config` values,
  with an error message like `world.npz: MAP 256 != 512 (rebuild the world)`. Then strip the two leaves.
- `load_world` also validates the loaded tree against `make_empty_world(None)`: same key paths, same shape and dtype
  per leaf (use `jax.tree_util.tree_flatten_with_path`). Extra keys in the file that `make_empty_world` lacks (future
  system blocks like `"stats"`) are allowed and kept; a missing key is an error. Do not validate values.
- Corrupt file (not an npz, truncated) -> `WorldFileError` (wrap `OSError, ValueError, zipfile.BadZipFile, zlib.error,
  EOFError, KeyError`).
- Tests: round trip exact (same shapes, dtypes, values; `rng` stays uint32; set a few non-default values incl. a drop
  and a monster `respawn_in`); a file written with `__version__ = 99` is refused; a hand-built npz where
  `players/hp` has shape `[MAX_PLAYERS + 1]` is refused naming `players/hp`; a layout with `MAP + 1` is refused
  naming MAP; a garbage file is refused; no `.tmp` left behind; extra key `stats/str` survives the round trip.
  Use `make_empty_world(jax.random.PRNGKey(3))`; keep the test under 30 s on CPU.

## Worker C: `docs/accounts_wire.md`, `docs/world_persist.md`, `bench/account_bench.py`

- `docs/accounts_wire.md`: the `login` message grows `password: str` and optional `create: true`. Reply errors
  (exact strings): `bad password`, `name taken`, `locked out`, plus the existing `bad name`, `already online`,
  `server full`, `corrupt character`. Show the exact change to `net/protocol.py`'s `validate_client` login branch
  (accept `password` str, optional `create` bool) and to `net/server.py`'s `_handler` around
  `slot, err = self.login(first["name"])` (line ~470): run `accounts.verify` (or `create`) in the default thread pool
  with `await asyncio.get_running_loop().run_in_executor(None, store.verify, name, password)` BEFORE the synchronous
  `self.login(name)` so the slot pick stays atomic on the loop; check `store.locked_out(name)` first and refuse with
  `locked out`; on `create: true` with an existing name refuse `name taken`; on success call `store.touch_login`
  and `store.add_character(name, name)` (v1: the account name is the character name; the 3-character limit is
  reserved for a later `character` field). Also the `GameServer.__init__` line `self.accounts = AccountStore(self.chars_dir / "accounts")`
  and the `prepare()` mkdir. Client side: one line for `net/client.py` (`login` gains `password`, `create`) and the
  Godot login scene field (report only). Explain the scrypt cost (~50 ms at n=2**14 on this CPU, from the bench) and why
  the verify must not run on the loop (a 50 ms stall is half a tick at 120 ms).
- `docs/world_persist.md`: the server rule. On `prepare()`: if `<chars_dir>/world.npz` exists, `load_world` it; on
  `WorldFileError` log a warning and build the fresh world (`_initial_world(seed)`); after a load, every player slot
  is re-seeded (`owner = "bot"` for active slots; humans are never persisted as humans: their files in `characters/`
  are authoritative and get written back on login). Save in `stop()` after the logouts, and every
  `WORLD_SAVE_EVERY` periodic saves (propose `WORLD_SAVE_EVERY = 5` as a `game/config.py` line, so every 12000 ticks
  = 24 min at 120 ms) from `_periodic_save` via `asyncio.to_thread(save_world, world_host_copy, path)` where
  `world_host_copy = jax.tree_util.tree_map(np.asarray, self.world)` is taken on the loop first (one device pull;
  note the cost: MAP*MAP*(1+1+1+2*NT) bytes + entity arrays; cite the number). Show the exact lines for `prepare`,
  `stop`, `_periodic_save`. State what is NOT restored (connections, paths, notices, `known` tables, the tick counter
  is restored).
- `bench/account_bench.py`: `--n 200 --workers 4`; creates one account in a temp dir, times 200 sequential
  `verify` calls (mean, p50, p95 ms) and 200 through `concurrent.futures.ThreadPoolExecutor(workers)` (wall ms, per
  verify); prints `ACCOUNT_BENCH verify_mean_ms=<x> p95_ms=<y> pool_wall_ms=<z> n=200 workers=4` and a one-line
  verdict (`run verify in the thread pool: <mean> ms > 10% of TICK_MS` or not). Import `AccountStore` from
  `net.accounts` as specified above; the bench will be run once Worker A lands.

## Hand-back format (each worker)
Files written, the pytest summary line for your tests and for the full suite, anything you could not do, and one
line of "what the lead must wire" if any.
