# World persistence: server rule for `characters/world.npz` (report only)

Line numbers from `ro/accounts` at f1b5d60. Depends on Worker B's `game/world_persist.py`
(`save_world(world, path)`, `load_world(path)`, `WorldFileError`).

## Rule

- On `prepare()`: if `<chars_dir>/world.npz` exists, `load_world` it. On `WorldFileError` log a warning and build the
  fresh world with `_initial_world(self.seed)`. A missing file is not an error.
- After a load every player slot is re-seeded from the loaded `active` flags: `owner = "bot"` for active slots,
  `None` otherwise. Humans are never persisted as humans: whoever was online at save time comes back as a bot-driven
  slot. Their files in `characters/<name>.npz` are authoritative and get written back into a slot on login
  (`_write_slot`).
- Save in `stop()` after the logouts (so the file holds the final state), and on every `WORLD_SAVE_EVERY`-th
  periodic save from `_periodic_save`, via `asyncio.to_thread(save_world, host_copy, path)`.
- `host_copy = jax.tree_util.tree_map(np.asarray, self.world)` is taken on the loop first: one device pull, after which
  the thread touches only numpy. `save_world` writes atomically (`.tmp` then replace).
- Config line, add to `game/config.py` after `SAVE_EVERY_TICKS = 2400` (existing line 78):

```python
WORLD_SAVE_EVERY = 5   # world.npz is written every 5th periodic save: 5 * 2400 = 12000 ticks = 24 min at 120 ms
```

## Restored and not restored

Restored: the whole world dict (terrain, walls, pz, dist_town, players, monsters, drops, `rng`, and the world's own
`tick` leaf). After the load the server sets `self.tick = int(self.world["tick"])`.

Not restored (rebuilt empty or recomputed in `prepare`): connections and `Client` objects, `latest_input`, `tap`,
`path`, `pending`, `stuck`, `notice`, `prev_dead`, `town` for humans, `need_full`, `_walked`, `save_gen`, the `known`
visibility table (zeroed, so each client gets a full snapshot on join), and `prev_tab` (rebuilt from the loaded
world by `entity_table`). Players online at save time lose their session (they reconnect and are placed from their
character file).

## Cost of the pull (from game/config.py values)

MAP = 512, MAX_PLAYERS = 512, MAX_MONSTERS = 4096, MAX_DROPS = 2048, NT = len(TOWNS) = 5.

Static layers: `MAP*MAP*(1+1+1+2*NT)` = 262144 * 13 = 3,407,872 bytes (terrain int8, walls int8, pz bool, dist_town
int16 x NT). Entity arrays add the remainder. Summing every leaf of `make_empty_world` gives
**3,632,140 bytes (about 3.46 MiB)** per pull and per uncompressed file. The static layers never change in v1, so a
later change could skip them, but one 3.5 MiB pull per 24 min is negligible.

## Exact edits

`prepare()` existing lines 117-135. Existing:

```python
121:        self.world = _initial_world(self.seed)
...
128:        self.owner = ["bot" if a else None for a in active]
...
132:        self.chars_dir.mkdir(parents=True, exist_ok=True)
133:        self.roster = characters.load_roster(self.chars_dir)
134:        self._prepared = True
```

Replace line 121 with (mkdir moves above it; keeping line 132 is harmless):

```python
        self.chars_dir.mkdir(parents=True, exist_ok=True)
        self.world = self._load_or_build_world()
        self.tick = int(self.world["tick"])
```

and add this method to `GameServer`, next to `prepare`:

```python
    def _load_or_build_world(self) -> dict:
        path = self.chars_dir / "world.npz"
        if path.exists():
            try:
                w = load_world(path)
                log.info("loaded world from %s", path)
                return w
            except WorldFileError as e:
                log.warning("ignoring %s: %s; building a fresh world", path, e)
        return _initial_world(self.seed)
```

Line 128 already re-seeds `owner` as bots from the loaded `active` flags, so no extra code is needed there.
Imports (near existing line 30):

```python
from game.world_persist import WorldFileError, load_world, save_world  # noqa: E402
```

and add `WORLD_SAVE_EVERY` to the `game.config` import at existing lines 25-26. In `__init__` add
`self._world_saves = 0` and `self._world_task: asyncio.Task | None = None`.

`stop()`: existing last line (157)

```python
        await asyncio.gather(*self._closing, *self._sending, *self._saving, return_exceptions=True)
```

Append after it (the logouts earlier in `stop()`, `for slot in list(self.conn): self.logout(slot)`, have already
handed humans back to bots in `self.world`):

```python
        if self._world_task is not None:
            await asyncio.gather(self._world_task, return_exceptions=True)  # let an in-flight world save finish
        if self._prepared:
            host = jax.tree_util.tree_map(np.asarray, self.world)
            await asyncio.to_thread(save_world, host, self.chars_dir / "world.npz")
```

`_periodic_save` existing lines 284-302. Existing:

```python
284:    def _periodic_save(self, pull: dict) -> None:
...
290:        if loop is not None and self._saving:
...
293:        batch = self._collect_batch(pull)
294:        if not batch:
295:            return
```

The `if not batch: return` would skip the world save on a server with no humans (bots still change the world), so put
the world check between lines 293 and 294:

```python
        batch = self._collect_batch(pull)
        self._world_saves += 1
        if loop is not None and self._world_saves % WORLD_SAVE_EVERY == 0 \
                and (self._world_task is None or self._world_task.done()):
            host = jax.tree_util.tree_map(np.asarray, self.world)  # one device pull, on the loop
            self._world_task = loop.create_task(asyncio.to_thread(save_world, host, self.chars_dir / "world.npz"))
            log.info("saving world at tick %d", self.tick)
        if not batch:
            return
```

Caveat: the early return at line 290-292 (`previous periodic save still running`) happens before the counter, so a
skipped periodic save is not counted. With no event loop (direct `tick_once` call) the world is not saved periodically;
`stop()` still saves it.
