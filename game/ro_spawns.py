"""Monster kind assignment by field band and dungeon (ro/world), spawn density per band (ro3-balance). Pure JAX.

Balance lives here as data (docs/round3_contract.md: never by softening the RO formulas):
- BAND_DENSITY / DUNGEON_DENSITY: relative monsters per walkable tile in each field band and inside dungeons.
  `spawn_tiles` samples the population's tiles with these weights (bots/runner.py's placement line), so a band's
  monster count is its tile count x its weight, normalised over the map. Band 0 (the Novice ring around every town)
  is small: at weight 1 everywhere it held 74 of 1024 monsters per world for 128 bots (2026-11-02 CPU soak), and
  the bots went looking in band 1, where Vexblob, Capling, Fiddlehop and Pilferbug killed level-1 Novices (deaths 4.0
  per 1000 bot-ticks there against 0.44 in band 0).
- SPAWN_MIX: per-band kind weights (the kinds themselves stay world_layout.FIELD_BANDS). Band 1 leans on Bramblet
  and Fiddlehop; Vexblob (59-72 ATK one-shots a 40 hp Novice) and the aggressive Pilferbug are rarer.
"""
import jax
import jax.numpy as jnp
import numpy as np

from game import map_gen, world_layout
from game.config import GATE_APRON_TILES, MAP
from game.data.monsters import KIND_BY_NAME
from game.map_gen import DIST_UNREACHABLE
from game.world_layout import DUNGEONS, FIELD_BANDS

N_BANDS = len(FIELD_BANDS)
N_DUNGEONS = len(DUNGEONS)


def _table(rows):
    """Return (int32 [R, MAX] padded by repeating the last kind, int32 [R] real counts)."""
    width = max(len(r) for r in rows)
    ids = [[KIND_BY_NAME[n] for n in r] for r in rows]
    tab = np.array([r + [r[-1]] * (width - len(r)) for r in ids], np.int32)
    return jnp.asarray(tab), jnp.asarray([len(r) for r in ids], jnp.int32)


BAND_KINDS, BAND_COUNT = _table([b[2] for b in FIELD_BANDS])
DUNGEON_KINDS, DUNGEON_COUNT = _table([d[4] for d in DUNGEONS])

# Relative spawn weight per walkable tile: field bands 0..N_BANDS-1, then dungeons. Tuned 2026-11-02 (ro3-balance).
BAND_DENSITY = (6.0, 1.5, 0.55)
DUNGEON_DENSITY = 1.0
assert len(BAND_DENSITY) == N_BANDS

# Kind weights per band, by name (every name must be in that band's FIELD_BANDS list; missing names weigh 0).
SPAWN_MIX = (
    {"Blobbin": 30, "Wrigglet": 25, "Moonbun": 25, "Emberblob": 20},
    {"Bramblet": 35, "Fiddlehop": 30, "Capling": 15, "Pilferbug": 12, "Vexblob": 8},
    {"Snarlfang": 25, "Gallopeck": 25, "Sandgulper": 20, "Grumblebear": 15, "Grukk Brute": 15},
)
assert len(SPAWN_MIX) == N_BANDS


def _mix_table():
    """float32 [N_BANDS, W] cumulative kind weights (ending at 1.0) in BAND_KINDS order, padded with 1.0."""
    width = BAND_KINDS.shape[1]
    out = np.ones((N_BANDS, width), np.float32)
    for b, (lo, hi, names) in enumerate(FIELD_BANDS):
        w = np.array([float(SPAWN_MIX[b].get(n, 0.0)) for n in names], np.float64)
        assert set(SPAWN_MIX[b]) <= set(names) and w.sum() > 0, (b, SPAWN_MIX[b])
        out[b, : len(names)] = np.cumsum(w / w.sum())
        out[b, len(names) - 1] = 1.0
    return jnp.asarray(out)


BAND_MIX_CUM = _mix_table()
DUNGEON_MVP = jnp.asarray([KIND_BY_NAME[d[5]] for d in DUNGEONS], jnp.int32)
_BAND_EDGES = tuple(float(b[1]) * MAP for b in FIELD_BANDS[:-1])   # upper edge of every band but the last


def _band_of_dist(d):
    band = jnp.zeros(d.shape, jnp.int32)
    for edge in _BAND_EDGES:
        band += (d >= edge).astype(jnp.int32)
    return jnp.where(d >= DIST_UNREACHABLE, N_BANDS - 1, band)


def band_of(world, pos):
    """Return int32 [n]: field band of each pos [n, 2] by walking distance to the nearest gate."""
    field = world["dist_town"].min(axis=0)
    return _band_of_dist(field[pos[:, 0], pos[:, 1]].astype(jnp.int32))


def kinds_for(world, pos, key):
    """Return int32 [n] monster kinds for pos [n, 2]; the first position in each dungeon is its MVP."""
    n = pos.shape[0]
    band = band_of(world, pos)
    dun = world_layout.dungeon_index(pos)
    inside = dun >= 0
    k_dun, k_band = jax.random.split(key)
    count = DUNGEON_COUNT[jnp.maximum(dun, 0)]
    pick = jax.random.randint(k_dun, (n,), 0, count)   # array maxval: uniform over the real dungeon kinds
    u = jax.random.uniform(k_band, (n,))
    band_pick = jnp.minimum((u[:, None] >= BAND_MIX_CUM[band]).sum(axis=1), BAND_COUNT[band] - 1)  # weighted
    kind = jnp.where(inside, DUNGEON_KINDS[jnp.maximum(dun, 0), pick], BAND_KINDS[band, band_pick])
    for k in range(N_DUNGEONS):
        in_k = dun == k
        kind = jnp.where(in_k & (jnp.cumsum(in_k) == 1), DUNGEON_MVP[k], kind)
    return kind.astype(jnp.int32)


def choose_kinds(world, key, n):
    """Return int32 [n]: kinds for the first n monster slots' positions."""
    return kinds_for(world, world["monsters"]["pos"][:n], key)


def tile_weights(world):
    """float32 [MAP, MAP] relative spawn weight per tile: BAND_DENSITY by band, DUNGEON_DENSITY inside a dungeon,
    0 on walls, protection zones and within GATE_APRON_TILES walking distance of a gate."""
    gate = world["dist_town"].min(axis=0).astype(jnp.int32)
    xs, ys = jnp.meshgrid(jnp.arange(MAP, dtype=jnp.int32), jnp.arange(MAP, dtype=jnp.int32), indexing="ij")
    dun = world_layout.dungeon_index(jnp.stack([xs.reshape(-1), ys.reshape(-1)], 1)).reshape(MAP, MAP)
    w = jnp.asarray(BAND_DENSITY, jnp.float32)[_band_of_dist(gate)]
    w = jnp.where(dun >= 0, jnp.float32(DUNGEON_DENSITY), w)
    ok = (world["walls"] == 0) & ~world["pz"] & (gate > GATE_APRON_TILES)
    return jnp.where(ok, w, 0.0).astype(jnp.float32)


def spawn_tiles(world, key, n):
    """int32 [n, 2] distinct spawn tiles drawn without replacement with probability ~ tile_weights (one score per
    tile: Efraimidis-Spirakis keys -log(u) / w, the n smallest win; weight 0 never wins while weighted tiles remain).
    Drop-in for bots/runner._distinct_tiles: `mpos = ro_spawns.spawn_tiles(world, key, POP_MONSTERS)`."""
    w = tile_weights(world).reshape(-1)
    u = jax.random.uniform(key, (MAP * MAP,), minval=1e-12, maxval=1.0)
    score = jnp.where(w > 0, -jnp.log(u) / jnp.maximum(w, 1e-12), jnp.inf)   # smaller = better
    idx = jnp.argsort(score)[:n]
    return jnp.stack([idx // MAP, idx % MAP], axis=1).astype(jnp.int32)


def repopulate(world, key, n):
    """The world with its first n monster slots re-placed by spawn_tiles and re-kinded (hp full, active). For
    benches that compare densities on worlds built by bots/runner.py; the runner itself calls spawn_tiles."""
    from game.data.monsters import HP as MON_HP
    k_pos, k_kind = jax.random.split(key)
    pos = spawn_tiles(world, k_pos, n)
    kind = kinds_for(world, pos, k_kind)
    m = world["monsters"]
    monsters = {**m, "pos": m["pos"].at[:n].set(pos), "spawn": m["spawn"].at[:n].set(pos),
                "kind": m["kind"].at[:n].set(kind), "hp": m["hp"].at[:n].set(MON_HP[kind]),
                "active": m["active"].at[:n].set(True)}
    return {**world, "monsters": monsters}


def spawn_plan(terrain, dist_town):
    """Return int32 [MAP, MAP]: band id per tile (field 0..N_BANDS-1, N_BANDS + dungeon index inside
    dungeons, -1 on blocked or protection-zone tiles). Debug / minimap helper."""
    xs, ys = jnp.meshgrid(jnp.arange(MAP, dtype=jnp.int32), jnp.arange(MAP, dtype=jnp.int32), indexing="ij")
    pos = jnp.stack([xs.reshape(-1), ys.reshape(-1)], 1)
    dun = world_layout.dungeon_index(pos).reshape(MAP, MAP)
    plan = jnp.where(dun >= 0, N_BANDS + dun, _band_of_dist(dist_town.min(axis=0).astype(jnp.int32)))
    bad = (map_gen.blocking(terrain) == 1) | map_gen.protection_zone(terrain)
    return jnp.where(bad, -1, plan).astype(jnp.int32)
