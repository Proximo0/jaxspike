"""Monster kind assignment by field band and dungeon (ro/world). Pure JAX, fixed shapes."""
import jax
import jax.numpy as jnp
import numpy as np

from game import map_gen, world_layout
from game.config import MAP
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
    count = jnp.where(inside, DUNGEON_COUNT[jnp.maximum(dun, 0)], BAND_COUNT[band])
    pick = jax.random.randint(key, (n,), 0, count)   # array maxval: uniform over the real kinds
    kind = jnp.where(inside, DUNGEON_KINDS[jnp.maximum(dun, 0), pick], BAND_KINDS[band, pick])
    for k in range(N_DUNGEONS):
        in_k = dun == k
        kind = jnp.where(in_k & (jnp.cumsum(in_k) == 1), DUNGEON_MVP[k], kind)
    return kind.astype(jnp.int32)


def choose_kinds(world, key, n):
    """Return int32 [n]: kinds for the first n monster slots' positions."""
    return kinds_for(world, world["monsters"]["pos"][:n], key)


def spawn_plan(terrain, dist_town):
    """Return int32 [MAP, MAP]: band id per tile (field 0..N_BANDS-1, N_BANDS + dungeon index inside
    dungeons, -1 on blocked or protection-zone tiles). Debug / minimap helper."""
    xs, ys = jnp.meshgrid(jnp.arange(MAP, dtype=jnp.int32), jnp.arange(MAP, dtype=jnp.int32), indexing="ij")
    pos = jnp.stack([xs.reshape(-1), ys.reshape(-1)], 1)
    dun = world_layout.dungeon_index(pos).reshape(MAP, MAP)
    plan = jnp.where(dun >= 0, N_BANDS + dun, _band_of_dist(dist_town.min(axis=0).astype(jnp.int32)))
    bad = (map_gen.blocking(terrain) == 1) | map_gen.protection_zone(terrain)
    return jnp.where(bad, -1, plan).astype(jnp.int32)
