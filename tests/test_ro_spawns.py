import jax
import jax.numpy as jnp
import numpy as np

from game import map_gen, ro_spawns, world_layout
from game.config import MAP
from game.data.terrain import T_CAVE
from game.data.monsters import KIND_BY_NAME as K
from game.state import make_empty_world
from game.world_layout import DUNGEONS, FIELD_BANDS

_terrain = map_gen.make_terrain_ro(jax.random.PRNGKey(0))
_fields = map_gen.distance_fields_ro(_terrain)
WORLD = {**make_empty_world(jax.random.PRNGKey(1)), "terrain": _terrain, "dist_town": _fields}
GATE = np.asarray(_fields.min(axis=0))
TER = np.asarray(_terrain)
WALK = (np.asarray(map_gen.blocking(_terrain)) == 0) & ~np.asarray(map_gen.protection_zone(_terrain))
DIST = (GATE < 32767)


def _names(band):
    return {K[n] for n in FIELD_BANDS[band][2]}


def _tiles(mask, n, seed=0):
    xy = np.argwhere(mask)
    rng = np.random.default_rng(seed)
    return jnp.asarray(xy[rng.choice(len(xy), min(n, len(xy)), replace=False)], jnp.int32)


def test_tables():
    assert ro_spawns.BAND_KINDS.shape[0] == len(FIELD_BANDS)
    assert ro_spawns.DUNGEON_KINDS.shape[0] == len(DUNGEONS)
    assert ro_spawns.BAND_KINDS.dtype == jnp.int32


def test_field_bands():
    outside = WALK & (TER != T_CAVE) & DIST & (np.asarray(world_layout.dungeon_index(
        jnp.stack(jnp.meshgrid(jnp.arange(MAP), jnp.arange(MAP), indexing="ij"), -1).reshape(-1, 2))).reshape(MAP, MAP) < 0)
    for band, (lo, hi, _) in enumerate(FIELD_BANDS):
        m = outside & (GATE >= lo * MAP) & (GATE < hi * MAP)
        assert m.any()
        pos = _tiles(m, 300, band)
        assert (np.asarray(ro_spawns.band_of(WORLD, pos)) == band).all()
        kinds = np.asarray(ro_spawns.kinds_for(WORLD, pos, jax.random.PRNGKey(band)))
        assert set(kinds.tolist()) <= _names(band)
        assert len(set(kinds.tolist())) > 1


def test_unreachable_is_last_band():
    f = jnp.full((1, MAP, MAP), 32767, jnp.int16)
    w = {**WORLD, "dist_town": f}
    assert (np.asarray(ro_spawns.band_of(w, jnp.asarray([[5, 5], [9, 9]], jnp.int32))) == len(FIELD_BANDS) - 1).all()


def test_dungeons_and_mvp():
    kfn = jax.jit(ro_spawns.kinds_for)
    for k, d in enumerate(DUNGEONS):
        x0, y0, x1, y1 = world_layout.dungeon_boxes()[k].tolist()
        m = np.zeros((MAP, MAP), bool)
        m[x0:x1 + 1, y0:y1 + 1] = TER[x0:x1 + 1, y0:y1 + 1] == T_CAVE
        assert m.any()
        pos = _tiles(m, 400, k)
        kinds = np.asarray(kfn(WORLD, pos, jax.random.PRNGKey(5)))
        mvp = K[d[5]]
        allowed = {K[n] for n in d[4]}
        assert kinds[0] == mvp and (kinds == mvp).sum() == 1
        assert set(kinds[1:].tolist()) <= allowed


def test_jit_matches_eager_and_plan():
    pos = _tiles(WALK, 500, 9)
    key = jax.random.PRNGKey(3)
    a = ro_spawns.kinds_for(WORLD, pos, key)
    b = jax.jit(ro_spawns.kinds_for)(WORLD, pos, key)
    assert (np.asarray(a) == np.asarray(b)).all() and a.dtype == jnp.int32
    plan = np.asarray(jax.jit(ro_spawns.spawn_plan)(_terrain, _fields))
    assert plan.shape == (MAP, MAP)
    assert (plan[~WALK] == -1).all()
    assert plan[TER == T_CAVE].min() >= len(FIELD_BANDS)
    assert set(np.unique(plan[WALK & (plan < len(FIELD_BANDS))]).tolist()) <= set(range(len(FIELD_BANDS)))
    dun = np.asarray(world_layout.dungeon_index(jnp.stack(jnp.meshgrid(jnp.arange(MAP), jnp.arange(MAP), indexing="ij"), -1).reshape(-1, 2))).reshape(MAP, MAP)
    assert (plan[WALK & (dun >= 0)] == len(FIELD_BANDS) + dun[WALK & (dun >= 0)]).all()


def test_choose_kinds():
    pos = _tiles(WALK, 64, 2)
    m = WORLD["monsters"]
    w = {**WORLD, "monsters": {**m, "pos": m["pos"].at[:64].set(pos)}}
    out = ro_spawns.choose_kinds(w, jax.random.PRNGKey(0), 64)
    assert out.shape == (64,)
    assert (np.asarray(out) == np.asarray(ro_spawns.kinds_for(w, pos, jax.random.PRNGKey(0)))).all()
