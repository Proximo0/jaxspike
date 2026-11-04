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


# ---- spawn density and kind mix (ro3-balance) -------------------------------------------------------------------
_WALLS_WORLD = {**WORLD, "walls": map_gen.blocking(_terrain), "pz": map_gen.protection_zone(_terrain)}


def test_tile_weights_follow_bands_and_exclusions():
    tw = np.asarray(jax.jit(ro_spawns.tile_weights)(_WALLS_WORLD))
    assert tw.shape == (MAP, MAP) and tw.dtype == np.float32
    assert (tw[~WALK] == 0).all()
    assert (tw[GATE <= 4] == 0).all()                       # GATE_APRON_TILES
    plan = np.asarray(ro_spawns.spawn_plan(_terrain, _fields))
    for b, w in enumerate(ro_spawns.BAND_DENSITY):
        sel = (plan == b) & (tw > 0)
        assert sel.any() and np.allclose(tw[sel], w)
    dun = (plan >= len(FIELD_BANDS)) & (tw > 0)
    assert dun.any() and np.allclose(tw[dun], ro_spawns.DUNGEON_DENSITY)


def test_spawn_tiles_distinct_weighted_and_jittable():
    n = 1024
    pos = np.asarray(jax.jit(ro_spawns.spawn_tiles, static_argnums=2)(_WALLS_WORLD, jax.random.PRNGKey(7), n))
    eager = np.asarray(ro_spawns.spawn_tiles(_WALLS_WORLD, jax.random.PRNGKey(7), n))
    np.testing.assert_array_equal(pos, eager)
    assert pos.shape == (n, 2) and pos.dtype == np.int32
    assert len({tuple(t) for t in pos.tolist()}) == n
    tw = np.asarray(ro_spawns.tile_weights(_WALLS_WORLD))
    assert (tw[pos[:, 0], pos[:, 1]] > 0).all()
    band = np.asarray(ro_spawns.band_of(_WALLS_WORLD, jnp.asarray(pos)))
    share = np.bincount(band, minlength=len(FIELD_BANDS)) / n
    tiles = np.array([((tw > 0) & (np.asarray(ro_spawns.spawn_plan(_terrain, _fields)) == b)).sum()
                      for b in range(len(FIELD_BANDS))], float)
    uniform = tiles / tiles.sum()
    # band 0 is at least as dense as band 1 (ro6-balance: 2.5 / 2.5, band 1 is where level 2+ bots hunt with the
    # starter kit) and both are far denser than the outer field
    assert ro_spawns.BAND_DENSITY[0] >= ro_spawns.BAND_DENSITY[1] > 2 * ro_spawns.BAND_DENSITY[2]
    assert share[0] > 1.5 * uniform[0] and share[2] < uniform[2]


def test_kind_mix_weights():
    plan = np.asarray(ro_spawns.spawn_plan(_terrain, _fields))
    for band in (0, 1):
        sel = np.argwhere(WALK & (plan == band))
        half = np.asarray(ro_spawns.half_of(WORLD, jnp.asarray(sel, jnp.int32)))
        for s in range(ro_spawns.N_SUB):
            xy = sel[half == s]
            assert len(xy) > 500, (band, s, len(xy))
            pos = jnp.asarray(xy[np.random.default_rng(11).choice(len(xy), min(4000, len(xy)), replace=False)])
            kinds = np.asarray(ro_spawns.kinds_for(WORLD, pos, jax.random.PRNGKey(2 + s)))
            count = np.bincount(kinds, minlength=len(K))
            mix = ro_spawns.SPAWN_MIX[band][s]
            total = sum(mix.values())
            for name in FIELD_BANDS[band][2]:
                want = mix.get(name, 0) / total
                assert abs(count[K[name]] / len(pos) - want) < 0.04, (band, s, name, count[K[name]] / len(pos), want)


def test_mix_steps_up_with_distance_and_novice_prey_is_hittable():
    """The inner half of band 0 is mostly prey a fresh Novice (HIT 2) hits 80%+ of the time and that cannot one-shot
    40 hp; the level 14+ band-1 kinds only spawn in band 1's outer half, and the only aggressive kind in its inner
    half is the level-6 Pilferbug at no more than a quarter of the mix (ro6-balance: with the starter Knife it is what
    keeps deaths at 0.3-0.6 per 1000 ticks)."""
    from game.data import monsters as md
    inner0 = ro_spawns.SPAWN_MIX[0][0]
    hit80 = sum(w for n, w in inner0.items() if 80 + 2 - (int(md.LEVEL[K[n]]) + int(md.AGI[K[n]])) >= 80)
    assert hit80 / sum(inner0.values()) >= 0.5
    for band0 in ro_spawns.SPAWN_MIX[0]:
        assert all(int(md.ATK_MAX[K[n]]) < 40 for n in band0)
    inner1 = ro_spawns.SPAWN_MIX[1][0]
    for n in inner1:
        assert int(md.LEVEL[K[n]]) < 14 and (not bool(md.AGGRO[K[n]]) or n == "Pilferbug")
    assert inner1.get("Pilferbug", 0) * 4 <= sum(inner1.values())
    level = lambda mix: sum(int(md.LEVEL[K[n]]) * w for n, w in mix.items()) / sum(mix.values())
    rows = [m for band in ro_spawns.SPAWN_MIX for m in band]
    assert all(level(a) < level(b) for a, b in zip(rows, rows[1:]))


def test_repopulate_sets_the_first_n_slots():
    n = 256
    w = ro_spawns.repopulate(_WALLS_WORLD, jax.random.PRNGKey(3), n)
    m = w["monsters"]
    assert int(m["active"][:n].sum()) == n and not bool(m["active"][n:].any())
    from game.data.monsters import HP
    np.testing.assert_array_equal(np.asarray(m["hp"][:n]), np.asarray(HP)[np.asarray(m["kind"][:n])])
    np.testing.assert_array_equal(np.asarray(m["spawn"][:n]), np.asarray(m["pos"][:n]))
