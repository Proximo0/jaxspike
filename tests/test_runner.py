import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bots import runner
from game import persistence
from game.config import (
    GATE_APRON_TILES,
    MAP,
    MAX_MONSTERS,
    MAX_PLAYERS,
    MONSTER_HP,
    MONSTER_KIND_ORC,
    MONSTER_KIND_RAT,
    T_TEMPLE,
    T_WATER,
)
from game.map_gen import blocking, distance_fields, make_terrain, protection_zone
from game.state import make_empty_world
from game.world_layout import RAT_RING, TOWNS, temple_tiles


def test_persistence_roundtrip_exact(tmp_path):
    w = make_empty_world(jax.random.PRNGKey(7))
    w["players"]["hp"] = w["players"]["hp"].at[3].set(42)
    w["players"]["active"] = w["players"]["active"].at[3].set(True)
    w["monsters"]["pos"] = w["monsters"]["pos"].at[5].set(jnp.array([9, 11]))
    w["walls"] = w["walls"].at[1, 2].set(1)
    w["terrain"] = w["terrain"].at[3, 4].set(T_WATER)
    w["pz"] = w["pz"].at[5, 6].set(True)
    w["tick"] = jnp.int32(12)
    path = tmp_path / "w.npz"
    persistence.save(w, path)
    r = persistence.load(path)
    fw, fr = jax.tree_util.tree_flatten_with_path(w)[0], jax.tree_util.tree_flatten_with_path(r)[0]
    assert [p for p, _ in fw] == [p for p, _ in fr]
    for (_, a), (_, b) in zip(fw, fr):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert r["rng"].dtype == jnp.uint32


def test_spawn_population_counts():
    w = make_empty_world(jax.random.PRNGKey(0))
    terrain = make_terrain(jax.random.PRNGKey(2))
    w = {**w, "terrain": terrain, "walls": blocking(terrain), "pz": protection_zone(terrain),
         "dist_town": distance_fields(terrain)}
    w = runner.spawn_population(w, jax.random.PRNGKey(1))
    assert int(w["players"]["active"].sum()) == min(128, MAX_PLAYERS)
    assert int(w["monsters"]["active"].sum()) == runner.POP_MONSTERS
    for e in ("players", "monsters"):
        pos = np.asarray(w[e]["pos"])[np.asarray(w[e]["active"])]
        assert (np.asarray(w["walls"])[pos[:, 0], pos[:, 1]] == 0).all()
        np.testing.assert_array_equal(pos, np.asarray(w[e]["spawn"])[np.asarray(w[e]["active"])])
    mpos = np.asarray(w["monsters"]["pos"])[np.asarray(w["monsters"]["active"])]
    gate_dist = np.asarray(w["dist_town"]).min(axis=0)[mpos[:, 0], mpos[:, 1]]
    assert (gate_dist > GATE_APRON_TILES).all()
    ppos = np.asarray(w["players"]["pos"])[np.asarray(w["players"]["active"])]
    assert (np.asarray(terrain)[ppos[:, 0], ppos[:, 1]] == T_TEMPLE).all()
    m = w["monsters"]
    kind = np.asarray(m["kind"])[np.asarray(m["active"])]
    assert set(kind.tolist()) <= set(range(len(MONSTER_HP)))  # RO kinds by band (game/ro_spawns.py)
    np.testing.assert_array_equal(np.asarray(m["hp"])[np.asarray(m["active"])], np.asarray(MONSTER_HP)[kind])


def _one_world():
    return jax.tree_util.tree_map(lambda x: x[0], runner.make_world_batch(0, 1))


def test_players_spawn_on_home_temple():
    w = _one_world()
    p = w["players"]
    act = np.asarray(p["active"])
    tiles = np.asarray(temple_tiles())
    nt = len(TOWNS)
    expect = np.stack([tiles[i % nt, i // nt] for i in range(int(act.sum()))])
    np.testing.assert_array_equal(np.asarray(p["spawn"])[act], expect)
    np.testing.assert_array_equal(np.asarray(p["pos"])[act], expect)
    assert len({tuple(t) for t in expect.tolist()}) == len(expect)


def test_monsters_spawn_off_pz_on_walkable():
    w = _one_world()
    m = w["monsters"]
    pos = np.asarray(m["pos"])[np.asarray(m["active"])]
    assert (np.asarray(w["walls"])[pos[:, 0], pos[:, 1]] == 0).all()
    assert not np.asarray(w["pz"])[pos[:, 0], pos[:, 1]].any()


def test_monster_kind_follows_zone():
    """Kinds follow the RO field bands: monsters in band 0 (nearest the gates) are band-0 kinds, and every
    monster outside a dungeon carries a kind from its band's list (game/ro_spawns.py)."""
    from game import ro_spawns
    from game.world_layout import FIELD_BANDS, dungeon_index

    w = _one_world()
    m = w["monsters"]
    act = np.asarray(m["active"])
    pos = np.asarray(m["pos"])[act]
    kind = np.asarray(m["kind"])[act]
    dist = np.asarray(w["dist_town"]).min(axis=0)[pos[:, 0], pos[:, 1]]
    in_dungeon = np.asarray(dungeon_index(jnp.asarray(pos))) >= 0
    band_kinds = np.asarray(ro_spawns.BAND_KINDS)
    band_count = np.asarray(ro_spawns.BAND_COUNT)
    edges = [float(b[1]) * MAP for b in FIELD_BANDS[:-1]]
    band = np.zeros(len(dist), np.int32)
    for e in edges:
        band += dist >= e
    for b in range(len(FIELD_BANDS)):
        sel = (band == b) & ~in_dungeon
        allowed = set(band_kinds[b, : band_count[b]].tolist())
        assert set(kind[sel].tolist()) <= allowed, (b, set(kind[sel].tolist()) - allowed)
    assert (band == 0).sum() > 0 and len(set(kind.tolist())) >= 3


def test_world_batch_has_terrain_walls_pz():
    ws = runner.make_world_batch(0, 2)
    assert ws["terrain"].shape == ws["walls"].shape == ws["pz"].shape == (2, MAP, MAP)
    for b in range(2):
        np.testing.assert_array_equal(np.asarray(ws["walls"][b]), np.asarray(blocking(ws["terrain"][b])))
        np.testing.assert_array_equal(np.asarray(ws["pz"][b]), np.asarray(protection_zone(ws["terrain"][b])))


def _modules_exist():
    try:
        import game.map_gen, game.step, bots.policies  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _modules_exist(), reason="map_gen/step/policies not available yet")
def test_run_games_stats_shape():
    ws = runner.make_world_batch(0, 2)
    final, stats = runner.run_games(ws, 3)
    assert set(stats) == {"kills", "deaths", "alive", "items"}
    for v in stats.values():
        assert v.shape == (2,) and v.dtype == jnp.int32
    assert np.asarray(final["tick"]).tolist() == [3, 3]
