import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bots import runner
from game import persistence
from game.config import MAX_MONSTERS, MAX_PLAYERS, MONSTER_HP
from game.state import make_empty_world


def test_persistence_roundtrip_exact(tmp_path):
    w = make_empty_world(jax.random.PRNGKey(7))
    w["players"]["hp"] = w["players"]["hp"].at[3].set(42)
    w["players"]["active"] = w["players"]["active"].at[3].set(True)
    w["monsters"]["pos"] = w["monsters"]["pos"].at[5].set(jnp.array([9, 11]))
    w["walls"] = w["walls"].at[1, 2].set(1)
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
    w["walls"] = w["walls"].at[0, :].set(1).at[:, 0].set(1)
    w = runner.spawn_population(w, jax.random.PRNGKey(1))
    assert int(w["players"]["active"].sum()) == min(128, MAX_PLAYERS)
    assert int(w["monsters"]["active"].sum()) == min(512, MAX_MONSTERS)
    for e in ("players", "monsters"):
        pos = np.asarray(w[e]["pos"])[np.asarray(w[e]["active"])]
        assert (np.asarray(w["walls"])[pos[:, 0], pos[:, 1]] == 0).all()
        np.testing.assert_array_equal(pos, np.asarray(w[e]["spawn"])[np.asarray(w[e]["active"])])
    m = w["monsters"]
    kind = np.asarray(m["kind"])[np.asarray(m["active"])]
    assert set(kind.tolist()) <= {0, 1}
    np.testing.assert_array_equal(np.asarray(m["hp"])[np.asarray(m["active"])], np.asarray(MONSTER_HP)[kind])


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
