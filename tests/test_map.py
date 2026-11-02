import jax
import jax.numpy as jnp
import numpy as np

from game.config import MAP
from game.map_gen import make_walls, walkable


def test_border_all_walls():
    w = np.asarray(make_walls(jax.random.PRNGKey(0)))
    assert w.shape == (MAP, MAP) and w.dtype == np.int8
    assert (w[0] == 1).all() and (w[-1] == 1).all()
    assert (w[:, 0] == 1).all() and (w[:, -1] == 1).all()


def test_spawn_square_clear():
    for s in range(3):
        w = np.asarray(make_walls(jax.random.PRNGKey(s)))
        assert (w[16:24, 16:24] == 0).all()


def test_density_reasonable():
    w = np.asarray(make_walls(jax.random.PRNGKey(1)))[1:-1, 1:-1]
    assert 0.10 < w.mean() < 0.20


def test_walkable_walls_and_oob():
    w = make_walls(jax.random.PRNGKey(0))
    pos = jnp.array([[0, 0], [20, 20], [-1, 5], [5, -1], [MAP, 5], [5, MAP]], jnp.int32)
    out = np.asarray(walkable(w, pos))
    assert not out[0] and out[1] and not out[2:].any()
    wn = np.asarray(w)
    xs, ys = np.nonzero(wn == 1)
    p = jnp.array(np.stack([xs[:50], ys[:50]], 1), jnp.int32)
    assert not np.asarray(walkable(w, p)).any()


def test_jit_and_vmap():
    keys = jax.random.split(jax.random.PRNGKey(2), 4)
    ws = jax.vmap(make_walls)(keys)
    pos = jnp.array([[20, 20], [0, 0], [-3, 4]], jnp.int32)
    a = jax.jit(walkable)(ws[0], pos)
    assert np.asarray(a).tolist() == [True, False, False]
    b = jax.jit(jax.vmap(walkable, in_axes=(0, None)))(ws, pos)
    assert b.shape == (4, 3)
    assert np.asarray(b).tolist() == [[True, False, False]] * 4


def test_deterministic():
    k = jax.random.PRNGKey(7)
    assert np.array_equal(np.asarray(make_walls(k)), np.asarray(make_walls(k)))
    assert not np.array_equal(np.asarray(make_walls(k)),
                              np.asarray(make_walls(jax.random.PRNGKey(8))))
