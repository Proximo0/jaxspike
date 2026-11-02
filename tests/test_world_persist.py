import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from game import world_persist as wp
from game.config import INV_SLOTS, MAP, MAX_DROPS, MAX_MONSTERS, MAX_PLAYERS
from game.persistence import _flatten
from game.state import make_empty_world
from game.world_layout import TOWNS


def _world():
    w = make_empty_world(jax.random.PRNGKey(3))
    w["players"]["hp"] = w["players"]["hp"].at[3].set(42)
    w["players"]["active"] = w["players"]["active"].at[3].set(True)
    w["monsters"]["respawn_in"] = w["monsters"]["respawn_in"].at[5].set(77)
    w["monsters"]["pos"] = w["monsters"]["pos"].at[5].set(jnp.array([9, 11]))
    w["walls"] = w["walls"].at[1, 2].set(1)
    w["tick"] = jnp.int32(12345)
    return w


def _raw(w, version=wp.WORLD_FORMAT_VERSION, layout=None):
    flat = _flatten(w)
    flat["__version__"] = np.asarray(version, np.int32)
    flat["__layout__"] = np.array(
        [MAP, MAX_PLAYERS, MAX_MONSTERS, MAX_DROPS, INV_SLOTS, len(TOWNS)] if layout is None else layout, np.int32)
    return flat


def _write(path, flat):
    with open(path, "wb") as f:
        np.savez(f, **flat)


def _leaves(t):
    return jax.tree_util.tree_flatten_with_path(t)[0]


def test_roundtrip_exact(tmp_path):
    w = _world()
    path = tmp_path / "world.npz"
    wp.save_world(w, path)
    r = wp.load_world(path)
    fw, fr = _leaves(w), _leaves(r)
    assert [p for p, _ in fw] == [p for p, _ in fr]
    for (_, a), (_, b) in zip(fw, fr):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert r["rng"].dtype == jnp.uint32
    assert int(r["monsters"]["respawn_in"][5]) == 77
    assert "__version__" not in r and "__layout__" not in r
    assert wp.world_matches(r, make_empty_world(None)) is None


def test_no_tmp_left_behind(tmp_path):
    wp.save_world(_world(), tmp_path / "world.npz")
    assert sorted(os.listdir(tmp_path)) == ["world.npz"]


def test_version_mismatch_refused(tmp_path):
    path = tmp_path / "w.npz"
    _write(path, _raw(_world(), version=99))
    with pytest.raises(wp.WorldFileError, match="version 99"):
        wp.load_world(path)


def test_shape_mismatch_names_key(tmp_path):
    flat = _raw(_world())
    flat["players/hp"] = np.zeros(MAX_PLAYERS + 1, np.int32)
    path = tmp_path / "w.npz"
    _write(path, flat)
    with pytest.raises(wp.WorldFileError, match="players/hp"):
        wp.load_world(path)


def test_dtype_mismatch_names_key(tmp_path):
    flat = _raw(_world())
    flat["players/hp"] = flat["players/hp"].astype(np.float32)
    path = tmp_path / "w.npz"
    _write(path, flat)
    with pytest.raises(wp.WorldFileError, match="players/hp"):
        wp.load_world(path)


def test_missing_key_refused(tmp_path):
    flat = _raw(_world())
    del flat["players/hp"]
    path = tmp_path / "w.npz"
    _write(path, flat)
    with pytest.raises(wp.WorldFileError, match="players/hp"):
        wp.load_world(path)


def test_layout_mismatch_names_map(tmp_path):
    path = tmp_path / "w.npz"
    _write(path, _raw(_world(), layout=[MAP + 1, MAX_PLAYERS, MAX_MONSTERS, MAX_DROPS, INV_SLOTS, len(TOWNS)]))
    with pytest.raises(wp.WorldFileError, match=f"MAP {MAP + 1} != {MAP}"):
        wp.load_world(path)


def test_garbage_file_refused(tmp_path):
    path = tmp_path / "w.npz"
    path.write_bytes(b"this is not an npz file")
    with pytest.raises(wp.WorldFileError):
        wp.load_world(path)
    wp.save_world(_world(), path)
    path.write_bytes(path.read_bytes()[:200])
    with pytest.raises(wp.WorldFileError):
        wp.load_world(path)
    with pytest.raises(wp.WorldFileError):
        wp.load_world(tmp_path / "missing.npz")


def test_extra_key_survives(tmp_path):
    w = _world()
    w["extra"] = {"str": jnp.arange(MAX_PLAYERS, dtype=jnp.int16)}
    path = tmp_path / "w.npz"
    wp.save_world(w, path)
    r = wp.load_world(path)
    assert r["extra"]["str"].dtype == jnp.int16
    np.testing.assert_array_equal(np.asarray(r["extra"]["str"]), np.arange(MAX_PLAYERS))


def test_world_matches_messages():
    t = make_empty_world(None)
    assert wp.world_matches(make_empty_world(jax.random.PRNGKey(1)), t) is None
    bad = {**t, "tick": jnp.zeros((1,), jnp.int32)}
    msg = wp.world_matches(bad, t)
    assert "tick" in msg and "(1,)" in msg and "()" in msg
