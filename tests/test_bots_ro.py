"""Tests for bots/ro_policies.py: the composed RO bot policy fills every engine input with the right shape."""
import jax
import jax.numpy as jnp

from bots import ro_policies
from bots.ro_craft_policy import _IS_GATHERER
from game.config import MAX_PLAYERS
from game.state import empty_inputs, make_empty_world

KEY = jax.random.PRNGKey(0)


def _world():
    w = make_empty_world(KEY)
    p = w["players"]
    n = 12
    pos = jnp.stack([100 + 2 * jnp.arange(n), jnp.full((n,), 100)], axis=1).astype(jnp.int32)
    p = {**p, "pos": p["pos"].at[:n].set(pos), "active": p["active"].at[:n].set(True)}
    return {**w, "players": p}


def test_every_engine_input_present_with_its_dtype_and_shape():
    out = jax.jit(ro_policies.ro_bot_inputs)(_world(), KEY)
    ref = empty_inputs()
    for k, v in ref.items():
        assert k in out, k
        assert out[k].shape == v.shape and out[k].dtype == v.dtype, k
    assert set(out) <= set(ref)


def test_vmap_over_worlds_and_one_step():
    from game.step import step
    ws = jax.tree.map(lambda a: jnp.stack([a, a]), _world())
    inp = jax.jit(jax.vmap(ro_policies.ro_bot_inputs))(ws, jax.random.split(KEY, 2))
    assert inp["move"].shape == (2, MAX_PLAYERS)
    out = jax.jit(jax.vmap(step))(ws, inp)
    assert out["tick"].shape == (2,)


def test_inactive_slots_stay_idle_and_idle_gatherers_take_the_nav_move():
    w = _world()
    out = jax.jit(ro_policies.ro_bot_inputs)(w, KEY)
    assert int(out["move"][MAX_PLAYERS - 1]) == 0 and int(out["skill"][MAX_PLAYERS - 1]) == 0
    assert "craft_busy" not in out
    from bots.ro_nav_policy import nav_moves
    from bots.ro_craft_policy import craft_bot_inputs
    k_craft, k_nav = jax.random.split(KEY, 5)[:2]
    base = jax.jit(craft_bot_inputs)(w, k_craft)
    assert not bool(base["craft_busy"].any())  # an empty world has no node fields: gatherers are not busy
    nav, _, _, _ = jax.jit(nav_moves)(w, k_nav, base["move"], None)
    g = jnp.nonzero(_IS_GATHERER[:12])[0]
    assert bool((out["move"][g] == nav[g]).all())
    assert not bool(out["gather"].any())
