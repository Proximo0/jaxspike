import jax
import jax.numpy as jnp

from bots.policies import bot_inputs
from game.config import MAX_PLAYERS
from game.state import make_empty_world


def _world(player_pos, monster_pos=None):
    w = make_empty_world(jax.random.PRNGKey(0))
    p = w["players"]
    p = {**p, "pos": p["pos"].at[0].set(jnp.array(player_pos, jnp.int32)),
         "active": p["active"].at[0].set(True)}
    m = w["monsters"]
    if monster_pos is not None:
        m = {**m, "pos": m["pos"].at[0].set(jnp.array(monster_pos, jnp.int32)),
             "active": m["active"].at[0].set(True)}
    return {**w, "players": p, "monsters": m}


KEY = jax.random.PRNGKey(1)


def test_steps_toward_monster():
    cases = [((50, 50), (50, 53), 2), ((50, 50), (50, 47), 1),
             ((50, 50), (47, 50), 3), ((50, 50), (53, 50), 4),
             ((50, 50), (54, 52), 4), ((50, 50), (51, 55), 2)]
    for p, m, code in cases:
        out = bot_inputs(_world(p, m), KEY)
        assert int(out["move"][0]) == code
        assert not bool(out["attack"][0])


def test_attacks_when_adjacent():
    out = bot_inputs(_world((50, 50), (51, 50)), KEY)
    assert bool(out["attack"][0])
    assert int(out["move"][0]) == 4


def test_out_of_range_monster_ignored():
    out = bot_inputs(_world((50, 50), (50, 57)), KEY)
    assert not bool(out["attack"][0])


def test_random_wander():
    w = make_empty_world(jax.random.PRNGKey(0))
    w = {**w, "players": {**w["players"], "active": jnp.ones((MAX_PLAYERS,), bool)}}
    out = bot_inputs(w, KEY)
    mv = out["move"]
    assert bool(jnp.all((mv >= 0) & (mv <= 4)))
    assert int(jnp.count_nonzero(mv)) > 0
    assert len(set(mv.tolist())) >= 3
    assert not bool(out["attack"].any())


def test_dead_and_inactive_noop():
    w = _world((50, 50), (51, 50))
    p = w["players"]
    p = {**p, "active": jnp.ones((MAX_PLAYERS,), bool),
         "dead": jnp.zeros((MAX_PLAYERS,), bool).at[0].set(True)}
    out = bot_inputs({**w, "players": p}, KEY)
    assert int(out["move"][0]) == 0 and not bool(out["attack"][0])
    p = {**p, "dead": jnp.zeros((MAX_PLAYERS,), bool),
         "active": jnp.ones((MAX_PLAYERS,), bool).at[0].set(False)}
    out = bot_inputs({**w, "players": p}, KEY)
    assert int(out["move"][0]) == 0 and not bool(out["attack"][0])
    # inactive players everywhere: all zero
    out = bot_inputs(make_empty_world(jax.random.PRNGKey(0)), KEY)
    assert not bool(out["move"].any()) and not bool(out["attack"].any())


def test_shapes_dtypes():
    out = bot_inputs(_world((5, 5)), KEY)
    assert out["move"].shape == (MAX_PLAYERS,) and out["move"].dtype == jnp.int32
    assert out["attack"].shape == (MAX_PLAYERS,) and out["attack"].dtype == jnp.bool_


def test_jit_and_vmap():
    worlds = [_world((50, 50), (50, 53)), _world((50, 50), (51, 50))]
    batch = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *worlds)
    keys = jax.random.split(KEY, 2)
    out = jax.jit(jax.vmap(bot_inputs))(batch, keys)
    assert out["move"].shape == (2, MAX_PLAYERS)
    assert int(out["move"][0, 0]) == 2 and not bool(out["attack"][0, 0])
    assert int(out["move"][1, 0]) == 4 and bool(out["attack"][1, 0])
    single = jax.jit(bot_inputs)(worlds[0], keys[0])
    assert int(single["move"][0]) == 2
