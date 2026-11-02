import jax
import jax.numpy as jnp

from game.config import MAP, MAX_PLAYERS, T_GRASS
from game.state import empty_inputs, make_empty_world


def test_empty_world_has_terrain_and_pz():
    w = make_empty_world(jax.random.PRNGKey(0))
    assert w["terrain"].shape == (MAP, MAP) and w["terrain"].dtype == jnp.int8
    assert w["pz"].shape == (MAP, MAP) and w["pz"].dtype == jnp.bool_
    assert int(w["terrain"].max()) == T_GRASS and not bool(w["pz"].any())


def test_empty_inputs_has_eat():
    i = empty_inputs()
    assert i["eat"].shape == (MAX_PLAYERS,) and i["eat"].dtype == jnp.bool_ and not bool(i["eat"].any())
