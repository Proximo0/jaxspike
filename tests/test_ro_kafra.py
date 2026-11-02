import jax
import jax.numpy as jnp
import numpy as np

from game import map_gen, ro_kafra, world_layout
from game.config import MAX_PLAYERS, T_TEMPLE
from game.state import empty_inputs, make_empty_world


def _world(slots, pos, dead=()):
    w = make_empty_world(jax.random.PRNGKey(0))
    p = w["players"]
    active = jnp.zeros((MAX_PLAYERS,), bool)
    posa = p["pos"]
    for s, xy in zip(slots, pos):
        active = active.at[s].set(True)
        posa = posa.at[s].set(jnp.array(xy, jnp.int32))
    d = p["dead"]
    for s in dead:
        d = d.at[s].set(True)
    return {**w, "players": {**p, "active": active, "pos": posa, "dead": d}}


def _inp(*slots):
    s = jnp.zeros((MAX_PLAYERS,), bool)
    for i in slots:
        s = s.at[i].set(True)
    return {**empty_inputs(), "save": s}


def _in_town(t):
    c = world_layout.town_centers()[t]
    return (int(c[0]) + 3, int(c[1]) - 2)


def test_save_in_town_sets_centre():
    for t in range(len(world_layout.TOWNS)):
        w = ro_kafra.phase(_world([0], [_in_town(t)]), _inp(0))
        assert (np.asarray(w["players"]["spawn"][0]) == world_layout.town_centers()[t]).all()


def test_save_outside_town_nothing():
    w0 = _world([0], [(0, 0)])
    w = ro_kafra.phase(w0, _inp(0))
    assert (np.asarray(w["players"]["spawn"]) == np.asarray(w0["players"]["spawn"])).all()
    assert ro_kafra.town_of_tile(jnp.array([[0, 0]], jnp.int32))[0] == -1


def test_no_save_input_nothing():
    w = ro_kafra.phase(_world([0], [_in_town(0)]), _inp())
    assert (np.asarray(w["players"]["spawn"]) == 0).all()


def test_dead_and_inactive_ignored():
    w0 = _world([0, 1], [_in_town(1), _in_town(2)], dead=[0])
    w = ro_kafra.phase(w0, _inp(0, 1, 2))  # slot 2 inactive
    sp = np.asarray(w["players"]["spawn"])
    assert (sp[0] == 0).all() and (sp[2] == 0).all()
    assert (sp[1] == world_layout.town_centers()[2]).all()


def test_spawn_is_temple_tile():
    terrain = np.asarray(map_gen.make_terrain(jax.random.PRNGKey(3)))
    for t in range(len(world_layout.TOWNS)):
        x, y = world_layout.town_centers()[t]
        assert terrain[x, y] == T_TEMPLE


def test_jit():
    w = jax.jit(ro_kafra.phase)(_world([0], [_in_town(3)]), _inp(0))
    assert (np.asarray(w["players"]["spawn"][0]) == world_layout.town_centers()[3]).all()


def test_vmap():
    ws = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), _world([0], [_in_town(0)]), _world([0], [(0, 0)]))
    ins = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), _inp(0))
    out = jax.vmap(ro_kafra.phase)(ws, ins)
    sp = np.asarray(out["players"]["spawn"])
    assert (sp[0, 0] == world_layout.town_centers()[0]).all() and (sp[1, 0] == 0).all()
