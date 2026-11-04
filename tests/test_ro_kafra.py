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


# ---- kafra teleport (round 3, ro3-world3) ------------------------------------------------------------------------
def _tp(pairs):
    """pairs: {slot: destination town}; everyone else -1."""
    t = jnp.full((MAX_PLAYERS,), -1, jnp.int32)
    for s, d in pairs.items():
        t = t.at[s].set(d)
    return {**empty_inputs(), "teleport": t}


def _at_kafra(t, dx=1, dy=0):
    k = world_layout.kafra_tiles()[t]
    return (int(k[0]) + dx, int(k[1]) + dy)


def _with_zeny(w, slot, z):
    items = w["items"]
    return {**w, "items": {**items, "zeny": items["zeny"].at[slot].set(z)}}


def test_teleport_moves_to_station_and_charges():
    w0 = _with_zeny(_world([0], [_at_kafra(0)]), 0, 5000)
    w = jax.jit(ro_kafra.phase)(w0, _tp({0: 1}))      # Prontera -> Geffen, 1200 z
    assert (np.asarray(w["players"]["pos"][0]) == world_layout.kafra_teleport_dest()[1]).all()
    assert int(w["items"]["zeny"][0]) == 5000 - 1200
    assert (np.asarray(w["players"]["spawn"][0]) == 0).all()   # teleport is not a save


def test_teleport_needs_route_zeny_and_kafra():
    # no route Geffen -> Alberta
    w0 = _with_zeny(_world([0], [_at_kafra(1)]), 0, 5000)
    w = ro_kafra.phase(w0, _tp({0: 3}))
    assert (np.asarray(w["players"]["pos"]) == np.asarray(w0["players"]["pos"])).all() and int(w["items"]["zeny"][0]) == 5000
    # not enough zeny
    w0 = _with_zeny(_world([0], [_at_kafra(0)]), 0, 1199)
    w = ro_kafra.phase(w0, _tp({0: 1}))
    assert (np.asarray(w["players"]["pos"]) == np.asarray(w0["players"]["pos"])).all() and int(w["items"]["zeny"][0]) == 1199
    # in town but not at the kafra
    w0 = _with_zeny(_world([0], [_in_town(0)]), 0, 5000)
    w = ro_kafra.phase(w0, _tp({0: 1}))
    assert (np.asarray(w["players"]["pos"]) == np.asarray(w0["players"]["pos"])).all() and int(w["items"]["zeny"][0]) == 5000
    # same town, bad index, dead
    w0 = _with_zeny(_with_zeny(_world([0, 1, 2], [_at_kafra(0), _at_kafra(0), _at_kafra(0)], dead=[2]), 0, 5000), 1, 5000)
    w0 = _with_zeny(w0, 2, 5000)
    w = ro_kafra.phase(w0, _tp({0: 0, 1: 7, 2: 1}))
    assert (np.asarray(w["players"]["pos"]) == np.asarray(w0["players"]["pos"])).all()
    assert (np.asarray(w["items"]["zeny"][:3]) == 5000).all()


def test_teleport_station_occupied_and_lowest_slot_wins():
    dest = tuple(int(v) for v in world_layout.kafra_teleport_dest()[2])
    # a living player on Payon's station blocks; a dead one does not
    w0 = _with_zeny(_world([0, 1], [_at_kafra(0), dest]), 0, 5000)
    w = ro_kafra.phase(w0, _tp({0: 2}))
    assert tuple(int(v) for v in w["players"]["pos"][0]) == _at_kafra(0) and int(w["items"]["zeny"][0]) == 5000
    w0 = _with_zeny(_world([0, 1], [_at_kafra(0), dest], dead=[1]), 0, 5000)
    w = ro_kafra.phase(w0, _tp({0: 2}))
    assert tuple(int(v) for v in w["players"]["pos"][0]) == dest
    # two players want Payon's station the same tick: slot 1 (lower) wins, slot 3 keeps its zeny and retries
    w0 = _with_zeny(_with_zeny(_world([3, 1], [_at_kafra(0), _at_kafra(1, 0, 1)]), 3, 5000), 1, 5000)
    w = ro_kafra.phase(w0, _tp({3: 2, 1: 2}))
    assert tuple(int(v) for v in w["players"]["pos"][1]) == dest and int(w["items"]["zeny"][1]) == 3800
    assert tuple(int(v) for v in w["players"]["pos"][3]) == _at_kafra(0) and int(w["items"]["zeny"][3]) == 5000
    # a monster on the station blocks too
    m = w0["monsters"]
    w0m = {**w0, "monsters": {**m, "pos": m["pos"].at[0].set(jnp.array(dest, jnp.int32)), "active": m["active"].at[0].set(True)}}
    w = ro_kafra.phase(w0m, _tp({1: 2}))
    assert int(w["items"]["zeny"][1]) == 5000


def test_teleport_input_absent_is_noop_and_vmap():
    w0 = _with_zeny(_world([0], [_at_kafra(0)]), 0, 5000)
    w = jax.jit(ro_kafra.phase)(w0, {**empty_inputs()})  # no "teleport" key: save-only behaviour
    assert (np.asarray(w["players"]["pos"]) == np.asarray(w0["players"]["pos"])).all()
    ws = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), w0, _with_zeny(_world([0], [_at_kafra(4)]), 0, 100))
    ins = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), _tp({0: 1}))
    out = jax.jit(jax.vmap(ro_kafra.phase))(ws, ins)
    assert (np.asarray(out["players"]["pos"][0, 0]) == world_layout.kafra_teleport_dest()[1]).all()
    assert tuple(int(v) for v in out["players"]["pos"][1, 0]) == _at_kafra(4) and int(out["items"]["zeny"][1, 0]) == 100


def test_teleport_table_matches_doc():
    c = world_layout.kafra_teleport_cost()
    assert c[0, 3] == 1800 and c[1, 3] == world_layout.KAFRA_NO_ROUTE and (np.diag(c) == -1).all()
    assert (world_layout.kafra_teleport_dest() == world_layout.town_station_tiles()).all()
