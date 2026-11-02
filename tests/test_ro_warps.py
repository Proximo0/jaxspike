import jax
import jax.numpy as jnp

from game import map_gen
from game.ro_warps import apply_warps
from game.state import make_empty_world


def world_with(players, monsters=()):
    """players: list of (x, y, active, dead); monsters: list of (x, y)."""
    w = make_empty_world(jax.random.PRNGKey(0))
    p, m = w["players"], w["monsters"]
    pos, act, dead = p["pos"], p["active"], p["dead"]
    for i, (x, y, a, d) in enumerate(players):
        pos = pos.at[i].set(jnp.array([x, y], jnp.int32))
        act = act.at[i].set(a)
        dead = dead.at[i].set(d)
    mpos, mact = m["pos"], m["active"]
    for i, (x, y) in enumerate(monsters):
        mpos = mpos.at[i].set(jnp.array([x, y], jnp.int32))
        mact = mact.at[i].set(True)
    return {**w, "players": {**p, "pos": pos, "active": act, "dead": dead}, "monsters": {**m, "pos": mpos, "active": mact}}


def pos_of(w, i):
    return tuple(int(v) for v in w["players"]["pos"][i])


def tab(rows):
    s = jnp.array([r[0] for r in rows] + [(-1, -1)], jnp.int32)
    d = jnp.array([r[1] for r in rows] + [(-1, -1)], jnp.int32)
    return s, d


T = tab([((10, 10), (50, 50)), ((20, 20), (60, 60))])


def test_player_on_src_lands_on_dst():
    w = world_with([(10, 10, True, False), (20, 20, True, False)])
    out = apply_warps(w, T)
    assert pos_of(out, 0) == (50, 50)
    assert pos_of(out, 1) == (60, 60)


def test_real_table_first_row():
    src, dst = map_gen.warp_table()
    x, y = (int(v) for v in src[0])
    w = world_with([(x, y, True, False)])
    assert pos_of(apply_warps(w), 0) == tuple(int(v) for v in dst[0])


def test_next_to_warp_stays():
    w = world_with([(11, 10, True, False)])
    assert pos_of(apply_warps(w, T), 0) == (11, 10)


def test_monster_dead_inactive_do_not_warp():
    w = world_with([(10, 10, True, True), (20, 20, False, False)], monsters=[(10, 10)])
    out = apply_warps(w, T)
    assert pos_of(out, 0) == (10, 10) and pos_of(out, 1) == (20, 20)
    assert tuple(int(v) for v in out["monsters"]["pos"][0]) == (10, 10)


def test_occupied_destination_blocks():
    w = world_with([(10, 10, True, False), (50, 50, True, False)])
    assert pos_of(apply_warps(w, T), 0) == (10, 10)
    w = world_with([(10, 10, True, False)], monsters=[(50, 50)])
    assert pos_of(apply_warps(w, T), 0) == (10, 10)
    # a dead player or inactive monster does not block
    w = world_with([(10, 10, True, False), (50, 50, True, True)])
    assert pos_of(apply_warps(w, T), 0) == (50, 50)


def test_padding_rows_never_match():
    w = world_with([(0, 0, True, False), (-1, -1, True, False)])
    out = apply_warps(w, T)
    assert pos_of(out, 0) == (0, 0)
    assert pos_of(out, 1) == (-1, -1)
    # inactive default slots sit at (0, 0): nothing moves anywhere
    assert bool(jnp.all(out["players"]["pos"][2:] == 0))


def test_lowest_slot_wins_shared_dst():
    t = tab([((10, 10), (50, 50)), ((20, 20), (50, 50))])
    w = world_with([(20, 20, True, False), (10, 10, True, False)])
    out = apply_warps(w, t)
    assert pos_of(out, 0) == (50, 50)
    assert pos_of(out, 1) == (10, 10)


def test_jit_vmap_match_eager():
    w1 = world_with([(10, 10, True, False)])
    w2 = world_with([(20, 20, True, False), (60, 60, True, False)])
    eager = [apply_warps(w, T) for w in (w1, w2)]
    jitted = jax.jit(lambda w: apply_warps(w, T))(w1)
    assert bool(jnp.all(jitted["players"]["pos"] == eager[0]["players"]["pos"]))
    batch = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), w1, w2)
    out = jax.jit(jax.vmap(lambda w: apply_warps(w, T)))(batch)
    for i in range(2):
        assert bool(jnp.all(out["players"]["pos"][i] == eager[i]["players"]["pos"]))
