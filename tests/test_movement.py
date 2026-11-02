import jax
import jax.numpy as jnp

from game.config import MAX_PLAYERS
from game.movement import move_players
from game.state import make_empty_world

STAY, N, S, W, E = 0, 1, 2, 3, 4


def world_with(players):
    """players: list of (x, y, active, dead)."""
    w = make_empty_world(jax.random.PRNGKey(0))
    p = w["players"]
    pos, act, dead = p["pos"], p["active"], p["dead"]
    for i, (x, y, a, d) in enumerate(players):
        pos = pos.at[i].set(jnp.array([x, y], jnp.int32))
        act = act.at[i].set(a)
        dead = dead.at[i].set(d)
    return {**w, "players": {**p, "pos": pos, "active": act, "dead": dead}}


def moves(**kw):
    m = jnp.zeros((MAX_PLAYERS,), jnp.int32)
    for k, v in kw.items():
        m = m.at[int(k[1:])].set(v)
    return {"move": m, "attack": jnp.zeros((MAX_PLAYERS,), jnp.bool_)}


def pos_of(w, i):
    return tuple(int(v) for v in w["players"]["pos"][i])


def test_simple_move():
    w = world_with([(10, 10, True, False)])
    assert pos_of(move_players(w, moves(p0=E)), 0) == (11, 10)
    assert pos_of(move_players(w, moves(p0=N)), 0) == (10, 9)
    assert pos_of(move_players(w, moves(p0=STAY)), 0) == (10, 10)


def test_wall_blocks():
    w = world_with([(10, 10, True, False)])
    w = {**w, "walls": w["walls"].at[11, 10].set(1)}
    assert pos_of(move_players(w, moves(p0=E)), 0) == (10, 10)


def test_out_of_bounds_blocks():
    w = world_with([(0, 0, True, False)])
    assert pos_of(move_players(w, moves(p0=W)), 0) == (0, 0)
    assert pos_of(move_players(w, moves(p0=N)), 0) == (0, 0)


def test_monster_blocks_but_inactive_monster_does_not():
    w = world_with([(10, 10, True, False)])
    m = w["monsters"]
    m = {**m, "pos": m["pos"].at[0].set(jnp.array([11, 10])), "active": m["active"].at[0].set(True)}
    w = {**w, "monsters": m}
    assert pos_of(move_players(w, moves(p0=E)), 0) == (10, 10)
    m = {**m, "active": m["active"].at[0].set(False)}
    w = {**w, "monsters": m}
    assert pos_of(move_players(w, moves(p0=E)), 0) == (11, 10)


def test_same_target_lowest_index_wins():
    w = world_with([(9, 10, True, False), (11, 10, True, False)])
    out = move_players(w, moves(p0=E, p1=W))
    assert pos_of(out, 0) == (10, 10)
    assert pos_of(out, 1) == (11, 10)


def test_lowest_wins_when_higher_slot_listed_first_in_space():
    w = world_with([(5, 5, True, False), (5, 5, False, False), (7, 6, True, False)])
    # slot 0 at (5,5) wants S -> (5,6); slot 2 at (7,6) wants W twice isn't needed; use slot 2 -> (6,6)? keep distinct
    out = move_players(w, moves(p0=S, p2=W))
    assert pos_of(out, 0) == (5, 6)
    assert pos_of(out, 2) == (6, 6)


def test_cannot_enter_tile_of_player_moving_away():
    w = world_with([(10, 10, True, False), (11, 10, True, False)])
    out = move_players(w, moves(p0=E, p1=E))
    assert pos_of(out, 0) == (10, 10)
    assert pos_of(out, 1) == (12, 10)


def test_dead_and_inactive_never_move():
    w = world_with([(10, 10, True, True), (20, 20, False, False)])
    out = move_players(w, moves(p0=E, p1=E))
    assert pos_of(out, 0) == (10, 10)
    assert pos_of(out, 1) == (20, 20)


def test_living_player_may_step_onto_dead_player():
    # Dead players are not physical (they are waiting to respawn), so they never block a tile.
    w = world_with([(10, 10, True, False), (11, 10, True, True)])
    out = move_players(w, moves(p0=E))
    assert pos_of(out, 0) == (11, 10)


def test_jit_and_vmap_over_three_worlds():
    ws = [world_with([(10 + i, 10, True, False)]) for i in range(3)]
    batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *ws)
    inp = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[moves(p0=E)] * 3)
    single = jax.jit(move_players)(ws[0], moves(p0=E))
    assert pos_of(single, 0) == (11, 10)
    out = jax.jit(jax.vmap(move_players))(batch, inp)
    assert out["players"]["pos"].shape == (3, MAX_PLAYERS, 2)
    for i in range(3):
        assert tuple(int(v) for v in out["players"]["pos"][i, 0]) == (11 + i, 10)
