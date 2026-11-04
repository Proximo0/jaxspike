"""8-direction movement with speed accumulators (game/ro_move.py, docs/round2_contract.md 1.1)."""
import jax
import jax.numpy as jnp
import pytest

from game import ro_move
from game.config import MAX_MONSTERS, MAX_PLAYERS, TICK_MS
from game.data.monsters import KIND_BY_NAME
from game.movement import move_players
from game.state import make_empty_world

S, SW, W, NW, N, NE, E, SE = range(1, 9)
_move = jax.jit(move_players)


def world_with(players, monsters=(), kind=1, block=True):
    """players / monsters: lists of (x, y). The move block is attached unless block=False (the legacy path)."""
    w = make_empty_world(jax.random.PRNGKey(0))
    p, m = dict(w["players"]), dict(w["monsters"])
    for i, xy in enumerate(players):
        p["pos"] = p["pos"].at[i].set(jnp.array(xy, jnp.int32))
        p["active"] = p["active"].at[i].set(True)
    for i, xy in enumerate(monsters):
        m["pos"] = m["pos"].at[i].set(jnp.array(xy, jnp.int32))
        m["spawn"] = m["spawn"].at[i].set(jnp.array(xy, jnp.int32))
        m["active"] = m["active"].at[i].set(True)
        m["hp"] = m["hp"].at[i].set(20)
        m["kind"] = m["kind"].at[i].set(kind)
    w = {**w, "players": p, "monsters": m}
    return {**w, "move": ro_move.make_state()} if block else w


def inputs(**kw):
    mv = jnp.zeros((MAX_PLAYERS,), jnp.int32)
    for k, v in kw.items():
        mv = mv.at[int(k[1:])].set(v)
    return {"move": mv, "attack": jnp.zeros((MAX_PLAYERS,), jnp.bool_)}


def pos(w, i=0):
    return tuple(int(v) for v in w["players"]["pos"][i])


def tick(w, inp):
    """One movement tick plus the tick counter, as step.py does."""
    w = _move(w, inp)
    return {**w, "tick": w["tick"] + 1}


def reference_steps(n_ticks, cost, acc=None):
    """Ticks (1-based) at which a continuously walking entity steps: acc += TICK_MS, step when acc >= cost.
    A rested entity starts with acc = cost (its first intent steps at once)."""
    acc, out = (cost if acc is None else acc), []
    for t in range(1, n_ticks + 1):
        acc += TICK_MS
        if acc >= cost:
            acc -= cost
            out.append(t)
    return out


def walk(w, code, n, slot=0):
    trail = []
    for _ in range(n):
        w = tick(w, inputs(**{f"p{slot}": code}))
        trail.append(pos(w, slot))
    return w, trail


def test_tables_agree():
    assert len(ro_move.DIR_DELTAS9) == 9 and ro_move.DIR_DELTAS9[0] == (0, 0)
    for code in range(1, 9):
        delta = jnp.array(ro_move.DIR_DELTAS9[code])
        assert int(ro_move.dir_toward(delta)) == code - 1
        assert int(ro_move.code_to_dir(code)) == code - 1
    assert int(ro_move.dir_toward(jnp.array([0, 0]))) == -1
    assert int(ro_move.dir_toward(jnp.array([5, -3]))) == NE - 1
    assert [ro_move.DIR_DELTAS9[c] for c in ro_move.LEGACY_TO_RO[1:]] == [(0, -1), (0, 1), (-1, 0), (1, 0)]


def test_straight_cadence_matches_accumulator():
    w, trail = walk(world_with([(10, 10)]), E, 20)
    steps = reference_steps(20, 150)
    xs = [x for x, _ in trail]
    assert [t for t in range(1, 21) if xs[t - 1] != (xs[t - 2] if t > 1 else 10)] == steps
    assert xs[-1] == 10 + 20 * TICK_MS // 150 + 1  # one cell per 150 ms on average, plus the rested first step


@pytest.mark.skipif(TICK_MS != 100, reason="contract example is stated for TICK_MS 100")
def test_contract_example_at_100ms():
    assert reference_steps(8, 150, acc=0) == [2, 3, 5, 6, 8]
    assert reference_steps(8, 150) == [1, 2, 3, 5, 6, 8]  # rested: the first step is immediate


def test_diagonal_costs_fourteen_tenths():
    w, trail = walk(world_with([(10, 10)]), SE, 30)
    steps = reference_steps(30, 150 * 14 // 10)
    assert trail[-1] == (10 + len(steps), 10 + len(steps))
    assert len(steps) == 30 * TICK_MS // 210 + 1  # rested first step


def test_first_step_is_immediate_and_idle_refills_to_one_step():
    w = world_with([(10, 10)])
    w, trail = walk(w, E, 1)
    assert trail == [(11, 10)] and int(w["move"]["acc_ms"][0]) == 250 - 150  # rested: steps at once, keeps the carry
    w, trail = walk(w, E, 1)
    assert trail == [(12, 10)] and int(w["move"]["acc_ms"][0]) == 200 - 150
    w, trail = walk(w, E, 1)
    assert trail == [(13, 10)] and int(w["move"]["acc_ms"][0]) == 0
    w, trail = walk(w, E, 1)
    assert trail == [(13, 10)] and int(w["move"]["acc_ms"][0]) == TICK_MS  # the fourth tick waits: 150 ms per cell
    for _ in range(3):
        w = tick(w, inputs())  # the pending intent steps once more, then idle refills to one step cost and no further
    assert pos(w) == (14, 10) and int(w["move"]["acc_ms"][0]) == 150 and not bool(w["move"]["moving"][0])


def test_tap_makes_exactly_one_step():
    w = tick(world_with([(10, 10)]), inputs(p0=W))
    for _ in range(6):
        w = tick(w, inputs())
    assert pos(w) == (9, 10)


def test_blocked_step_keeps_accumulator_and_retries():
    w = world_with([(10, 10), (11, 10)])
    for _ in range(4):
        w = tick(w, inputs(p0=E))
    assert pos(w) == (10, 10)
    assert int(w["move"]["acc_ms"][0]) == 150  # capped at the cost, not reset
    assert int(w["move"]["dir"][0]) == E - 1  # faces where it tries to go
    p = w["players"]
    w = {**w, "players": {**p, "pos": p["pos"].at[1].set(jnp.array([30, 30], jnp.int32))}}
    w = tick(w, inputs(p0=E))
    assert pos(w) == (11, 10)  # steps on the first free tick


def test_no_corner_cutting():
    for wall in ((11, 10), (10, 11)):
        w = world_with([(10, 10)])
        w = {**w, "walls": w["walls"].at[wall].set(1)}
        w, trail = walk(w, SE, 6)
        assert trail[-1] == (10, 10)
    w, trail = walk(world_with([(10, 10)]), SE, 6)
    assert trail[-1] != (10, 10)


def test_diagonal_out_of_bounds_blocked():
    w, trail = walk(world_with([(0, 0)]), NW, 5)
    assert trail[-1] == (0, 0)


def test_lowest_slot_wins_contested_tile():
    w = world_with([(9, 9), (11, 9)])
    for _ in range(2):  # rested: both contest (10, 10) on tick 1; a diagonal (210 ms) does not step again on tick 2
        w = tick(w, inputs(p0=SE, p1=SW))
    assert pos(w, 0) == (10, 10) and pos(w, 1) == (11, 9)


def test_speed_is_per_entity():
    w = world_with([(10, 10), (10, 20)])
    mv = w["move"]
    w = {**w, "move": {**mv, "speed_ms": mv["speed_ms"].at[1].set(75)}}
    for _ in range(10):
        w = tick(w, inputs(p0=E, p1=E))
    assert pos(w, 0)[0] - 10 == len(reference_steps(10, 150))
    assert pos(w, 1)[0] - 10 == len(reference_steps(10, 75))  # at most one cell per tick


def test_dead_player_drops_intent():
    w = tick(world_with([(10, 10)]), inputs(p0=E))
    assert pos(w) == (11, 10)  # rested: the first tick steps
    p = w["players"]
    w = {**w, "players": {**p, "dead": p["dead"].at[0].set(True)}}
    w = tick(w, inputs(p0=E))
    assert pos(w) == (11, 10) and int(w["move"]["path_dir"][0]) == -1


def test_monster_walks_at_its_kind_speed():
    k = KIND_BY_NAME["Rotling"]
    w = world_with([], monsters=[(28, 20)], kind=k)
    speed = int(ro_move._MON_SPEED[k])
    step = jax.jit(lambda w: ro_move.step_entities(ro_move.want_monsters(w, jnp.full((MAX_MONSTERS,), W - 1)), "monsters"))
    for _ in range(12):
        w = step(w)
    assert int(w["move"]["speed_ms"][MAX_PLAYERS]) == speed
    assert int(w["monsters"]["pos"][0][0]) == 28 - len(reference_steps(12, speed))
    assert int(w["move"]["dir"][MAX_PLAYERS]) == W - 1


def test_monster_and_player_share_occupancy():
    w = world_with([(10, 10)], monsters=[(11, 10)], kind=KIND_BY_NAME["Blobbin"])
    w, trail = walk(w, E, 4)
    assert trail[-1] == (10, 10)


def test_legacy_path_without_block_steps_every_tick():
    w = world_with([(10, 10)], block=False)
    w = {k: v for k, v in w.items() if k != "move"}  # state.make_empty_world carries the block; drop it here
    w = tick(w, inputs(p0=SE))
    assert pos(w) == (11, 11) and "move" not in w


def test_jit_and_vmap_over_worlds():
    ws = [world_with([(10 + i, 10)]) for i in range(3)]
    batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *ws)
    inp = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[inputs(p0=S)] * 3)
    f = jax.jit(jax.vmap(move_players))
    out = f(f(batch, inp), inp)
    assert out["move"]["acc_ms"].shape == (3, MAX_PLAYERS + MAX_MONSTERS)
    for i in range(3):
        assert tuple(int(v) for v in out["players"]["pos"][i, 0]) == (10 + i, 12)  # rested: both ticks step
