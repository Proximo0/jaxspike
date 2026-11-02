import jax
import jax.numpy as jnp

from bots.policies import HUNT_RADIUS, bot_inputs
from game.config import HUNT_RETREAT_HP, ITEM_MEAT, MAP, MAX_PLAYERS, PLAYER_MAX_HP
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


def _meat_world(hp, meat=True, dead=False):
    w = _world((50, 50))
    p = w["players"]
    p = {**p, "hp": p["hp"].at[0].set(hp), "dead": p["dead"].at[0].set(dead)}
    if meat:
        p["inv"] = p["inv"].at[0, 3].set(ITEM_MEAT)
    return {**w, "players": p}


def test_bot_eats_below_half_hp():
    assert bool(bot_inputs(_meat_world(PLAYER_MAX_HP // 2 - 1), KEY)["eat"][0])
    assert not bool(bot_inputs(_meat_world(PLAYER_MAX_HP // 2), KEY)["eat"][0])


def test_bot_eat_needs_meat_and_life():
    assert not bool(bot_inputs(_meat_world(10, meat=False), KEY)["eat"][0])
    assert not bool(bot_inputs(_meat_world(10, dead=True), KEY)["eat"][0])


def _hunt_world(pos, hp=PLAYER_MAX_HP, pz=False, monster_pos=None, wall=None):
    """Player 0 (home town 0) with a gate-distance field that grows with x: gates lie to the west."""
    w = _world(pos, monster_pos)
    xs = jnp.arange(MAP)[:, None] + jnp.zeros((MAP, MAP), jnp.int32)
    field = jnp.broadcast_to(jnp.abs(xs - 40).astype(jnp.int16), w["dist_town"].shape)
    p = {**w["players"], "hp": w["players"]["hp"].at[0].set(hp)}
    w = {**w, "players": p, "dist_town": field,
         "pz": w["pz"].at[pos[0], pos[1]].set(pz)}
    if wall is not None:
        w["walls"] = w["walls"].at[wall[0], wall[1]].set(1)
    return w


def test_hunting_bot_walks_away_from_gate():
    out = bot_inputs(_hunt_world((50, 50)), KEY)
    assert int(out["move"][0]) == 4  # east: larger distance


def test_healthy_bot_in_town_heads_for_gate():
    out = bot_inputs(_hunt_world((50, 50), pz=True), KEY)
    assert int(out["move"][0]) == 3  # west: smaller distance


def test_hurt_bot_goes_home_and_ignores_monster():
    out = bot_inputs(_hunt_world((50, 50), hp=HUNT_RETREAT_HP - 1, monster_pos=(51, 50)), KEY)
    assert int(out["move"][0]) == 3 and not bool(out["attack"][0])


def test_resting_bot_stays_in_town():
    out = bot_inputs(_hunt_world((50, 50), hp=PLAYER_MAX_HP - 1, pz=True), KEY)
    assert int(out["move"][0]) == 0


def test_bot_far_beyond_hunt_radius_turns_back():
    out = bot_inputs(_hunt_world((40 + 2 * HUNT_RADIUS + 5, 50)), KEY)
    assert int(out["move"][0]) == 3


def test_bot_at_hunt_radius_keeps_a_heading():
    w = _hunt_world((40 + HUNT_RADIUS + 5, 50))
    a = int(bot_inputs(w, KEY)["move"][0])
    b = int(bot_inputs(w, jax.random.PRNGKey(9))["move"][0])
    assert a == b and 1 <= a <= 4  # same tick, different keys: the heading does not change
    moves = {int(bot_inputs({**w, "tick": jnp.int32(t * 50)}, KEY)["move"][0]) for t in range(8)}
    assert len(moves) >= 2


def test_bot_does_not_step_into_wall():
    for pos in ((50, 50), (40 + HUNT_RADIUS + 5, 50)):
        for k in range(6):
            out = bot_inputs(_hunt_world(pos, wall=(pos[0] + 1, pos[1])), jax.random.PRNGKey(k))
            assert int(out["move"][0]) != 4


def test_healthy_bot_outside_still_fights():
    out = bot_inputs(_hunt_world((50, 50), monster_pos=(51, 50)), KEY)
    assert bool(out["attack"][0]) and int(out["move"][0]) == 4


def test_bot_on_gate_apron_does_not_walk_back_into_town():
    # Apron at x=40 (dist 0); the gate tile to the west is pz with dist 1, the road east has dist 1 too.
    w = _hunt_world((40, 50))
    w["pz"] = w["pz"].at[39, 50].set(True)
    for k in range(6):
        assert int(bot_inputs(w, jax.random.PRNGKey(k))["move"][0]) != 3
