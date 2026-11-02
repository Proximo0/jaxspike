import jax
import jax.numpy as jnp

from game.config import MONSTER_AGGRO_RANGE, MONSTER_HP
from game.data.monsters import AGGRO, KIND_BY_NAME, MOVE_TICKS
from game.monsters import monster_ai
from game.state import make_empty_world


def build(players=(), monsters=(), walls=(), pz=(), kind=1, tick=0):
    w = make_empty_world(jax.random.PRNGKey(0))
    p, m = w["players"], w["monsters"]
    for i, pos in enumerate(players):
        p["pos"] = p["pos"].at[i].set(jnp.array(pos, jnp.int32))
        p["active"] = p["active"].at[i].set(True)
    for i, pos in enumerate(monsters):
        m["pos"] = m["pos"].at[i].set(jnp.array(pos, jnp.int32))
        m["spawn"] = m["spawn"].at[i].set(jnp.array(pos, jnp.int32))
        m["active"] = m["active"].at[i].set(True)
        m["hp"] = m["hp"].at[i].set(20)
        m["kind"] = m["kind"].at[i].set(kind)  # default 1 = Orc Warrior: aggro, moves every tick
    w["tick"] = jnp.int32(tick)
    for x, y in walls:
        w["walls"] = w["walls"].at[x, y].set(1)
    for x, y in pz:
        w["pz"] = w["pz"].at[x, y].set(True)
    return w


def mpos(w, i):
    return tuple(int(v) for v in w["monsters"]["pos"][i])


def test_moves_toward_player_in_range():
    w = monster_ai(build(players=[(20, 20)], monsters=[(24, 21)]))
    assert mpos(w, 0) == (23, 21)  # longer axis is x


def test_ignores_player_outside_aggro():
    far = MONSTER_AGGRO_RANGE + 1
    w = monster_ai(build(players=[(20, 20)], monsters=[(20 + far, 20)]))
    assert mpos(w, 0) == (20 + far, 20)


def test_dead_player_not_targeted():
    w = build(players=[(20, 20)], monsters=[(24, 20)])
    w["players"]["dead"] = w["players"]["dead"].at[0].set(True)
    assert mpos(monster_ai(w), 0) == (24, 20)


def test_wall_blocks_uses_fallback_axis():
    # Monster at (24,21), player at (20,20): x preferred, wall at (23,21) forces y step.
    w = monster_ai(build(players=[(20, 20)], monsters=[(24, 21)], walls=[(23, 21)]))
    assert mpos(w, 0) == (24, 20)


def test_monster_does_not_enter_pz():
    w = monster_ai(build(players=[(20, 20)], monsters=[(24, 20)], pz=[(23, 20)]))
    assert mpos(w, 0) == (24, 20)


def test_monster_steps_around_pz_tile():
    w = monster_ai(build(players=[(20, 20)], monsters=[(24, 21)], pz=[(23, 21)]))
    assert mpos(w, 0) == (24, 20)


def test_does_not_step_onto_player_or_monster():
    w = monster_ai(build(players=[(20, 20)], monsters=[(21, 20), (23, 20)]))
    assert mpos(w, 0) == (21, 20)  # adjacent to the player, stays
    assert mpos(w, 1) == (22, 20)  # (22,20) was free, so it advances
    # A monster directly behind another one is blocked:
    w = monster_ai(build(players=[(20, 20)], monsters=[(22, 20), (23, 20)]))
    assert mpos(w, 1) == (23, 20)


def test_respawn_countdown_and_reset():
    w = build(monsters=[(30, 30)])
    m = w["monsters"]
    m["active"] = m["active"].at[0].set(False)
    m["kind"] = m["kind"].at[0].set(1)
    m["hp"] = m["hp"].at[0].set(0)
    m["pos"] = m["pos"].at[0].set(jnp.array([40, 40], jnp.int32))
    m["respawn_in"] = m["respawn_in"].at[0].set(2)
    w = monster_ai(w)
    assert not bool(w["monsters"]["active"][0])
    assert int(w["monsters"]["respawn_in"][0]) == 1
    assert mpos(w, 0) == (40, 40)
    w = monster_ai(w)
    assert bool(w["monsters"]["active"][0])
    assert int(w["monsters"]["hp"][0]) == MONSTER_HP[1]
    assert mpos(w, 0) == (30, 30)
    assert int(w["monsters"]["attack_cd"][0]) == 0


def test_dead_monster_does_not_move():
    w = build(players=[(20, 20)], monsters=[(22, 20)])
    m = w["monsters"]
    m["active"] = m["active"].at[0].set(False)
    m["respawn_in"] = m["respawn_in"].at[0].set(5)
    assert mpos(monster_ai(w), 0) == (22, 20)


def test_lowest_index_wins_collision():
    # m0 (dx=2, dy=-1) steps in x and m1 (dx=1, dy=-2) steps in y: both want (23,21).
    w = monster_ai(build(players=[(24, 20)], monsters=[(22, 21), (23, 22)]))
    assert mpos(w, 0) == (23, 21)
    assert mpos(w, 1) == (23, 22)


def test_chase_does_not_step_onto_reviving_monster():
    # B (index 1) revives this tick on its spawn (20,20); A's one-step chase target is exactly that tile.
    w = build(players=[(18, 20)], monsters=[(21, 20), (20, 20)])
    m = w["monsters"]
    m["active"] = m["active"].at[1].set(False)
    m["pos"] = m["pos"].at[1].set(jnp.array([40, 40], jnp.int32))
    m["respawn_in"] = m["respawn_in"].at[1].set(1)
    w = monster_ai(w)
    assert bool(w["monsters"]["active"][0]) and bool(w["monsters"]["active"][1])
    assert mpos(w, 1) == (20, 20)
    assert mpos(w, 0) != mpos(w, 1)


def test_jit_and_vmap_batch():
    w1 = build(players=[(20, 20)], monsters=[(24, 20)])
    w2 = build(players=[(20, 20)], monsters=[(20, 25)])
    batch = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), w1, w2)
    out = jax.jit(jax.vmap(monster_ai))(batch)
    assert tuple(int(v) for v in out["monsters"]["pos"][0, 0]) == (23, 20)
    assert tuple(int(v) for v in out["monsters"]["pos"][1, 0]) == (20, 24)


def test_monster_chases_reachable_player_not_pz_player():
    # Player 0 is nearer (2 tiles, west) but on pz; player 1 is 3 tiles north.
    w = build(players=[(18, 20), (20, 23)], monsters=[(20, 20)], pz=[(18, 20)])
    assert mpos(monster_ai(w), 0) == (20, 21)


def test_non_aggro_kind_never_chases():
    for name in ("Poring", "Fabre", "Lunatic", "Drops"):
        k = KIND_BY_NAME[name]
        assert not bool(AGGRO[k])
        for tick in range(4):
            w = monster_ai(build(players=[(20, 20)], monsters=[(24, 21)], kind=k, tick=tick))
            assert mpos(w, 0) == (24, 21)


def test_aggro_kind_chases():
    k = KIND_BY_NAME["Thief Bug"]
    assert bool(AGGRO[k]) and int(MOVE_TICKS[k]) == 1
    assert mpos(monster_ai(build(players=[(20, 20)], monsters=[(24, 21)], kind=k)), 0) == (23, 21)


def test_move_ticks_gate_the_step():
    for name in ("Zombie", "Skeleton", "Orc Warrior"):
        k = KIND_BY_NAME[name]
        mt = int(MOVE_TICKS[k])
        assert bool(AGGRO[k])
        for tick in range(2 * mt + 1):
            w = monster_ai(build(players=[(20, 20)], monsters=[(24, 21)], kind=k, tick=tick))
            expect = (23, 21) if tick % mt == 0 else (24, 21)
            assert mpos(w, 0) == expect, (name, tick)
    assert max(int(v) for v in MOVE_TICKS) > 1
