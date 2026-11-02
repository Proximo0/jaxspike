import jax
import jax.numpy as jnp

from game.combat import combat
from game.config import (
    MONSTER_ATTACK_COOLDOWN,
    MONSTER_DAMAGE,
    MONSTER_HP,
    MONSTER_RESPAWN_TICKS,
    PLAYER_ATTACK,
    PLAYER_ATTACK_COOLDOWN,
    PLAYER_MAX_HP,
)
from game.state import empty_inputs, make_empty_world


def _world(player_pos=(20, 20), monster_pos=(21, 20), kind=0):
    w = make_empty_world(jax.random.PRNGKey(0))
    p, m = w["players"], w["monsters"]
    p = {**p, "pos": p["pos"].at[0].set(jnp.array(player_pos)),
         "spawn": p["spawn"].at[0].set(jnp.array([30, 30])),
         "active": p["active"].at[0].set(True)}
    m = {**m, "pos": m["pos"].at[0].set(jnp.array(monster_pos)),
         "hp": m["hp"].at[0].set(MONSTER_HP[kind]),
         "kind": m["kind"].at[0].set(kind),
         "active": m["active"].at[0].set(True)}
    return {**w, "players": p, "monsters": m}


def _attack():
    i = empty_inputs()
    return {**i, "attack": i["attack"].at[0].set(True)}


def test_player_kills_rat_and_gets_credit():
    w = combat(_world(), _attack())
    assert int(w["monsters"]["hp"][0]) == MONSTER_HP[0] - PLAYER_ATTACK
    assert int(w["players"]["attack_cd"][0]) == PLAYER_ATTACK_COOLDOWN
    assert int(w["players"]["kills"][0]) == 0
    w = {**w, "players": {**w["players"], "attack_cd": jnp.zeros_like(w["players"]["attack_cd"])}}
    w = {**w, "monsters": {**w["monsters"], "hp": w["monsters"]["hp"].at[0].set(PLAYER_ATTACK)}}
    w = combat(w, _attack())
    assert not bool(w["monsters"]["active"][0])
    assert int(w["players"]["kills"][0]) == 1


def test_cooldown_blocks_second_attack():
    w = combat(_world(), _attack())
    w = combat(w, _attack())
    assert int(w["monsters"]["hp"][0]) == MONSTER_HP[0] - PLAYER_ATTACK
    assert int(w["players"]["attack_cd"][0]) == PLAYER_ATTACK_COOLDOWN - 1


def test_not_adjacent_no_attack():
    w = combat(_world(monster_pos=(23, 20)), _attack())
    assert int(w["monsters"]["hp"][0]) == MONSTER_HP[0]


def test_monster_hits_adjacent_player():
    w = combat(_world(kind=1), empty_inputs())
    assert int(w["players"]["hp"][0]) == PLAYER_MAX_HP - MONSTER_DAMAGE[1]
    assert int(w["monsters"]["attack_cd"][0]) == MONSTER_ATTACK_COOLDOWN
    w = combat(w, empty_inputs())
    assert int(w["players"]["hp"][0]) == PLAYER_MAX_HP - MONSTER_DAMAGE[1]


def test_monster_cannot_hit_player_on_pz():
    w = _world(kind=1)
    w = {**w, "pz": w["pz"].at[20, 20].set(True)}
    w = combat(w, empty_inputs())
    assert int(w["players"]["hp"][0]) == PLAYER_MAX_HP
    assert int(w["monsters"]["attack_cd"][0]) == 0


def test_player_on_pz_can_still_attack():
    w = _world()
    w = {**w, "pz": w["pz"].at[20, 20].set(True)}
    w = combat(w, _attack())
    assert int(w["monsters"]["hp"][0]) == MONSTER_HP[0] - PLAYER_ATTACK


def test_player_death_counts():
    w = _world()
    w = {**w, "players": {**w["players"], "hp": w["players"]["hp"].at[0].set(1)}}
    w = combat(w, empty_inputs())
    assert bool(w["players"]["dead"][0])
    assert int(w["players"]["deaths"][0]) == 1
    assert bool(w["players"]["active"][0])
    w = combat(w, empty_inputs())  # respawn, no double count
    assert int(w["players"]["deaths"][0]) == 1


def test_dead_player_respawns():
    w = _world(monster_pos=(100, 100))
    p = w["players"]
    p = {**p, "dead": p["dead"].at[0].set(True), "hp": p["hp"].at[0].set(0)}
    w = combat({**w, "players": p}, empty_inputs())
    assert not bool(w["players"]["dead"][0])
    assert int(w["players"]["hp"][0]) == PLAYER_MAX_HP
    assert w["players"]["pos"][0].tolist() == [30, 30]


def _two_dead(second_pos, second_dead):
    """Player 0 dead with spawn (30, 30); player 1 at second_pos, dead or living."""
    w = _world(monster_pos=(100, 100))
    p = w["players"]
    p = {**p, "dead": p["dead"].at[0].set(True).at[1].set(second_dead),
         "hp": p["hp"].at[0].set(0),
         "pos": p["pos"].at[1].set(jnp.array(second_pos)),
         "spawn": p["spawn"].at[1].set(jnp.array([40, 40])),
         "active": p["active"].at[1].set(True)}
    return {**w, "players": p}


def test_respawn_blocked_by_living_player_on_spawn():
    w = combat(_two_dead((30, 30), False), empty_inputs())
    assert bool(w["players"]["dead"][0])
    assert w["players"]["pos"][0].tolist() == [20, 20]


def test_respawn_allowed_onto_waiting_dead_player():
    # Dead players are not physical: player 1 is dead on player 0's spawn tile, and its own
    # spawn (40, 40) is free, so both come back; neither blocks the other.
    w = combat(_two_dead((30, 30), True), empty_inputs())
    assert not bool(w["players"]["dead"][0]) and not bool(w["players"]["dead"][1])
    assert w["players"]["pos"][0].tolist() == [30, 30]
    assert w["players"]["pos"][1].tolist() == [40, 40]


def test_monster_death_flags():
    w = _world()
    w = {**w, "monsters": {**w["monsters"], "hp": w["monsters"]["hp"].at[0].set(PLAYER_ATTACK)}}
    w = combat(w, _attack())
    assert bool(w["monsters"]["died"][0])
    assert not bool(w["monsters"]["active"][0])
    assert int(w["monsters"]["respawn_in"][0]) == MONSTER_RESPAWN_TICKS


def test_jit_and_vmap():
    w = _world()
    batch = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), w)
    inp = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), _attack())
    out = jax.jit(jax.vmap(combat))(batch, inp)
    assert out["monsters"]["hp"][:, 0].tolist() == [MONSTER_HP[0] - PLAYER_ATTACK] * 2
    single = jax.jit(combat)(w, _attack())
    assert int(single["monsters"]["hp"][0]) == MONSTER_HP[0] - PLAYER_ATTACK
