import jax
import jax.numpy as jnp
import numpy as np

from game import ro_mvp
from game.config import ITEM_GOLD_COIN, ITEM_MEAT, ITEM_NONE, ITEM_SWORD, MAX_MONSTERS
from game.state import make_empty_world

SLOT = MAX_MONSTERS - 3  # kind 2


def mk():
    w = make_empty_world(jax.random.PRNGKey(0))
    w = {**w, "mvp": ro_mvp.make_state()}
    p = w["players"]
    w = {**w, "players": {**p, "active": p["active"].at[:4].set(True)}}
    return ro_mvp.place_mvps(w)


def i32(*a):
    return jnp.array(a, jnp.int32)


def kill(w, slot=SLOT):
    m = w["monsters"]
    return {**w, "monsters": {**m, "died": m["died"].at[slot].set(True), "active": m["active"].at[slot].set(False)}}


def test_record_damage_ledger_and_tie():
    w = mk()
    w = ro_mvp.record_damage(w, i32(5, 3, 7, 2, 1), i32(SLOT, SLOT, SLOT, -1, SLOT), i32(10, 10, 4, 99, 0))
    v = w["mvp"]
    assert int(v["damage_ledger"][SLOT]) == 24
    assert int(v["top_damage"][SLOT]) == 10
    assert int(v["top_damager"][SLOT]) == 3
    w = ro_mvp.record_damage(w, i32(9), i32(SLOT), i32(10))
    assert int(w["mvp"]["top_damager"][SLOT]) == 3
    w = ro_mvp.record_damage(w, i32(1), i32(SLOT), i32(10))
    assert int(w["mvp"]["top_damager"][SLOT]) == 1
    w = ro_mvp.record_damage(w, i32(8), i32(SLOT), i32(11))
    assert int(w["mvp"]["top_damager"][SLOT]) == 8
    assert int(w["mvp"]["damage_ledger"][SLOT]) == 55


def test_event_drops_and_reset():
    w = mk()
    w = {**w, "tick": jnp.int32(42)}
    w = ro_mvp.record_damage(w, i32(2), i32(SLOT), i32(50))
    w = ro_mvp.phase(kill(w))
    v = w["mvp"]
    assert list(map(int, v["event"][0])) == [2, 2, 42]
    assert int(v["event_head"]) == 1
    assert int(v["next_spawn"][2]) == 3000
    assert int(v["top_damager"][SLOT]) == -1 and int(v["damage_ledger"][SLOT]) == 0
    inv = np.asarray(w["players"]["inv"][2])
    assert list(inv[:4]) == [ITEM_SWORD, ITEM_GOLD_COIN, ITEM_GOLD_COIN, ITEM_MEAT]
    assert (np.asarray(w["players"]["inv"][0]) == ITEM_NONE).all()


def test_drops_skip_full_slots():
    w = mk()
    inv = w["players"]["inv"].at[1, 0].set(9)
    w = {**w, "players": {**w["players"], "inv": inv}}
    w = ro_mvp.record_damage(w, i32(1), i32(SLOT), i32(5))
    w = ro_mvp.phase(kill(w))
    assert list(np.asarray(w["players"]["inv"][1][:5])) == [9, ITEM_SWORD, ITEM_GOLD_COIN, ITEM_GOLD_COIN, ITEM_MEAT]


def test_non_mvp_death_writes_nothing():
    w = mk()
    m = w["monsters"]
    w = {**w, "monsters": {**m, "active": m["active"].at[5].set(True)}}
    w = ro_mvp.record_damage(w, i32(0), i32(5), i32(5))
    w = ro_mvp.phase(kill(w, 5))
    assert int(w["mvp"]["event_head"]) == 0
    assert (np.asarray(w["mvp"]["event"]) == -1).all()
    assert int(w["mvp"]["damage_ledger"][5]) == 0
    assert (np.asarray(w["players"]["inv"]) == ITEM_NONE).all()


def test_respawn_on_fixed_tile_full_hp():
    ph = jax.jit(ro_mvp.phase)
    w = mk()
    w = ro_mvp.phase(kill(w))
    m = w["monsters"]
    w = {**w, "monsters": {**m, "died": jnp.zeros_like(m["died"]), "respawn_in": m["respawn_in"].at[SLOT].set(5),
                           "pos": m["pos"].at[SLOT].set(i32(1, 1))}}
    w = ph(w)
    assert int(w["monsters"]["respawn_in"][SLOT]) == 0
    for _ in range(2998):
        w = ph(w)
    assert not bool(w["monsters"]["active"][SLOT])
    w = ph(w)
    m = w["monsters"]
    assert bool(m["active"][SLOT]) and int(m["hp"][SLOT]) == 2000
    assert tuple(map(int, m["pos"][SLOT])) == ro_mvp.MVP_SPAWN_TILE[2]
    assert int(w["mvp"]["next_spawn"][2]) == 0


def test_ring_wraps():
    w = mk()
    for t in range(9):
        w = {**w, "tick": jnp.int32(100 + t)}
        w = ro_mvp.record_damage(w, i32(t % 4), i32(SLOT), i32(3))
        w = ro_mvp.phase(kill(w))
        m = w["monsters"]
        w = {**w, "monsters": {**m, "died": jnp.zeros_like(m["died"])},
             "mvp": {**w["mvp"], "next_spawn": jnp.zeros(3, jnp.int32)}}
    assert int(w["mvp"]["event_head"]) == 9
    assert int(w["mvp"]["event"][0, 2]) == 108
    assert int(w["mvp"]["event"][1, 2]) == 101


def test_jit_and_vmap():
    w = mk()
    w = ro_mvp.record_damage(w, i32(2), i32(SLOT), i32(50))
    out = jax.jit(ro_mvp.phase)(kill(w))
    assert int(out["mvp"]["event_head"]) == 1
    b = jax.tree_util.tree_map(lambda a, c: jnp.stack([a, c]), kill(w), mk())
    out = jax.jit(jax.vmap(ro_mvp.phase))(b)
    assert list(map(int, out["mvp"]["event_head"])) == [1, 0]
    rd = jax.vmap(ro_mvp.record_damage)(
        b, jnp.stack([i32(1), i32(1)]), jnp.stack([i32(SLOT)] * 2), jnp.stack([i32(7)] * 2))
    assert list(map(int, rd["mvp"]["top_damage"][:, SLOT])) == [50, 7]
