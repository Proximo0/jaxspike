import jax
import jax.numpy as jnp
import numpy as np

from game import ro_mvp
from game.config import MAX_MONSTERS
from game.data import monsters as mon
from game.data.items import ITEM_NONE
from game.state import make_empty_world

KIND = mon.KIND_BY_NAME["Eddga"]
assert bool(mon.MVP[KIND])
SLOT = MAX_MONSTERS - 3
TILE = (40, 41)
DROP = [int(x) for x in ro_mvp.MVP_DROP[KIND]]


def mk():
    w = make_empty_world(jax.random.PRNGKey(0))
    w = {**w, "mvp": ro_mvp.make_state()}
    p = w["players"]
    w = {**w, "players": {**p, "active": p["active"].at[:4].set(True)}}
    m = w["monsters"]
    return {**w, "monsters": {**m, "kind": m["kind"].at[SLOT].set(KIND), "hp": m["hp"].at[SLOT].set(int(mon.HP[KIND])),
                              "pos": m["pos"].at[SLOT].set(i32(*TILE)), "spawn": m["spawn"].at[SLOT].set(i32(*TILE)),
                              "active": m["active"].at[SLOT].set(True)}}


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
    assert list(map(int, v["event"][0])) == [KIND, 2, 42]
    assert int(v["event_head"]) == 1
    assert int(v["next_spawn"][KIND]) == 3000
    assert int(v["top_damager"][SLOT]) == -1 and int(v["damage_ledger"][SLOT]) == 0
    inv = np.asarray(w["players"]["inv"][2])
    assert list(inv[:4]) == DROP
    assert (np.asarray(w["players"]["inv"][0]) == ITEM_NONE).all()


def test_drops_skip_full_slots():
    w = mk()
    inv = w["players"]["inv"].at[1, 0].set(9)
    w = {**w, "players": {**w["players"], "inv": inv}}
    w = ro_mvp.record_damage(w, i32(1), i32(SLOT), i32(5))
    w = ro_mvp.phase(kill(w))
    assert list(np.asarray(w["players"]["inv"][1][:5])) == [9] + DROP


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
    assert bool(m["active"][SLOT]) and int(m["hp"][SLOT]) == int(mon.HP[KIND])
    assert tuple(map(int, m["pos"][SLOT])) == TILE
    assert int(w["mvp"]["next_spawn"][KIND]) == 0


def test_ring_wraps():
    w = mk()
    for t in range(9):
        w = {**w, "tick": jnp.int32(100 + t)}
        w = ro_mvp.record_damage(w, i32(t % 4), i32(SLOT), i32(3))
        w = ro_mvp.phase(kill(w))
        m = w["monsters"]
        w = {**w, "monsters": {**m, "died": jnp.zeros_like(m["died"])},
             "mvp": {**w["mvp"], "next_spawn": jnp.zeros(mon.N_KINDS, jnp.int32)}}
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


def test_mvp_identified_by_table_flag():
    w = mk()
    m = w["monsters"]
    flags = np.asarray(ro_mvp.is_mvp(w))
    assert flags[SLOT] and flags.sum() == (np.asarray(m["kind"]) == KIND).sum()
    assert list(np.flatnonzero(np.asarray(mon.MVP))) == [18, 19]
    assert ro_mvp.MVP_DROP[0] == (0,) * 4 and len(ro_mvp.MVP_DROP) == mon.N_KINDS
    w2 = {**w, "monsters": {**m, "kind": m["kind"].at[SLOT].set(2)}}  # Fabre is not an MVP
    assert not bool(ro_mvp.is_mvp(w2)[SLOT])


def test_other_mvp_kind_independent_timer():
    gtb = mon.KIND_BY_NAME["Golden Thief Bug"]
    w = mk()
    m = w["monsters"]
    s2 = SLOT - 1
    w = {**w, "monsters": {**m, "kind": m["kind"].at[s2].set(gtb), "spawn": m["spawn"].at[s2].set(i32(50, 51)),
                           "pos": m["pos"].at[s2].set(i32(50, 51)), "active": m["active"].at[s2].set(True)}}
    w = ro_mvp.phase(kill(w, s2))
    assert int(w["mvp"]["next_spawn"][gtb]) == 3000 and int(w["mvp"]["next_spawn"][KIND]) == 0
    assert int(w["mvp"]["event"][0, 0]) == gtb
    assert bool(w["monsters"]["active"][SLOT])
