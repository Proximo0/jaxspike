import jax
import jax.numpy as jnp

from game.config import (
    INV_SLOTS, ITEM_MEAT, ITEM_NONE, MAX_PLAYERS, MEAT_HEAL, PLAYER_MAX_HP, REGEN_HP, REGEN_TICKS,
)
from game import ro_stats
from game.state import empty_inputs, make_empty_world
from game.survival import survival


def _world(hp=100, inv=None, tick=0, dead=False):
    w = make_empty_world(jax.random.PRNGKey(0))
    p = w["players"]
    p = {**p, "active": p["active"].at[0].set(True), "dead": p["dead"].at[0].set(dead),
         "hp": p["hp"].at[0].set(hp)}
    if inv is not None:
        p["inv"] = p["inv"].at[0].set(jnp.array(inv, jnp.int32))
    return {**w, "players": p, "tick": jnp.int32(tick)}


def _inv(*items):
    return list(items) + [ITEM_NONE] * (INV_SLOTS - len(items))


def _eat():
    return {**empty_inputs(), "eat": jnp.zeros((MAX_PLAYERS,), bool).at[0].set(True)}


def test_regen_on_multiple_of_regen_ticks():
    out = survival(_world(tick=REGEN_TICKS), empty_inputs())
    assert int(out["players"]["hp"][0]) == 100 + REGEN_HP


def test_no_regen_off_schedule():
    out = survival(_world(tick=REGEN_TICKS + 1), empty_inputs())
    assert int(out["players"]["hp"][0]) == 100


def test_regen_capped_at_max():
    w = _world(tick=REGEN_TICKS)
    cap = _cap(w)
    w = _set_hp(w, cap)
    assert int(survival(w, empty_inputs())["players"]["hp"][0]) == cap


def test_dead_and_inactive_do_not_regen():
    out = survival(_world(tick=REGEN_TICKS, dead=True), empty_inputs())
    assert int(out["players"]["hp"][0]) == 100
    w = _world(tick=REGEN_TICKS)
    hp = w["players"]["hp"].at[1].set(100)
    out = survival({**w, "players": {**w["players"], "hp": hp}}, empty_inputs())
    assert int(out["players"]["hp"][1]) == 100


def test_eat_removes_one_meat_and_heals():
    out = survival(_world(hp=50, inv=_inv(ITEM_MEAT, ITEM_MEAT), tick=1), _eat())
    assert int(out["players"]["hp"][0]) == 50 + MEAT_HEAL
    inv = out["players"]["inv"][0]
    assert int(inv[0]) == ITEM_NONE and int(inv[1]) == ITEM_MEAT
    assert int((inv == ITEM_MEAT).sum()) == 1


def _cap(w):
    return int(ro_stats.derived(w)["max_hp"][0])


def _set_hp(w, hp):
    return {**w, "players": {**w["players"], "hp": w["players"]["hp"].at[0].set(hp)}}


def _leveled(w, level=10, vit=20):
    st = dict(w["stats"])
    st["base_level"] = st["base_level"].at[0].set(level)
    st["vit"] = st["vit"].at[0].set(vit)
    return {**w, "stats": st}


def test_regen_caps_at_derived_max_hp_for_leveled_player():
    w = _leveled(_world(hp=1, tick=REGEN_TICKS))
    cap = int(ro_stats.derived(w)["max_hp"][0])
    assert cap > PLAYER_MAX_HP
    w = {**w, "players": {**w["players"], "hp": w["players"]["hp"].at[0].set(cap - 1)}}
    assert int(survival(w, empty_inputs())["players"]["hp"][0]) == cap
    w2 = {**w, "players": {**w["players"], "hp": w["players"]["hp"].at[0].set(cap)}}
    assert int(survival(w2, empty_inputs())["players"]["hp"][0]) == cap


def test_eat_heals_capped():
    w = _world(inv=_inv(ITEM_MEAT), tick=1)
    cap = _cap(w)
    out = survival(_set_hp(w, cap - 10), _eat())
    assert int(out["players"]["hp"][0]) == cap


def test_eat_at_full_hp_consumes_meat():
    w = _world(inv=_inv(ITEM_MEAT), tick=1)
    cap = _cap(w)
    out = survival(_set_hp(w, cap), _eat())
    assert int(out["players"]["hp"][0]) == cap
    assert not bool((out["players"]["inv"][0] == ITEM_MEAT).any())


def test_eat_without_meat_noop():
    out = survival(_world(hp=50, tick=1), _eat())
    assert int(out["players"]["hp"][0]) == 50
    assert not bool(out["players"]["inv"].any())


def test_dead_player_cannot_eat():
    w = _world(hp=50, inv=_inv(ITEM_MEAT), tick=1, dead=True)
    out = survival(w, _eat())
    assert bool((out["players"]["inv"] == w["players"]["inv"]).all())
    assert int(out["players"]["hp"][0]) == 50


def test_jit_and_vmap():
    w = _world(hp=50, inv=_inv(ITEM_MEAT), tick=REGEN_TICKS)
    eager = survival(w, _eat())
    jitted = jax.jit(survival)(w, _eat())
    assert bool((eager["players"]["hp"] == jitted["players"]["hp"]).all())
    assert bool((eager["players"]["inv"] == jitted["players"]["inv"]).all())
    batch = jax.tree_util.tree_map(lambda *a: jnp.stack(a), w, w)
    inputs = jax.tree_util.tree_map(lambda *a: jnp.stack(a), _eat(), _eat())
    out = jax.vmap(survival)(batch, inputs)
    assert out["players"]["hp"].shape == (2, MAX_PLAYERS)


def test_regen_and_eat_sum_before_cap():
    out = survival(_world(hp=50, inv=_inv(ITEM_MEAT), tick=REGEN_TICKS), _eat())
    assert int(out["players"]["hp"][0]) == 50 + REGEN_HP + MEAT_HEAL
    assert not bool((out["players"]["inv"][0] == ITEM_MEAT).any())
