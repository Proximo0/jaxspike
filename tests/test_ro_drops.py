"""Per-10000 drop chances, ground drops that carry count / refine / cards, and ro_drops.pickup."""
import jax
import jax.numpy as jnp
import numpy as np

from game import ro_drops, ro_items
from game.config import ITEM_GOLD_COIN, ITEM_NONE, ITEM_SWORD, MAX_DROPS, MAX_MONSTERS, MAX_PLAYERS
from game.data import drops as D
from game.data import items as I
from game.state import empty_inputs, make_empty_world

POS = (10, 12)


def make_world(fields=True, seed=0):
    w = make_empty_world(jax.random.PRNGKey(seed))
    w = {**w, "items": ro_items.make_state()}
    if fields:
        w = {**w, "drops": {**w["drops"], **ro_drops.drop_fields()}}
    else:  # state.make_empty_world carries the fields since round 3; the legacy tests strip them
        w = {**w, "drops": {k: v for k, v in w["drops"].items() if k not in ro_drops.drop_fields()}}
    return w


def add_player(w, i=0, pos=POS):
    p = dict(w["players"])
    p["pos"] = p["pos"].at[i].set(jnp.array(pos, jnp.int32))
    p["active"] = p["active"].at[i].set(True)
    p["dead"] = p["dead"].at[i].set(False)
    return {**w, "players": p}


def set_drop(w, slot, item, count=1, refine=0, cards=(0, 0, 0, 0), pos=POS):
    d = dict(w["drops"])
    d["pos"] = d["pos"].at[slot].set(jnp.array(pos, jnp.int32))
    d["item"] = d["item"].at[slot].set(item)
    d["active"] = d["active"].at[slot].set(True)
    d["ttl"] = d["ttl"].at[slot].set(100)
    if "count" in d:
        d["count"] = d["count"].at[slot].set(count)
        d["refine"] = d["refine"].at[slot].set(refine)
        d["cards"] = d["cards"].at[slot].set(jnp.array(cards, jnp.int32))
    return {**w, "drops": d}


def put(w, slot, item, count, refine=0, cards=(0, 0, 0, 0), p=0):
    pl, it = dict(w["players"]), dict(w["items"])
    pl["inv"] = pl["inv"].at[p, slot].set(item)
    it["inv_count"] = it["inv_count"].at[p, slot].set(count)
    it["inv_refine"] = it["inv_refine"].at[p, slot].set(refine)
    it["inv_cards"] = it["inv_cards"].at[p, slot].set(jnp.array(cards, jnp.int32))
    return {**w, "players": pl, "items": it}


def drop_input(slot):
    inp = {**empty_inputs(), **ro_items.empty_item_inputs()}
    inp["drop"] = inp["drop"].at[0].set(slot)
    return inp


def test_table_for_client_card_chance_and_drop_fields_shapes():
    for row in D.table_for_client():
        assert row["drops"][-1]["chance"] == 1
    f = ro_drops.drop_fields()
    assert set(f) == {"count", "refine", "cards"}
    assert f["count"].shape == f["refine"].shape == (MAX_DROPS,)
    assert f["cards"].shape == (MAX_DROPS, I.MAX_CARDS)
    assert all(v.dtype == jnp.int32 and not bool(v.any()) for v in f.values())


def test_card_chance_is_exactly_one_per_10000_for_every_kind():
    for k in range(D.N_DROP_KINDS):
        row = [(int(D.DROP_ITEM[k, j]), int(D.DROP_CHANCE[k, j])) for j in range(D.DROPS_PER_KIND) if int(D.DROP_ITEM[k, j])]
        cards = [c for i, c in row if int(I.KIND[i]) == I.KIND_CARD]
        assert cards == [1], k
    assert int(D.DROP_CHANCE.max()) <= 10000


def test_roll_chances_per_10000_within_three_sigma():
    roll = jax.jit(ro_drops.roll_drops)
    for kind in (0, 1, 10):  # Poring, Orc Warrior, Zombie
        w = make_world(False)
        w = {**w, "monsters": {**w["monsters"], "kind": jnp.full((MAX_MONSTERS,), kind, jnp.int32)}}
        items = np.concatenate([np.asarray(roll(w, jax.random.PRNGKey(7 + k))[0]) for k in range(8)])
        n = items.shape[0]
        for j in range(D.DROPS_PER_KIND):
            want = int(D.DROP_ITEM[kind, j])
            if want == 0:
                continue
            p = int(D.DROP_CHANCE[kind, j]) / 10000.0
            hits = int((items[:, j] == want).sum())
            assert abs(hits - p * n) <= 3 * np.sqrt(n * p * (1 - p)) + 1, (kind, j, hits, p * n)


def test_coin_drop_carries_rolled_zeny_and_pickup_gives_exactly_that():
    w = add_player(make_world())
    m = dict(w["monsters"])
    m["died"] = m["died"].at[3].set(True)
    m["pos"] = m["pos"].at[3].set(jnp.array(POS, jnp.int32))
    w = {**w, "monsters": m}
    items = jnp.zeros((MAX_MONSTERS, D.DROPS_PER_KIND), jnp.int32)
    zeny = jnp.zeros((MAX_MONSTERS,), jnp.int32).at[3].set(73)
    w = ro_drops.place_drops(w, items, items, zeny)
    assert int(w["drops"]["active"].sum()) == 1
    assert int(w["drops"]["item"][0]) == ITEM_GOLD_COIN and int(w["drops"]["count"][0]) == 73
    w, picked = ro_drops.pickup(w)
    assert bool(picked[0]) and int(w["items"]["inv_count"][0, 0]) == 73
    w = ro_items._normalize(w)
    assert int(w["items"]["zeny"][0]) == 73
    assert int(w["players"]["inv"][0, 0]) == ITEM_NONE


def test_refined_slotted_sword_survives_drop_and_pickup():
    cards = (I.ITEM_PORING_CARD, I.ITEM_FABRE_CARD, 0, 0)
    w = put(add_player(make_world()), 2, ITEM_SWORD, 1, refine=5, cards=cards)
    out = jax.jit(ro_items.phase)(w, drop_input(2))
    d = out["drops"]
    i = int(jnp.argmax(d["active"]))
    assert int(d["item"][i]) == ITEM_SWORD and int(d["refine"][i]) == 5 and int(d["count"][i]) == 1
    assert d["cards"][i].tolist() == list(cards)
    assert int(out["players"]["inv"][0, 2]) == ITEM_NONE
    back, picked = ro_drops.pickup(out)
    assert bool(picked[0])
    slot = int(jnp.argmax(back["players"]["inv"][0] == ITEM_SWORD))
    assert int(back["items"]["inv_refine"][0, slot]) == 5
    assert back["items"]["inv_cards"][0, slot].tolist() == list(cards)
    assert not bool(back["drops"]["active"].any()) and not bool(back["drops"]["cards"].any())
    assert int(back["drops"]["refine"].sum()) == 0 and int(back["drops"]["count"].sum()) == 0


def test_dropping_a_stack_of_four_potions_leaves_one_on_the_ground():
    w = put(add_player(make_world()), 1, I.ITEM_RED_POTION, 4)
    out = jax.jit(ro_items.phase)(w, drop_input(1))
    i = int(jnp.argmax(out["drops"]["active"]))
    assert int(out["drops"]["count"][i]) == 1
    assert int(out["items"]["inv_count"][0, 1]) == 3


def test_legacy_drops_block_without_fields_still_works():
    w = put(add_player(make_world(fields=False)), 1, I.ITEM_RED_POTION, 4)
    assert "count" not in w["drops"]
    out = jax.jit(ro_items.phase)(w, drop_input(1))
    assert "count" not in out["drops"] and int(out["drops"]["active"].sum()) == 1
    assert int(out["items"]["inv_count"][0, 1]) == 3
    w2 = add_player(make_world(fields=False))
    m = dict(w2["monsters"])
    m["died"] = m["died"].at[0].set(True)
    w2 = {**w2, "monsters": m}
    items = jnp.zeros((MAX_MONSTERS, D.DROPS_PER_KIND), jnp.int32).at[0, 0].set(I.ITEM_JELLOPY)
    placed = ro_drops.place_drops(w2, items, jnp.where(items > 0, 9, 0), jnp.zeros((MAX_MONSTERS,), jnp.int32))
    assert int(placed["drops"]["active"].sum()) == 1 and "count" not in placed["drops"]
    placed = {**placed, "drops": {**placed["drops"], "pos": placed["drops"]["pos"].at[0].set(jnp.array(POS, jnp.int32))}}
    got, picked = ro_drops.pickup(placed)
    assert bool(picked[0]) and int(got["players"]["inv"][0, 0]) == I.ITEM_JELLOPY
    assert int(got["items"]["inv_count"][0, 0]) == 1 and int(got["items"]["inv_refine"][0, 0]) == 0


def test_pickup_jit_and_vmap():
    w = set_drop(add_player(make_world()), 0, I.ITEM_JELLOPY, count=5)
    out, picked = jax.jit(ro_drops.pickup)(w)
    assert bool(picked[0]) and int(out["items"]["inv_count"][0, 0]) == 5
    assert picked.shape == (MAX_PLAYERS,)
    batch = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), w, make_world(seed=1))
    outs, picks = jax.vmap(ro_drops.pickup)(batch)
    assert picks.tolist()[0][0] and not picks[1].any()
    assert int(outs["items"]["inv_count"][0, 0, 0]) == 5
