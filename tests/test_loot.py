import jax
import jax.numpy as jnp

from game.config import (
    DROP_TTL, INV_SLOTS, ITEM_GOLD_COIN, ITEM_MEAT, ITEM_NONE, ITEM_SWORD,
    LOOT_COIN_MAX, LOOT_MEAT_MAX, LOOT_SWORD_MAX, MAX_DROPS, MAX_MONSTERS,
)
from game.loot import loot, roll_item
from game.state import make_empty_world, next_key


def world_with(seed=0):
    return make_empty_world(jax.random.PRNGKey(seed))


def set_drop(w, i, pos, item, ttl=DROP_TTL):
    d = w["drops"]
    d = {
        "pos": d["pos"].at[i].set(jnp.array(pos, jnp.int32)),
        "item": d["item"].at[i].set(item),
        "active": d["active"].at[i].set(True),
        "ttl": d["ttl"].at[i].set(ttl),
    }
    return {**w, "drops": d}


def set_player(w, i, pos, dead=False):
    p = dict(w["players"])
    p["pos"] = p["pos"].at[i].set(jnp.array(pos, jnp.int32))
    p["active"] = p["active"].at[i].set(True)
    p["dead"] = p["dead"].at[i].set(dead)
    return {**w, "players": p}


def set_died(w, idxs, positions):
    m = dict(w["monsters"])
    idxs = jnp.array(idxs)
    m["died"] = m["died"].at[idxs].set(True)
    m["pos"] = m["pos"].at[idxs].set(jnp.array(positions, jnp.int32))
    return {**w, "monsters": m}


def test_roll_table():
    # Just below and at each config threshold, so the table follows the constants.
    eps = 1e-3
    u = jnp.array([0.0, LOOT_COIN_MAX - eps, LOOT_COIN_MAX, LOOT_MEAT_MAX - eps,
                   LOOT_MEAT_MAX, LOOT_SWORD_MAX - eps, LOOT_SWORD_MAX, 0.999])
    assert roll_item(u).tolist() == [
        ITEM_GOLD_COIN, ITEM_GOLD_COIN, ITEM_MEAT, ITEM_MEAT,
        ITEM_SWORD, ITEM_SWORD, ITEM_NONE, ITEM_NONE,
    ]


def test_died_monster_creates_drop_at_pos():
    seen = set()
    for seed in range(12):
        w = set_died(world_with(seed), [5], [[10, 12]])
        sub, _ = next_key(w)
        expected = int(roll_item(jax.random.uniform(sub, (MAX_MONSTERS,)))[5])
        out = loot(w)
        d = out["drops"]
        seen.add(expected)
        if expected == ITEM_NONE:
            assert int(d["active"].sum()) == 0
        else:
            assert int(d["active"].sum()) == 1
            i = int(jnp.argmax(d["active"]))
            assert i == 0
            assert d["pos"][i].tolist() == [10, 12]
            assert int(d["item"][i]) == expected
            assert int(d["ttl"][i]) == DROP_TTL - 1
    assert len(seen) > 1


def test_no_died_no_drop():
    out = loot(world_with())
    assert int(out["drops"]["active"].sum()) == 0


def test_pickup_first_empty_slot():
    w = set_drop(world_with(), 3, [7, 8], ITEM_MEAT)
    w = set_player(w, 2, [7, 8])
    inv = w["players"]["inv"].at[2, 0].set(ITEM_SWORD).at[2, 1].set(ITEM_GOLD_COIN)
    w = {**w, "players": {**w["players"], "inv": inv}}
    out = loot(w)
    assert int(out["players"]["inv"][2, 2]) == ITEM_MEAT
    assert int(out["players"]["inv"][2, 0]) == ITEM_SWORD
    assert not bool(out["drops"]["active"][3])


def test_dead_player_does_not_pick_up():
    w = set_drop(world_with(), 0, [7, 8], ITEM_MEAT)
    w = set_player(w, 0, [7, 8], dead=True)
    out = loot(w)
    assert bool(out["drops"]["active"][0])
    assert int(out["players"]["inv"][0, 0]) == ITEM_NONE


def test_lowest_drop_index_wins_on_shared_tile():
    w = set_drop(world_with(), 9, [4, 4], ITEM_SWORD)
    w = set_drop(w, 2, [4, 4], ITEM_MEAT)
    w = set_player(w, 0, [4, 4])
    out = loot(w)
    assert int(out["players"]["inv"][0, 0]) == ITEM_MEAT
    assert not bool(out["drops"]["active"][2])
    assert bool(out["drops"]["active"][9])


def test_two_players_one_drop_lowest_player_wins():
    w = set_drop(world_with(), 0, [4, 4], ITEM_MEAT)
    w = set_player(w, 1, [4, 4])
    w = set_player(w, 3, [4, 4])
    out = loot(w)
    assert int(out["players"]["inv"][1, 0]) == ITEM_MEAT
    assert int(out["players"]["inv"][3].sum()) == 0


def test_full_inventory_leaves_drop():
    w = set_drop(world_with(), 0, [7, 8], ITEM_MEAT)
    w = set_player(w, 0, [7, 8])
    inv = jnp.full_like(w["players"]["inv"], ITEM_NONE).at[0].set(ITEM_GOLD_COIN)
    w = {**w, "players": {**w["players"], "inv": inv}}
    out = loot(w)
    assert bool(out["drops"]["active"][0])
    assert int(out["drops"]["item"][0]) == ITEM_MEAT
    assert (out["players"]["inv"][0] == ITEM_GOLD_COIN).all()
    assert out["players"]["inv"].shape[1] == INV_SLOTS


def test_ttl_expiry():
    w = set_drop(world_with(), 0, [20, 20], ITEM_MEAT, ttl=2)
    w1 = loot(w)
    assert bool(w1["drops"]["active"][0]) and int(w1["drops"]["ttl"][0]) == 1
    w2 = loot(w1)
    assert not bool(w2["drops"]["active"][0])
    assert int(w2["drops"]["item"][0]) == ITEM_NONE


def test_several_died_get_distinct_slots():
    idxs = list(range(0, 40))
    positions = [[i + 1, 2] for i in idxs]
    w = set_died(world_with(1), idxs, positions)
    # occupy slots 0 and 2 so free slots are not contiguous
    w = set_drop(w, 0, [200, 200], ITEM_MEAT)
    w = set_drop(w, 2, [201, 200], ITEM_MEAT)
    sub, _ = next_key(w)
    items = roll_item(jax.random.uniform(sub, (MAX_MONSTERS,)))[:40]
    n_expected = int((items != ITEM_NONE).sum())
    out = loot(w)
    d = out["drops"]
    assert int(d["active"].sum()) == n_expected + 2
    assert bool(d["active"][0]) and bool(d["active"][2])
    assert d["pos"][0].tolist() == [200, 200] and d["pos"][2].tolist() == [201, 200]
    # each spawned monster's position appears exactly once, with the rolled item
    spawned = [i for i in idxs if int(items[i]) != ITEM_NONE]
    free_slots = [s for s in range(MAX_DROPS) if s not in (0, 2)][: len(spawned)]
    for slot, m in zip(free_slots, spawned):
        assert d["pos"][slot].tolist() == positions[m]
        assert int(d["item"][slot]) == int(items[m])


def test_extra_drops_discarded_when_full():
    w = world_with(3)
    d = w["drops"]
    d = {**d, "active": d["active"].at[: MAX_DROPS - 1].set(True),
         "ttl": d["ttl"].at[:].set(50), "pos": d["pos"].at[:].set(100)}
    w = {**w, "drops": d}
    w = set_died(w, list(range(60)), [[1, 1]] * 60)
    out = loot(w)
    assert int(out["drops"]["active"].sum()) <= MAX_DROPS
    assert int(out["drops"]["active"][: MAX_DROPS - 1].sum()) == MAX_DROPS - 1


def test_jit_and_vmap():
    ws = [set_died(world_with(s), [1, 2, 3], [[5, 5], [6, 5], [7, 5]]) for s in (0, 1)]
    ws[1] = set_drop(set_player(ws[1], 0, [30, 30]), 4, [30, 30], ITEM_SWORD)
    batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *ws)
    out = jax.jit(jax.vmap(loot))(batch)
    for b in range(2):
        single = jax.jit(loot)(ws[b])
        for a, c in zip(jax.tree_util.tree_leaves(single),
                        jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x[b], out))):
            assert (a == c).all()
    assert int(out["players"]["inv"][1, 0, 0]) == ITEM_SWORD
