import jax
import jax.numpy as jnp

from game.config import DROP_TTL, INV_SLOTS, ITEM_GOLD_COIN, ITEM_MEAT, ITEM_NONE, ITEM_SWORD, MAX_DROPS
from game.data import drops as drop_data
from game.data import items as item_data
from game.data import monsters as monster_data
from game.data.drops import DROP_CHANCE, DROP_ITEM
from game.data.items import ITEM_JELLOPY, ITEM_TIGER_SKIN
from game.loot import loot
from game.ro_drops import roll_drops
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


def _expected_drops(w, m):
    """The drop table rolls for monster m with the loot phase's key: (items in entry order, coin or 0)."""
    sub, _ = next_key(w)
    items, _, zeny = roll_drops(w, sub)
    got = [int(i) for i in items[m] if int(i) != ITEM_NONE]
    return got, (ITEM_GOLD_COIN if int(zeny[m]) > 0 else ITEM_NONE), int(zeny[m])


def set_kind(w, idxs, kind):
    m = w["monsters"]
    return {**w, "monsters": {**m, "kind": m["kind"].at[jnp.array(idxs)].set(kind)}}


def credit(w, m, player):
    mvp = w["mvp"]
    return {**w, "mvp": {**mvp, "top_damager": mvp["top_damager"].at[m].set(player)}}


def test_drop_table_covers_every_monster_kind_with_a_card_at_one_per_mille():
    assert drop_data.N_DROP_KINDS == monster_data.N_KINDS == 20
    assert DROP_ITEM.shape == (20, drop_data.DROPS_PER_KIND)
    for k in range(20):
        row = [int(i) for i in DROP_ITEM[k] if int(i)]
        assert row, k
        cards = [i for i in row if int(item_data.KIND[i]) == item_data.KIND_CARD]
        assert len(cards) == 1 and int(DROP_CHANCE[k][row.index(cards[0])]) == 1, k
        assert 0 < int(drop_data.ZENY_MIN[k]) <= int(drop_data.ZENY_MAX[k])
    assert ((DROP_CHANCE >= 0) & (DROP_CHANCE <= 1000)).all()
    assert len(drop_data.table_for_client()) == 20


def test_died_monster_creates_table_drops_at_pos():
    seen = set()
    for seed in range(12):
        w = set_died(world_with(seed), [5], [[10, 12]])
        items, coin, _ = _expected_drops(w, 5)
        out = loot(w)  # nobody credited: zeny stays on the ground as a coin drop
        d = out["drops"]
        want = items + ([coin] if coin else [])
        seen.update(want)
        n = len(want)
        assert int(d["active"].sum()) == n
        assert d["active"][:n].all()
        assert [int(x) for x in d["item"][:n]] == want
        assert (d["pos"][:n] == jnp.array([10, 12])).all()
        assert (d["ttl"][:n] == DROP_TTL - 1).all()
    assert ITEM_JELLOPY in seen and ITEM_GOLD_COIN in seen  # kind 0 is the Poring row


def test_zeny_goes_to_the_credited_killer_not_the_ground():
    w = set_player(set_died(world_with(4), [7], [[30, 30]]), 2, [90, 90])
    w = credit(w, 7, 2)
    items, _, zeny = _expected_drops(w, 7)
    out = loot(w)
    assert int(out["items"]["zeny"][2]) == zeny > 0
    assert int(out["items"]["zeny"].sum()) == zeny
    assert int((out["drops"]["item"] == ITEM_GOLD_COIN).sum()) == 0
    assert int(out["drops"]["active"].sum()) == len(items)


def test_killer_who_logged_out_gets_nothing_and_coins_drop():
    w = credit(set_died(world_with(4), [7], [[30, 30]]), 7, 2)  # player 2 not active
    out = loot(w)
    assert int(out["items"]["zeny"].sum()) == 0
    assert int((out["drops"]["item"] == ITEM_GOLD_COIN).sum()) == 1


def test_zeny_from_two_kills_adds_up_and_caps():
    w = set_player(set_died(world_with(6), [3, 9], [[5, 5], [6, 6]]), 0, [50, 50])
    w = credit(credit(w, 3, 0), 9, 0)
    _, _, z3 = _expected_drops(w, 3)
    _, _, z9 = _expected_drops(w, 9)
    assert int(loot(w)["items"]["zeny"][0]) == z3 + z9
    it = w["items"]
    w = {**w, "items": {**it, "zeny": it["zeny"].at[0].set(item_data.MAX_ZENY - 1)}}
    assert int(loot(w)["items"]["zeny"][0]) == item_data.MAX_ZENY


def test_kind_selects_the_row():
    # MVP Eddga (kind 19): only its own row's items (or a coin) ever drop, and its tiger skin does.
    row = {int(i) for i in DROP_ITEM[19]} | {ITEM_GOLD_COIN}
    seen = set()
    for seed in range(8):
        w = set_kind(set_died(world_with(seed), [4], [[8, 8]]), [4], 19)
        out = loot(w)
        seen |= {int(x) for x in out["drops"]["item"][out["drops"]["active"]]}
    assert seen <= row and ITEM_TIGER_SKIN in seen


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
    # expected candidates in rank order: monster-major, table entries then the coin
    spawned = []
    for m in idxs:
        items, coin, _ = _expected_drops(w, m)
        spawned += [(m, i) for i in items + ([coin] if coin else [])]
    out = loot(w)
    d = out["drops"]
    assert int(d["active"].sum()) == len(spawned) + 2
    assert bool(d["active"][0]) and bool(d["active"][2])
    assert d["pos"][0].tolist() == [200, 200] and d["pos"][2].tolist() == [201, 200]
    free_slots = [s for s in range(MAX_DROPS) if s not in (0, 2)][: len(spawned)]
    for slot, (m, item) in zip(free_slots, spawned):
        assert d["pos"][slot].tolist() == positions[m]
        assert int(d["item"][slot]) == item


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
