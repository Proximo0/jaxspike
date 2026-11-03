import jax
import jax.numpy as jnp
import numpy as np

from game import ro_items, ro_shop, world_layout
from game.config import MAX_PLAYERS
from game.data import items as I
from game.data import skills as S
from game.data.terrain import T_SHOP
from game.state import empty_inputs, make_empty_world

SHOP_TILES, SHOP_TOWN = world_layout.shop_tiles()
_phase = jax.jit(ro_shop.phase)


def town_shop(t):
    return [int(v) for v in SHOP_TILES[np.flatnonzero(SHOP_TOWN == t)[0]]]


def make_world(towns=(0,)):
    w = make_empty_world(jax.random.PRNGKey(0))
    terrain = w["terrain"]
    for t in towns:
        x, y = town_shop(t)
        terrain = terrain.at[x, y].set(T_SHOP)
    return {**w, "terrain": terrain}


def add_player(w, i, pos, zeny=0, dead=False):
    p = dict(w["players"])
    p["pos"] = p["pos"].at[i].set(jnp.array(pos, jnp.int32))
    p["active"] = p["active"].at[i].set(True)
    p["dead"] = p["dead"].at[i].set(dead)
    it = {**w["items"], "zeny": w["items"]["zeny"].at[i].set(zeny)}
    return {**w, "players": p, "items": it}


def put(w, p, slot, item, count=1):
    pl = {**w["players"], "inv": w["players"]["inv"].at[p, slot].set(item)}
    it = {**w["items"], "inv_count": w["items"]["inv_count"].at[p, slot].set(count)}
    return {**w, "players": pl, "items": it}


def run(w, buy=None, sell=None):
    inp = {**empty_inputs(), **ro_shop.empty_shop_inputs()}
    if buy is not None:
        inp["buy"] = inp["buy"].at[0].set(buy)
    if sell is not None:
        inp["sell"] = inp["sell"].at[0].set(sell)
    return _phase(w, inp)


def zeny(w, p=0):
    return int(w["items"]["zeny"][p])


def slots(w, p=0):
    return np.asarray(w["players"]["inv"][p]).tolist(), np.asarray(w["items"]["inv_count"][p]).tolist()


def test_stock_tables_cover_every_town_and_have_prices():
    assert ro_shop.SHOP_STOCK.shape == (len(world_layout.TOWNS), ro_shop.MAX_STOCK)
    assert not ro_shop.SHOP_FALLBACK_KAFRA
    for t, row in enumerate(ro_shop.stock_for_client()):
        assert row["name"] == world_layout.TOWN_NAMES[t] and I.ITEM_RED_POTION in row["items"]
    assert I.ITEM_BOW in ro_shop.stock_for_client()[2]["items"]  # Payon sells bows
    assert I.ITEM_BOW not in ro_shop.stock_for_client()[0]["items"]


def test_buy_red_potion_on_and_next_to_a_shop_tile():
    x, y = town_shop(0)
    for pos in ([x, y], [x + 1, y + 1]):
        w = add_player(make_world(), 0, pos, zeny=100)
        out = run(w, buy=I.ITEM_RED_POTION)
        assert zeny(out) == 100 - int(I.BUY[I.ITEM_RED_POTION])
        inv, cnt = slots(out)
        assert inv[0] == I.ITEM_RED_POTION and cnt[0] == 1


def test_buy_refused_away_from_shop_without_zeny_or_out_of_stock():
    x, y = town_shop(0)
    w = add_player(make_world(), 0, [x + 2, y], zeny=10_000)  # two tiles away
    assert slots(run(w, buy=I.ITEM_RED_POTION))[0][0] == 0
    w = add_player(make_world(), 0, [x, y], zeny=int(I.BUY[I.ITEM_RED_POTION]) - 1)
    out = run(w, buy=I.ITEM_RED_POTION)
    assert slots(out)[0][0] == 0 and zeny(out) == int(I.BUY[I.ITEM_RED_POTION]) - 1
    w = add_player(make_world(), 0, [x, y], zeny=10_000)
    for bad in (I.ITEM_BOW, I.ITEM_PORING_CARD, 0, -1, 999):  # not in Prontera's stock / none / out of range
        out = run(w, buy=bad)
        assert slots(out)[0][0] == 0 and zeny(out) == 10_000


def test_buy_stacks_onto_existing_stack_then_needs_a_free_slot_for_gear():
    x, y = town_shop(0)
    w = put(add_player(make_world(), 0, [x, y], zeny=10_000), 0, 5, I.ITEM_RED_POTION, count=3)
    inv, cnt = slots(run(w, buy=I.ITEM_RED_POTION))
    assert inv[5] == I.ITEM_RED_POTION and cnt[5] == 4 and inv[0] == 0
    full = w
    for s in range(16):
        full = put(full, 0, s, I.ITEM_JELLOPY if s != 5 else I.ITEM_RED_POTION, count=1)
    out = run(full, buy=I.ITEM_KNIFE)  # no room for a non-stackable
    assert zeny(out) == 10_000 and I.ITEM_KNIFE not in slots(out)[0]
    out = run(full, buy=I.ITEM_RED_POTION)  # the potion still stacks
    assert slots(out)[1][5] == 2


def test_dead_or_outside_town_cannot_trade():
    x, y = town_shop(0)
    w = add_player(make_world(), 0, [x, y], zeny=1000, dead=True)
    assert zeny(run(w, buy=I.ITEM_RED_POTION)) == 1000
    w = make_empty_world(jax.random.PRNGKey(0))
    w = {**w, "terrain": w["terrain"].at[3, 3].set(T_SHOP)}  # a shop tile outside every town box
    w = add_player(w, 0, [3, 3], zeny=1000)
    assert zeny(run(w, buy=I.ITEM_RED_POTION)) == 1000


def test_sell_one_unit_of_a_stack_and_whole_gear():
    x, y = town_shop(0)
    w = put(add_player(make_world(), 0, [x, y]), 0, 2, I.ITEM_JELLOPY, count=5)
    w = put(w, 0, 3, I.ITEM_KNIFE)
    out = run(w, sell=2)
    assert zeny(out) == int(I.SELL[I.ITEM_JELLOPY]) and slots(out)[1][2] == 4
    out = run(out, sell=3)
    assert zeny(out) == int(I.SELL[I.ITEM_JELLOPY]) + int(I.SELL[I.ITEM_KNIFE]) and slots(out)[0][3] == 0


def test_sell_refuses_worn_items_and_empty_slots():
    x, y = town_shop(0)
    w = put(add_player(make_world(), 0, [x, y]), 0, 0, I.ITEM_KNIFE)
    w = {**w, "items": {**w["items"], "equip": w["items"]["equip"].at[0, I.EQ_WEAPON].set(0)}}
    out = run(w, sell=0)
    assert zeny(out) == 0 and slots(out)[0][0] == I.ITEM_KNIFE
    assert zeny(run(w, sell=7)) == 0 and zeny(run(w, sell=99)) == 0


def test_sell_then_buy_in_one_tick():
    x, y = town_shop(0)
    w = put(add_player(make_world(), 0, [x, y]), 0, 0, I.ITEM_MAIN_GAUCHE)  # pays for a red potion
    assert int(I.SELL[I.ITEM_MAIN_GAUCHE]) >= int(I.BUY[I.ITEM_RED_POTION])
    out = run(w, sell=0, buy=I.ITEM_RED_POTION)
    assert slots(out)[0][0] == I.ITEM_RED_POTION
    assert zeny(out) == int(I.SELL[I.ITEM_MAIN_GAUCHE]) - int(I.BUY[I.ITEM_RED_POTION])


def test_discount_and_overcharge_follow_classic_percent():
    x, y = town_shop(0)
    w = add_player(make_world(), 0, [x, y], zeny=100_000)
    sk = w["skills"]
    w = {**w, "skills": {**sk, "learned": sk["learned"].at[0, S.SK_DISCOUNT].set(10)}}
    assert int(ro_shop.discount_pct(w)[0]) == 24 and int(ro_shop.discount_pct(w)[1]) == 0
    out = run(w, buy=I.ITEM_SWORD)
    assert zeny(out) == 100_000 - int(I.BUY[I.ITEM_SWORD]) * 76 // 100
    w1 = {**w, "skills": {**sk, "learned": sk["learned"].at[0, S.SK_DISCOUNT].set(1)}}
    assert int(ro_shop.discount_pct(w1)[0]) == 7
    if ro_shop.SK_OVERCHARGE is not None:
        w2 = {**w, "skills": {**sk, "learned": sk["learned"].at[0, ro_shop.SK_OVERCHARGE].set(5)}}
        assert int(ro_shop.overcharge_pct(w2)[0]) == 15
        w2 = put(w2, 0, 0, I.ITEM_MAIN_GAUCHE)
        assert zeny(run(w2, sell=0)) == 100_000 + int(I.SELL[I.ITEM_MAIN_GAUCHE]) * 115 // 100


def test_zeny_capped_on_sell():
    x, y = town_shop(0)
    w = put(add_player(make_world(), 0, [x, y], zeny=I.MAX_ZENY - 1), 0, 0, I.ITEM_KNIFE)
    assert zeny(run(w, sell=0)) == I.MAX_ZENY


def test_other_town_stock_and_vmap_matches_single():
    x, y = town_shop(2)
    w = add_player(make_world((0, 2)), 0, [x, y], zeny=5000)
    out = run(w, buy=I.ITEM_BOW)  # Payon
    assert slots(out)[0][0] == I.ITEM_BOW
    a, b = w, add_player(make_world((0, 2)), 0, town_shop(0), zeny=5000)
    inp = {**empty_inputs(), **ro_shop.empty_shop_inputs()}
    inp = {**inp, "buy": inp["buy"].at[0].set(I.ITEM_BOW)}
    batch = jax.tree_util.tree_map(lambda p, q: jnp.stack([p, q]), a, b)
    ib = jax.tree_util.tree_map(lambda p: jnp.stack([p, p]), inp)
    got = jax.jit(jax.vmap(ro_shop.phase))(batch, ib)
    assert int(got["players"]["inv"][0, 0, 0]) == I.ITEM_BOW and int(got["players"]["inv"][1, 0, 0]) == 0
    assert got["items"]["zeny"].shape == (2, MAX_PLAYERS)


def test_items_phase_after_shop_keeps_arrays_consistent():
    x, y = town_shop(0)
    w = add_player(make_world(), 0, [x, y], zeny=1000)
    out = run(w, buy=I.ITEM_KNIFE)
    out = jax.jit(ro_items.phase)(out, {**empty_inputs(), **ro_items.empty_item_inputs()})
    inv, cnt = slots(out)
    assert inv[0] == I.ITEM_KNIFE and cnt[0] == 1
