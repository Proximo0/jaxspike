"""M3 items: table, inventory normalize, use/equip/drop, equipped stats, refine, cards, drop rolls, bot policy."""
import jax
import jax.numpy as jnp
import numpy as np

from bots.ro_items_policy import item_inputs
from game import ro_drops, ro_items, ro_refine, ro_stats
from game.config import (DROP_TTL, HUNT_RETREAT_HP, INV_SLOTS, ITEM_GOLD_COIN, ITEM_MEAT, ITEM_NONE, ITEM_SWORD,
                         MAX_DROPS, MAX_MONSTERS, MAX_PLAYERS, PLAYER_MAX_HP)
from game.data import drops as D
from game.data import items as I
from game.data.stats import JOB_ARCHER, JOB_MAGE, JOB_NOVICE, JOB_SWORDSMAN, JOB_THIEF
from game.state import empty_inputs, make_empty_world

_phase = jax.jit(ro_items.phase)


def make_world(seed=0):
    w = make_empty_world(jax.random.PRNGKey(seed))
    return {**w, "items": ro_items.make_state()}


def add_player(w, i, pos=(10, 12), hp=PLAYER_MAX_HP, dead=False):
    p = dict(w["players"])
    p["pos"] = p["pos"].at[i].set(jnp.array(pos, jnp.int32))
    p["active"] = p["active"].at[i].set(True)
    p["dead"] = p["dead"].at[i].set(dead)
    p["hp"] = p["hp"].at[i].set(hp)
    return {**w, "players": p}


def set_job(w, p, job):
    st = dict(w["stats"])
    st["job"] = st["job"].at[p].set(job)
    return {**w, "stats": st}


def put(w, p, slot, item, count=None, refine=0, cards=None):
    """Place an item in a slot. count=None leaves the count 0, like a fresh pickup from loot.py."""
    pl, it = dict(w["players"]), dict(w["items"])
    pl["inv"] = pl["inv"].at[p, slot].set(item)
    if count is not None:
        it["inv_count"] = it["inv_count"].at[p, slot].set(count)
    it["inv_refine"] = it["inv_refine"].at[p, slot].set(refine)
    if cards is not None:
        it["inv_cards"] = it["inv_cards"].at[p, slot].set(jnp.array(cards, jnp.int32))
    return {**w, "players": pl, "items": it}


def wear(w, p, eq_slot, inv_slot):
    it = dict(w["items"])
    it["equip"] = it["equip"].at[p, eq_slot].set(inv_slot)
    return {**w, "items": it}


def set_zeny(w, p, z):
    return {**w, "items": {**w["items"], "zeny": w["items"]["zeny"].at[p].set(z)}}


def inputs_for(**kw):
    inp = {**empty_inputs(), **ro_items.empty_item_inputs()}
    for k, v in kw.items():
        inp[k] = inp[k].at[0].set(v)
    return inp


def run(w, **kw):
    return _phase(w, inputs_for(**kw))


def inv(w, p=0):
    return np.asarray(w["players"]["inv"][p])


def count(w, p=0):
    return np.asarray(w["items"]["inv_count"][p])


def equip(w, p=0):
    return np.asarray(w["items"]["equip"][p])


def cards_of(w, slot, p=0):
    return np.asarray(w["items"]["inv_cards"][p, slot]).tolist()


# --- table -----------------------------------------------------------------------------------------

def test_table_ids_and_shape():
    assert I.N_ITEMS == 300 and I.N_BASE_ITEMS <= I.CRAFT_ID_BASE == 100
    assert (I.ITEM_NONE, I.ITEM_GOLD_COIN, I.ITEM_MEAT, I.ITEM_SWORD) == (ITEM_NONE, ITEM_GOLD_COIN, ITEM_MEAT, ITEM_SWORD)
    assert (ITEM_NONE, ITEM_GOLD_COIN, ITEM_MEAT, ITEM_SWORD) == (0, 1, 2, 3)
    assert len(I.NAMES) == I.N_ITEMS
    for name, col in I.ITEM_TABLE.items():
        assert col.shape == (I.N_ITEMS,), name
    assert I.KIND[I.ITEM_GOLD_COIN] == I.KIND_ETC and bool(I.STACKABLE[I.ITEM_GOLD_COIN])
    assert I.KIND[I.ITEM_MEAT] == I.KIND_CONSUMABLE and not bool(I.STACKABLE[I.ITEM_MEAT])
    assert I.KIND[I.ITEM_SWORD] == I.KIND_WEAPON and int(I.ATK[I.ITEM_SWORD]) == 25 and int(I.SLOTS[I.ITEM_SWORD]) == 3
    assert I.KIND[I.ITEM_PORING_CARD] == I.KIND_CARD
    assert I.REFINE_ORE.shape == (5,) and I.REFINE_ZENY.shape == (5,)
    assert len(I.table_for_client()) == I.N_ITEMS


# --- normalize -------------------------------------------------------------------------------------

def test_fresh_pickup_gets_count_one():
    w = put(add_player(make_world(), 0), 0, 4, I.ITEM_JELLOPY)
    out = run(w)
    assert count(out)[4] == 1 and inv(out)[4] == I.ITEM_JELLOPY


def test_coin_becomes_zeny_and_slot_clears():
    w = put(add_player(make_world(), 0), 0, 2, ITEM_GOLD_COIN)
    out = run(w)
    assert int(out["items"]["zeny"][0]) == I.ZENY_PER_COIN
    assert inv(out)[2] == 0 and count(out)[2] == 0
    out = run(put(w, 0, 5, ITEM_GOLD_COIN, count=7))
    assert int(out["items"]["zeny"][0]) == 8 * I.ZENY_PER_COIN


def test_zeny_clamped_to_max():
    w = set_zeny(put(add_player(make_world(), 0), 0, 0, ITEM_GOLD_COIN, count=5), 0, I.MAX_ZENY - 3)
    assert int(run(w)["items"]["zeny"][0]) == I.MAX_ZENY


def test_stacks_merge_into_lowest_slot():
    w = add_player(make_world(), 0)
    w = put(w, 0, 3, I.ITEM_RED_POTION, count=3)
    w = put(w, 0, 6, I.ITEM_RED_POTION, count=4)
    w = put(w, 0, 9, I.ITEM_RED_POTION)  # fresh pickup, count 0 -> 1
    out = run(w)
    assert inv(out)[3] == I.ITEM_RED_POTION and count(out)[3] == 8
    assert inv(out)[6] == 0 and inv(out)[9] == 0 and count(out)[6] == 0 and count(out)[9] == 0


def test_stack_overflow_is_capped():
    w = add_player(make_world(), 0)
    w = put(put(w, 0, 0, I.ITEM_JELLOPY, count=I.MAX_STACK), 0, 1, I.ITEM_JELLOPY, count=50)
    out = run(w)
    assert count(out)[0] == I.MAX_STACK and inv(out)[1] == 0


def test_non_stackables_do_not_merge():
    w = add_player(make_world(), 0)
    w = put(put(w, 0, 0, ITEM_MEAT), 0, 1, ITEM_MEAT)
    out = run(w)
    assert inv(out)[:2].tolist() == [ITEM_MEAT, ITEM_MEAT] and count(out)[:2].tolist() == [1, 1]


def test_equip_pointer_to_emptied_slot_clears():
    w = wear(put(add_player(make_world(), 0), 0, 3, ITEM_SWORD, count=1), 0, I.EQ_WEAPON, 3)
    assert equip(run(w))[I.EQ_WEAPON] == 3
    gone = {**w, "players": {**w["players"], "inv": w["players"]["inv"].at[0, 3].set(0)}}
    assert equip(run(gone))[I.EQ_WEAPON] == -1
    potion = put(w, 0, 3, I.ITEM_RED_POTION, count=1)  # worn slot now holds a non weapon/armor
    assert equip(run(potion))[I.EQ_WEAPON] == -1


# --- use_item --------------------------------------------------------------------------------------

def test_use_potion_heals_decrements_and_clears():
    w = put(add_player(make_world(), 0, hp=1), 0, 1, I.ITEM_RED_POTION, count=2)
    cap = int(ro_stats.derived(w)["max_hp"][0])
    out = run(w, use_item=1)
    assert int(out["players"]["hp"][0]) == min(1 + int(I.HEAL_HP[I.ITEM_RED_POTION]), cap) > 1
    assert count(out)[1] == 1 and inv(out)[1] == I.ITEM_RED_POTION
    out = run(out, use_item=1)
    assert inv(out)[1] == 0 and count(out)[1] == 0


def test_use_potion_caps_at_max_hp():
    w = add_player(make_world(), 0)
    cap = int(ro_stats.derived(w)["max_hp"][0])
    w = put(add_player(w, 0, hp=cap - 5), 0, 0, I.ITEM_WHITE_POTION, count=1)
    assert int(run(w, use_item=0)["players"]["hp"][0]) == cap


def test_use_potion_caps_at_derived_max_hp_for_leveled_player():
    w = put(add_player(make_world(), 0, hp=10), 0, 0, I.ITEM_WHITE_POTION, count=1)
    st = dict(w["stats"])
    st["base_level"] = st["base_level"].at[0].set(40)
    st["vit"] = st["vit"].at[0].set(50)
    w = {**w, "stats": st}
    cap = int(ro_stats.derived(w)["max_hp"][0])
    assert cap > PLAYER_MAX_HP
    w = {**w, "players": {**w["players"], "hp": w["players"]["hp"].at[0].set(cap - 3)}}
    assert int(run(w, use_item=0)["players"]["hp"][0]) == cap


def test_dead_player_cannot_use_and_bad_slots_ignored():
    w = put(add_player(make_world(), 0, hp=0, dead=True), 0, 1, I.ITEM_RED_POTION, count=2)
    out = run(w, use_item=1)
    assert int(out["players"]["hp"][0]) == 0 and count(out)[1] == 2
    alive = put(add_player(make_world(), 0, hp=50), 0, 1, I.ITEM_RED_POTION, count=2)
    for bad in (-1, INV_SLOTS, 99, 5):  # 5 is an empty slot
        out = run(alive, use_item=bad)
        assert int(out["players"]["hp"][0]) == 50 and count(out)[1] == 2


# --- equip -----------------------------------------------------------------------------------------

def test_equip_toggles_weapon():
    w = put(add_player(make_world(), 0), 0, 2, ITEM_SWORD, count=1)
    out = run(w, equip=2)
    assert equip(out)[I.EQ_WEAPON] == 2
    assert equip(run(out, equip=2))[I.EQ_WEAPON] == -1


def test_equip_replaces_previous_occupant():
    w = add_player(make_world(), 0)
    w = put(put(w, 0, 0, ITEM_SWORD, count=1), 0, 1, I.ITEM_BLADE, count=1)
    out = run(run(w, equip=0), equip=1)
    assert equip(out)[I.EQ_WEAPON] == 1


def test_equip_armor_slots_and_non_equipment():
    w = set_job(add_player(make_world(), 0), 0, JOB_SWORDSMAN)  # a Novice may not wear a Shield
    w = put(put(put(w, 0, 0, I.ITEM_HAT, count=1), 0, 1, I.ITEM_SHIELD, count=1), 0, 2, I.ITEM_RED_POTION, count=1)
    out = run(run(run(w, equip=0), equip=1), equip=2)
    e = equip(out)
    assert e[I.EQ_HEAD] == 0 and e[I.EQ_SHIELD] == 1
    assert (e >= 0).sum() == 2  # the potion did nothing


def test_accessories_fill_acc1_then_acc2_then_replace():
    w = add_player(make_world(), 0)
    for s, it in enumerate((I.ITEM_RING, I.ITEM_CLIP, I.ITEM_NECKLACE)):
        w = put(w, 0, s, it, count=1)
    out = run(w, equip=0)
    assert equip(out)[I.EQ_ACC1] == 0 and equip(out)[I.EQ_ACC2] == -1
    out = run(out, equip=1)
    assert equip(out)[I.EQ_ACC1] == 0 and equip(out)[I.EQ_ACC2] == 1
    out = run(out, equip=2)
    assert equip(out)[I.EQ_ACC1] == 0 and equip(out)[I.EQ_ACC2] == 2


def test_dead_player_cannot_equip():
    w = put(add_player(make_world(), 0, dead=True), 0, 0, ITEM_SWORD, count=1)
    assert equip(run(w, equip=0))[I.EQ_WEAPON] == -1


# --- equipped --------------------------------------------------------------------------------------

def test_equipped_zero_when_nothing_worn():
    w = put(add_player(make_world(), 0), 0, 0, ITEM_SWORD, count=1)
    eq = ro_items.equipped(w)
    for k in ("weapon_atk", "weapon_elem", "armor_def", "armor_elem"):
        assert eq[k].shape == (MAX_PLAYERS,) and int(eq[k].sum()) == 0
    assert eq["card_bonus"].shape == (MAX_PLAYERS, I.N_BONUS) and int(jnp.abs(eq["card_bonus"]).sum()) == 0


def test_equipped_weapon_with_refine():
    w = wear(put(add_player(make_world(), 0), 0, 0, ITEM_SWORD, count=1, refine=3), 0, I.EQ_WEAPON, 0)
    eq = ro_items.equipped(w)
    assert int(eq["weapon_atk"][0]) == int(I.ATK[ITEM_SWORD]) + 2 * 3  # REFINE_ATK[weapon level 1] = 2 per step
    assert int(eq["weapon_atk"][1]) == 0


def test_equipped_armor_def_with_refine_and_item_bonus():
    w = add_player(make_world(), 0)
    w = put(w, 0, 0, I.ITEM_HAT, count=1, refine=1)
    w = put(w, 0, 1, I.ITEM_SHIELD, count=1)
    w = put(w, 0, 2, I.ITEM_RING, count=1)
    w = wear(wear(wear(w, 0, I.EQ_HEAD, 0), 0, I.EQ_SHIELD, 1), 0, I.EQ_ACC1, 2)
    eq = ro_items.equipped(w)
    assert int(eq["armor_def"][0]) == int(I.DEF[I.ITEM_HAT]) + 1 + int(I.DEF[I.ITEM_SHIELD]) + int(I.DEF[I.ITEM_RING])
    bonus = np.asarray(eq["card_bonus"][0])
    assert bonus[I.BONUS_STR] == 2 and bonus.sum() == 2


def test_equipped_card_bonus_by_id_and_sockets_beyond_slots_ignored():
    w = add_player(make_world(), 0)
    cards = [I.ITEM_HORNET_CARD, I.ITEM_ANDRE_CARD, I.ITEM_ANDRE_CARD, I.ITEM_ANDRE_CARD]  # sword has 3 sockets
    w = wear(put(w, 0, 0, ITEM_SWORD, count=1, cards=cards), 0, I.EQ_WEAPON, 0)
    bonus = np.asarray(ro_items.equipped(w)["card_bonus"][0])
    assert bonus[I.BONUS_ATK] == int(I.CARD_VALUE[I.ITEM_HORNET_CARD]) + 2 * int(I.CARD_VALUE[I.ITEM_ANDRE_CARD])
    assert bonus.sum() == bonus[I.BONUS_ATK]
    # a card in an item that is not worn contributes nothing
    unworn = put(add_player(make_world(), 0), 0, 0, ITEM_SWORD, count=1, cards=cards)
    assert int(jnp.abs(ro_items.equipped(unworn)["card_bonus"]).sum()) == 0


# --- refine ----------------------------------------------------------------------------------------

def refine_world(level=0, ore_count=2, zeny=100, item=ITEM_SWORD):
    w = add_player(make_world(), 0)
    w = put(w, 0, 0, item, count=1, refine=level)
    if ore_count:
        w = put(w, 0, 4, I.ITEM_PHRACON, count=ore_count)
    return wear(set_zeny(w, 0, zeny), 0, I.EQ_WEAPON, 0)


def test_refine_chance_table():
    c = np.asarray(ro_refine.REFINE_CHANCE)
    assert c.shape == (5, 11) and (c[:, 10] == 0).all()
    for wl, safe in enumerate((4, 7, 6, 5, 4)):
        assert (c[wl, :safe] == 1000).all() and c[wl, safe] < 1000, wl


def test_safe_refine_always_succeeds_and_pays():
    for seed in range(6):
        w = {**refine_world(), "rng": make_world(seed)["rng"]}
        out = run(w, refine=0)
        assert int(out["items"]["inv_refine"][0, 0]) == 1
        assert int(out["items"]["zeny"][0]) == 100 - int(I.REFINE_ZENY[1])
        assert count(out)[4] == 1
        assert equip(out)[I.EQ_WEAPON] == 0


def test_refine_consumes_last_ore_slot():
    out = run(refine_world(ore_count=1), refine=0)
    assert inv(out)[4] == 0 and count(out)[4] == 0 and int(out["items"]["inv_refine"][0, 0]) == 1


def test_refine_needs_ore_zeny_and_headroom():
    for w in (refine_world(ore_count=0), refine_world(zeny=int(I.REFINE_ZENY[1]) - 1), refine_world(level=10)):
        out = run(w, refine=0)
        assert int(out["items"]["inv_refine"][0, 0]) == int(w["items"]["inv_refine"][0, 0])
        assert int(out["items"]["zeny"][0]) == int(w["items"]["zeny"][0])
        assert inv(out)[0] == ITEM_SWORD
    out = run(refine_world(), refine=4)  # an ore slot is not equipment
    assert count(out)[4] == 2


def test_failed_refine_destroys_item_and_clears_equip():
    # from +9 the sword's chance is below 1000, so some seed fails; search for it
    hit = None
    for seed in range(60):
        w = {**refine_world(level=9), "rng": make_world(seed)["rng"]}
        out = run(w, refine=0)
        if inv(out)[0] == 0:
            hit = out
            break
        assert int(out["items"]["inv_refine"][0, 0]) == 10  # the other outcome is a clean +10
    assert hit is not None, "no failing seed in 60 tries"
    assert count(hit)[0] == 0 and int(hit["items"]["inv_refine"][0, 0]) == 0
    assert cards_of(hit, 0) == [0, 0, 0, 0]
    assert equip(hit)[I.EQ_WEAPON] == -1
    assert int(hit["items"]["zeny"][0]) == 100 - int(I.REFINE_ZENY[1]) and count(hit)[4] == 1


# --- compound_card ---------------------------------------------------------------------------------

def card_world(cards_in=2, card=I.ITEM_HORNET_CARD, weapon=ITEM_SWORD):
    w = add_player(make_world(), 0)
    w = put(w, 0, 0, weapon, count=1)
    w = put(w, 0, 1, card, count=cards_in)
    return wear(w, 0, I.EQ_WEAPON, 0)


def test_compound_fills_lowest_free_socket():
    out = run(card_world(), use_item=1)
    assert cards_of(out, 0) == [I.ITEM_HORNET_CARD, 0, 0, 0]
    assert count(out)[1] == 1
    out = run(out, use_item=1)
    assert cards_of(out, 0) == [I.ITEM_HORNET_CARD] * 2 + [0, 0]
    assert inv(out)[1] == 0 and count(out)[1] == 0  # last card consumed


def test_compound_refuses_when_full():
    out = card_world(cards_in=5)
    for _ in range(4):  # the sword has 3 sockets, the fourth use must be refused
        out = run(out, use_item=1)
    assert cards_of(out, 0) == [I.ITEM_HORNET_CARD] * 3 + [0]
    assert count(out)[1] == 2
    full = put(card_world(cards_in=2), 0, 0, ITEM_SWORD, count=1, cards=[I.ITEM_HORNET_CARD] * 3 + [0])
    out = run(full, use_item=1)
    assert count(out)[1] == 2 and cards_of(out, 0) == [I.ITEM_HORNET_CARD] * 3 + [0]


def test_compound_refuses_wrong_target_or_nothing_worn():
    out = run(card_world(card=I.ITEM_PORING_CARD), use_item=1)  # body card, only a weapon worn
    assert count(out)[1] == 2 and cards_of(out, 0) == [0, 0, 0, 0]
    base = card_world()
    unworn = {**base, "items": {**base["items"], "equip": jnp.full((MAX_PLAYERS, 8), -1, jnp.int32)}}
    out = run(unworn, use_item=1)
    assert count(out)[1] == 2 and cards_of(out, 0) == [0, 0, 0, 0]


def test_compound_card_direct_done_flags():
    w = card_world()
    w = {**w, "items": {**w["items"], "inv_count": w["items"]["inv_count"].at[0, 1].set(2)}}
    slot = jnp.full((MAX_PLAYERS,), -1, jnp.int32).at[0].set(1)
    ok = jnp.zeros((MAX_PLAYERS,), jnp.bool_).at[0].set(True)
    out, done = ro_refine.compound_card(w, slot, ok)
    assert bool(done[0]) and int(done.sum()) == 1
    assert int(out["items"]["inv_cards"][0, 0, 0]) == I.ITEM_HORNET_CARD
    out2, done2 = ro_refine.compound_card(w, slot, jnp.zeros((MAX_PLAYERS,), jnp.bool_))
    assert int(done2.sum()) == 0 and int(jnp.abs(out2["items"]["inv_cards"]).sum()) == 0


# --- drop ------------------------------------------------------------------------------------------

def test_drop_stackable_places_one_unit_at_player_tile():
    w = put(add_player(make_world(), 0, pos=(10, 12)), 0, 1, I.ITEM_RED_POTION, count=3)
    out = run(w, drop=1)
    d = out["drops"]
    assert int(d["active"].sum()) == 1
    i = int(jnp.argmax(d["active"]))
    assert d["pos"][i].tolist() == [10, 12] and int(d["item"][i]) == I.ITEM_RED_POTION and int(d["ttl"][i]) == DROP_TTL
    assert count(out)[1] == 2 and inv(out)[1] == I.ITEM_RED_POTION


def test_drop_weapon_whole_and_unequips():
    w = wear(put(add_player(make_world(), 0, pos=(3, 4)), 0, 2, ITEM_SWORD, count=1, refine=2), 0, I.EQ_WEAPON, 2)
    out = run(w, drop=2)
    assert inv(out)[2] == 0 and count(out)[2] == 0 and int(out["items"]["inv_refine"][0, 2]) == 0
    assert equip(out)[I.EQ_WEAPON] == -1
    i = int(jnp.argmax(out["drops"]["active"]))
    assert int(out["drops"]["item"][i]) == ITEM_SWORD and out["drops"]["pos"][i].tolist() == [3, 4]


def test_drop_with_full_drop_array_keeps_item():
    w = put(add_player(make_world(), 0), 0, 1, I.ITEM_RED_POTION, count=3)
    full = {**w, "drops": {**w["drops"], "active": jnp.ones((MAX_DROPS,), jnp.bool_)}}
    out = run(full, drop=1)
    assert count(out)[1] == 3 and int(out["drops"]["active"].sum()) == MAX_DROPS
    assert int(out["drops"]["item"].sum()) == int(full["drops"]["item"].sum())


def test_drop_rank_lowest_player_gets_last_free_slot():
    w = add_player(add_player(make_world(), 0, pos=(1, 1)), 1, pos=(2, 2))
    w = put(put(w, 0, 0, I.ITEM_APPLE, count=1), 1, 0, I.ITEM_APPLE, count=1)
    active = jnp.ones((MAX_DROPS,), jnp.bool_).at[7].set(False)
    w = {**w, "drops": {**w["drops"], "active": active}}
    inp = inputs_for(drop=0)
    inp["drop"] = inp["drop"].at[1].set(0)
    out = _phase(w, inp)
    assert int(out["drops"]["pos"][7][0]) == 1  # player 0 won the single free slot
    assert count(out, 0)[0] == 0 and count(out, 1)[0] == 1


def test_dead_player_cannot_drop():
    w = put(add_player(make_world(), 0, dead=True), 0, 1, I.ITEM_RED_POTION, count=3)
    out = run(w, drop=1)
    assert int(out["drops"]["active"].sum()) == 0 and count(out)[1] == 3


# --- drop rolls ------------------------------------------------------------------------------------

def _roll_kind(kind, n_keys=5):
    w = make_world()
    w = {**w, "monsters": {**w["monsters"], "kind": jnp.full((MAX_MONSTERS,), kind, jnp.int32)}}
    roll = jax.jit(ro_drops.roll_drops)
    outs = [roll(w, jax.random.PRNGKey(100 + k)) for k in range(n_keys)]
    return tuple(np.concatenate([np.asarray(o[i]) for o in outs]) for i in range(3))


def test_roll_drops_chances_within_three_sigma():
    for kind in range(D.N_DROP_KINDS):
        items, counts, zeny = _roll_kind(kind)
        n = items.shape[0]
        assert n >= 20000
        assert items.shape[1] == D.DROPS_PER_KIND and zeny.shape == (n,)
        for j in range(D.DROPS_PER_KIND):
            want = int(D.DROP_ITEM[kind, j])
            if want == 0:
                assert (items[:, j] == 0).all()
                continue
            p = int(D.DROP_CHANCE[kind, j]) / 10000.0
            hits = int((items[:, j] == want).sum())
            assert abs(hits - p * n) <= 3 * np.sqrt(n * p * (1 - p)) + 1, (kind, j, hits, p * n)
            assert (counts[:, j][items[:, j] == want] == int(D.DROP_COUNT[kind, j])).all()
            assert (counts[:, j][items[:, j] == 0] == 0).all()
        assert zeny.min() >= int(D.ZENY_MIN[kind]) and zeny.max() <= int(D.ZENY_MAX[kind])


def _died_world(idx, pos, occupied=0):
    w = make_world()
    m = dict(w["monsters"])
    m["died"] = m["died"].at[jnp.array(idx)].set(True)
    m["pos"] = m["pos"].at[jnp.array(idx)].set(jnp.array(pos, jnp.int32))
    d = dict(w["drops"])
    if occupied:
        d["active"] = d["active"].at[:occupied].set(True)
        d["item"] = d["item"].at[:occupied].set(I.ITEM_FEATHER)
    return {**w, "monsters": m, "drops": d}


def test_place_drops_ranks_free_slots_like_loot():
    w = _died_world([3, 7, 9], [[1, 2], [3, 4], [5, 6]], occupied=2)
    items = jnp.zeros((MAX_MONSTERS, D.DROPS_PER_KIND), jnp.int32)
    counts = jnp.zeros_like(items)
    items = items.at[3, 0].set(I.ITEM_JELLOPY).at[7, 1].set(I.ITEM_APPLE)
    counts = counts.at[3, 0].set(1).at[7, 1].set(1)
    zeny = jnp.zeros((MAX_MONSTERS,), jnp.int32).at[9].set(25).at[4].set(99)  # monster 4 did not die
    out = jax.jit(ro_drops.place_drops)(w, items, counts, zeny)
    d = jax.device_get(out["drops"])
    assert d["active"][:2].all() and d["item"][:2].tolist() == [I.ITEM_FEATHER] * 2  # occupied slots untouched
    assert d["active"].sum() == 5 and d["active"][2:5].all()  # the three new drops take the lowest free slots
    got = {(tuple(int(v) for v in d["pos"][i]), int(d["item"][i])) for i in range(2, 5)}
    assert got == {((1, 2), I.ITEM_JELLOPY), ((3, 4), I.ITEM_APPLE), ((5, 6), ITEM_GOLD_COIN)}
    assert (d["ttl"][2:5] == DROP_TTL).all()


def test_place_drops_discards_beyond_free_slots_and_ignores_living():
    w = _died_world([0, 1, 2], [[1, 1], [2, 2], [3, 3]])
    d0 = dict(w["drops"])
    d0["active"] = jnp.ones((MAX_DROPS,), jnp.bool_).at[11].set(False).at[40].set(False)
    w = {**w, "drops": d0}
    items = jnp.zeros((MAX_MONSTERS, D.DROPS_PER_KIND), jnp.int32).at[0, 0].set(I.ITEM_JELLOPY).at[1, 0].set(I.ITEM_APPLE)
    items = items.at[2, 0].set(I.ITEM_CLOVER).at[500, 0].set(I.ITEM_FLUFF)  # 500 is alive: no drop
    counts = (items > 0).astype(jnp.int32)
    out = ro_drops.place_drops(w, items, counts, jnp.zeros((MAX_MONSTERS,), jnp.int32))
    d = jax.device_get(out["drops"])
    assert d["active"].all()  # the two free slots filled, the third drop discarded
    assert int(d["item"][11]) == I.ITEM_JELLOPY and int(d["item"][40]) == I.ITEM_APPLE
    assert I.ITEM_FLUFF not in d["item"].tolist() and I.ITEM_CLOVER not in d["item"].tolist()


# --- one jit + vmap batch --------------------------------------------------------------------------

def test_vmap_batch_matches_per_world():
    a = put(add_player(make_world(1), 0, hp=1), 0, 1, I.ITEM_RED_POTION, count=2)
    a = put(a, 0, 3, I.ITEM_RED_POTION, count=1)
    a = put(a, 0, 5, ITEM_GOLD_COIN, count=4)
    b = wear(put(add_player(make_world(2), 0), 0, 0, ITEM_SWORD, count=1, refine=2), 0, I.EQ_WEAPON, 0)
    b = set_zeny(put(b, 0, 4, I.ITEM_PHRACON, count=3), 0, 500)
    ia = inputs_for(use_item=1)
    ib = inputs_for(refine=0)

    def stack(x, y):
        return jax.tree_util.tree_map(lambda p, q: jnp.stack([p, q]), x, y)

    def lean(w):  # the items phase never reads the map; dropping it keeps the batch cheap
        return {k: v for k, v in w.items() if k not in ("walls", "terrain", "pz", "dist_town")}

    out = jax.jit(jax.vmap(ro_items.phase))(stack(lean(a), lean(b)), stack(ia, ib))
    for i, (w, inp) in enumerate(((lean(a), ia), (lean(b), ib))):
        ref = _phase(w, inp)
        got = jax.tree_util.tree_map(lambda x: x[i], out)
        same = jax.tree_util.tree_map(lambda x, y: bool(np.array_equal(np.asarray(x), np.asarray(y))), got, ref)
        assert all(jax.tree_util.tree_leaves(same)), i
    cap = int(ro_stats.derived(lean(a))["max_hp"][0])
    assert int(out["players"]["hp"][0, 0]) == min(1 + int(I.HEAL_HP[I.ITEM_RED_POTION]), cap)
    assert int(out["items"]["inv_refine"][1, 0, 0]) == 3


# --- bot policy ------------------------------------------------------------------------------------

_policy = jax.jit(item_inputs)


def policy(w):
    return {k: int(v[0]) for k, v in _policy(w, jax.random.PRNGKey(0)).items()}


def test_policy_equips_best_weapon_only_when_better():
    w = add_player(make_world(), 0)
    w = put(put(w, 0, 2, I.ITEM_KNIFE, count=1), 0, 5, I.ITEM_BLADE, count=1)
    assert policy(w)["equip"] == 5  # BLADE atk 53 beats KNIFE atk 17
    assert policy(wear(w, 0, I.EQ_WEAPON, 5))["equip"] == -1
    assert policy(wear(w, 0, I.EQ_WEAPON, 2))["equip"] == 5


def test_policy_heals_when_hurt_with_potion_not_meat():
    w = put(put(add_player(make_world(), 0, hp=1), 0, 0, ITEM_MEAT, count=1), 0, 3, I.ITEM_APPLE, count=2)
    p = policy(w)
    assert p["use_item"] == 3 and p["drop"] == -1
    healthy = put(put(add_player(make_world(), 0), 0, 0, ITEM_MEAT, count=1), 0, 3, I.ITEM_APPLE, count=2)
    assert policy(healthy)["use_item"] == -1
    dead = put(add_player(make_world(), 0, hp=1, dead=True), 0, 3, I.ITEM_APPLE, count=2)
    assert all(v == -1 for v in policy(dead).values())


def test_policy_refines_below_safe_limit_only():
    chance = np.asarray(ro_refine.REFINE_CHANCE)[1]
    safe_top = int(np.argmax(chance < 1000))  # first level whose step can fail
    assert policy(refine_world(level=3))["refine"] == 0
    assert policy(refine_world(level=safe_top))["refine"] == -1
    assert policy(refine_world(level=safe_top - 1))["refine"] == 0
    assert policy(refine_world(ore_count=0))["refine"] == -1
    assert policy(refine_world(zeny=0))["refine"] == -1


def test_policy_compounds_only_when_a_socket_fits():
    assert policy(card_world())["use_item"] == 1
    assert policy(card_world(card=I.ITEM_PORING_CARD))["use_item"] == -1
    full = put(card_world(), 0, 0, ITEM_SWORD, count=1, cards=[I.ITEM_HORNET_CARD] * 3 + [0])
    assert policy(full)["use_item"] == -1


def test_policy_drives_phase_end_to_end():
    w = add_player(make_world(), 0, hp=1)
    cap = int(ro_stats.derived(w)["max_hp"][0])
    w = put(put(w, 0, 0, I.ITEM_KNIFE), 0, 1, I.ITEM_RED_POTION)
    for _ in range(3):
        inp = {**empty_inputs(), **_policy(w, jax.random.PRNGKey(0))}
        w = _phase(w, inp)
    assert equip(w)[I.EQ_WEAPON] == 0
    assert int(w["players"]["hp"][0]) == min(1 + int(I.HEAL_HP[I.ITEM_RED_POTION]), cap)


# --- round 2: craft ids in the table, job equip restrictions, weapon range -------------------------

def test_craft_items_have_rows_at_their_own_ids():
    from game.data import craft as C
    table = I.table_for_client()
    assert len(table) == I.N_ITEMS == 300
    for cid in C.CRAFT_ITEM_IDS:
        assert I.NAMES[cid] == C.ITEM_NAMES[cid] and table[cid]["name"] == C.ITEM_NAMES[cid]
        assert int(I.KIND[cid]) != I.KIND_NONE and int(I.BUY[cid]) > 0 and int(I.SELL[cid]) > 0
        assert table[cid]["icon"].startswith("item.")
        assert not bool(I.STACKABLE[cid])  # ro_craft counts slots per input
    assert int(I.KIND[C.ITEM_IRON_SWORD]) == I.KIND_WEAPON and int(I.CLASS[C.ITEM_IRON_SWORD]) == I.WC_SWORD
    assert int(I.HEAL_HP[C.ITEM_COOKED_FISH]) > 0
    assert table[150]["name"] == "" and table[150]["kind"] == I.KIND_NONE  # reserved
    assert all(t["icon"] for t in table if t["kind"] != I.KIND_NONE)


def test_craft_item_in_inventory_reads_its_own_row_not_a_card():
    from game.data import craft as C
    w = put(put(add_player(make_world(), 0), 0, 0, C.ITEM_COPPER_ORE), 0, 1, C.ITEM_COPPER_ORE)
    out = run(w)
    assert list(inv(out)[:2]) == [C.ITEM_COPPER_ORE] * 2 and list(count(out)[:2]) == [1, 1]  # not merged


def test_equip_jobs_table_follows_classic_classes():
    J = I.EQUIP_JOBS
    assert J.shape == (I.N_ITEMS, I.N_JOBS)
    assert bool(J[I.ITEM_BOW, JOB_ARCHER]) and bool(J[I.ITEM_BOW, JOB_THIEF]) and not bool(J[I.ITEM_BOW, JOB_NOVICE])
    assert bool(J[I.ITEM_ROD, JOB_MAGE]) and not bool(J[I.ITEM_SWORD, JOB_MAGE])
    assert bool(J[I.ITEM_KNIFE, JOB_MAGE]) and not bool(J[I.ITEM_KNIFE, 4])  # Acolyte: no daggers
    assert bool(J[I.ITEM_HAT].all()) and bool(J[I.ITEM_JELLOPY].all())  # all-job armor, non-gear
    assert not bool(J[I.ITEM_BOOTS, JOB_NOVICE]) and bool(J[I.ITEM_BOOTS, JOB_SWORDSMAN])
    assert I.table_for_client()[I.ITEM_BOW]["jobs"] == [JOB_ARCHER, JOB_THIEF]


def test_equip_refuses_items_the_job_may_not_wear():
    w = put(put(add_player(make_world(), 0), 0, 0, I.ITEM_BOW, count=1), 0, 1, I.ITEM_KNIFE, count=1)
    out = run(w, equip=0)  # Novice and a bow: refused
    assert equip(out)[I.EQ_WEAPON] == -1
    out = run(out, equip=1)  # Novice and a knife: fine
    assert equip(out)[I.EQ_WEAPON] == 1
    out = run(set_job(out, 0, JOB_ARCHER), equip=0)
    assert equip(out)[I.EQ_WEAPON] == 0


def test_unequip_is_allowed_after_a_job_change():
    w = wear(put(add_player(make_world(), 0), 0, 0, ITEM_SWORD, count=1), 0, I.EQ_WEAPON, 0)
    out = run(set_job(w, 0, JOB_MAGE), equip=0)  # a Mage may not wield a sword but may take it off
    assert equip(out)[I.EQ_WEAPON] == -1


def test_weapon_range_bow_nine_melee_and_bare_hands_one():
    w = add_player(add_player(add_player(make_world(), 0), 1), 2)
    w = wear(put(w, 0, 0, I.ITEM_BOW, count=1), 0, I.EQ_WEAPON, 0)
    w = wear(put(w, 1, 0, I.ITEM_COMPOSITE_BOW, count=1), 1, I.EQ_WEAPON, 0)
    w = wear(put(w, 2, 3, I.ITEM_BLADE, count=1), 2, I.EQ_WEAPON, 3)
    r = np.asarray(jax.jit(ro_items.weapon_range)(w))
    assert r.shape == (MAX_PLAYERS,) and r.dtype == np.int32
    assert r[:4].tolist() == [9, 9, 1, 1]


def test_node_view_kinds_sit_above_every_drop_view_kind():
    from game import view
    from game.data import craft as C
    assert C.NODE_VIEW_KIND_BASE == view.NODE_KIND_BASE >= view.DROP_KIND_BASE + I.N_ITEMS
