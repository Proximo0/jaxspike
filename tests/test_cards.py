"""Round 4 (ro4-cards): card effect tables, welcome.cards, the card-drop announcement compare and its wire message."""
import jax
import jax.numpy as jnp
import numpy as np

from game import loot, ro_drops, ro_items
from game.config import MAX_MONSTERS
from game.config import MAX_DROPS, MAX_PLAYERS
from game.data import drops as D
from game.data import items as I
from game.data import monsters as M
from game.state import make_empty_world
from net import cards, protocol

POS = (40, 41)
_events = jax.jit(cards.card_events)


def _world():
    w = make_empty_world(jax.random.PRNGKey(3))
    return {**w, "items": ro_items.make_state()}


def _player(w, i, pos, dead=False):
    p = dict(w["players"])
    p["pos"] = p["pos"].at[i].set(jnp.array(pos, jnp.int32))
    p["active"] = p["active"].at[i].set(True)
    p["dead"] = p["dead"].at[i].set(dead)
    return {**w, "players": p}


def _died(w, m, kind, pos):
    mo = dict(w["monsters"])
    mo["pos"] = mo["pos"].at[m].set(jnp.array(pos, jnp.int32))
    mo["kind"] = mo["kind"].at[m].set(kind)
    mo["active"] = mo["active"].at[m].set(True)
    mo["died"] = mo["died"].at[m].set(True)
    return {**w, "monsters": mo}


def _place(w, m_items):
    """Run the real ro_drops.place_drops for the died monsters with the given {monster: item} rolls."""
    items = np.zeros((MAX_MONSTERS, D.DROPS_PER_KIND), np.int32)
    for m, it in m_items.items():
        items[m, 0] = it
    counts = (items > 0).astype(np.int32)
    return ro_drops.place_drops(w, jnp.asarray(items), jnp.asarray(counts), jnp.zeros((MAX_MONSTERS,), jnp.int32))


def _equip_shield_with(card):
    w = _world()
    p = dict(w["players"])
    p["inv"] = p["inv"].at[0, 0].set(I.ITEM_GUARD)
    p["active"] = p["active"].at[0].set(True)
    it = dict(w["items"])
    it["inv_count"] = it["inv_count"].at[0, 0].set(1)
    it["inv_cards"] = it["inv_cards"].at[0, 0, 0].set(card)
    it["equip"] = it["equip"].at[0, I.EQ_SHIELD].set(0)
    return {**w, "players": p, "items": it}


# --- tables -------------------------------------------------------------------------------------------------------

def test_every_monster_card_is_in_the_album_table_with_its_monster():
    tab = ro_items.card_table_for_client()
    ids = [r["id"] for r in tab]
    assert ids == sorted(ids) and len(ids) == int((np.asarray(I.KIND) == I.KIND_CARD).sum())
    by_kind = {r["kind"]: r for r in tab if r["kind"] >= 0}
    assert sorted(by_kind) == list(range(M.N_KINDS))  # all 20 monsters drop exactly their own card
    for k, r in by_kind.items():
        assert r["monster"] == M.MONSTERS[k].name and r["name"] == M.MONSTERS[k].name + " Card"
        assert r["chance"] == 1 and r["icon"] == "card." + M.MONSTERS[k].name.lower().replace(" ", "_")
        assert r["mvp"] == bool(M.MONSTERS[k].mvp) and r["where"] != ""
    for r in tab:
        assert r["text"] != "" and r["effects"] and r["slot_name"] in ro_items.SLOT_LABELS


def test_mvp_cards_are_strong_and_named():
    tab = {r["id"]: r for r in ro_items.card_table_for_client()}
    g, s = tab[I.ITEM_GILDSCARAB_CARD], tab[I.ITEM_SCORCHSTRIPE_CARD]
    assert g["mvp"] and s["mvp"] and g["text"] == "DEF +10, Max HP +1000, VIT +3, Max SP -60"
    assert s["text"] == "STR +5, ATK +30, FLEE +10, Max HP -400"
    assert "Solmera Culvert" in g["where"] and "Pinehollow Cave" in s["where"]
    normal = [sum(abs(v) for _, v in I.card_effects(i)) for i in tab if not tab[i]["mvp"]]
    assert all(sum(abs(v) for _, v in I.card_effects(i)) > 0 for i in tab)
    assert sum(abs(v) for _, v in I.card_effects(I.ITEM_GILDSCARAB_CARD)) > max(normal)


def test_equipped_folds_every_card_effect():
    w = _equip_shield_with(I.ITEM_GILDSCARAB_CARD)
    b = np.asarray(jax.jit(ro_items.equipped)(w)["card_bonus"][0])
    assert (b[I.BONUS_DEF], b[I.BONUS_MAX_HP], b[I.BONUS_VIT], b[I.BONUS_MAX_SP]) == (10, 1000, 3, -60)
    assert int(np.abs(b).sum()) == 10 + 1000 + 3 + 60
    hood = _equip_shield_with(I.ITEM_SCORCHSTRIPE_CARD)  # a shoes card in a shield socket still counts (socket rule)
    b2 = np.asarray(ro_items.equipped(hood)["card_bonus"][0])
    assert (b2[I.BONUS_STR], b2[I.BONUS_ATK], b2[I.BONUS_FLEE], b2[I.BONUS_MAX_HP]) == (5, 30, 10, -400)


def test_welcome_static_carries_the_card_table():
    static = protocol.build_welcome_static()
    assert static["cards"] == ro_items.card_table_for_client()


# --- announcement ---------------------------------------------------------------------------------------------------

def test_card_from_a_kill_is_announced_with_the_top_damager():
    w = _player(_world(), 3, (POS[0] + 1, POS[1]))
    pre = w["drops"]
    w = _died(w, 7, M.KIND_BY_NAME["Emberblob"], POS)
    post = _place(w, {7: I.ITEM_EMBERBLOB_CARD})["drops"]
    top = jnp.full((MAX_MONSTERS,), -1, jnp.int32).at[7].set(3)
    ev = jax.device_get(_events(pre, post, w["monsters"], top, w["players"]))
    assert int(ev["valid"].sum()) == 1
    assert (int(ev["item"][0]), int(ev["kind"][0]), int(ev["finder"][0])) == (
        I.ITEM_EMBERBLOB_CARD, M.KIND_BY_NAME["Emberblob"], 3)
    msgs = cards.card_messages(ev, {3: "Ana"}, 55)
    assert msgs == [{"t": "card", "item": I.ITEM_EMBERBLOB_CARD, "kind": M.KIND_BY_NAME["Emberblob"], "finder": "Ana",
                     "tick": 55, "mvp": False, "text": "Ana found an Emberblob Card!"}]
    assert protocol.decode(protocol.encode(msgs[0]))["text"] == "Ana found an Emberblob Card!"


def test_one_shot_kill_falls_back_to_the_nearest_player_and_bots_are_someone():
    w = _player(_player(_world(), 5, (POS[0] + 2, POS[1])), 9, (POS[0] + 1, POS[1]), dead=True)
    pre = w["drops"]
    w = _died(w, 0, 0, POS)
    post = _place(w, {0: I.ITEM_BLOBBIN_CARD})["drops"]
    ev = jax.device_get(_events(pre, post, w["monsters"], jnp.full((MAX_MONSTERS,), -1, jnp.int32), w["players"]))
    assert int(ev["finder"][0]) == 5  # the dead player 9 is nearer but cannot be the finder
    assert cards.card_messages(ev, {}, 1)[0]["text"] == "Someone found a Blobbin Card!"


def test_non_cards_old_piles_and_player_drops_are_not_announced():
    w = _player(_world(), 0, POS)
    d = dict(w["drops"])  # a card the player dropped on a tile where no monster died, plus an old card pile
    d["pos"] = d["pos"].at[0].set(jnp.array((5, 5), jnp.int32)).at[1].set(jnp.array(POS, jnp.int32))
    d["item"] = d["item"].at[0].set(I.ITEM_ANTLET_CARD).at[1].set(I.ITEM_PEEPIT_CARD)
    d["active"] = d["active"].at[0].set(True).at[1].set(True)
    d["ttl"] = d["ttl"].at[0].set(50).at[1].set(50)
    pre = {**w["drops"], "active": d["active"].at[0].set(False), "item": d["item"].at[0].set(0),
           "pos": d["pos"], "ttl": d["ttl"].at[1].set(51)}
    w = _died({**w, "drops": d}, 2, 1, POS)
    post = _place(w, {2: I.ITEM_JELLY_BEAD})["drops"]  # the kill dropped a non-card on the old pile's tile
    ev = jax.device_get(_events(pre, post, w["monsters"], jnp.zeros((MAX_MONSTERS,), jnp.int32), w["players"]))
    assert not ev["valid"].any() and cards.card_messages(ev, {}, 1) == []


def test_many_cards_in_one_tick_cap_at_max_events_and_vmap():
    w = _player(_world(), 0, (1, 1))
    pre = w["drops"]
    rolls = {}
    for m in range(cards.MAX_CARD_EVENTS + 4):
        w = _died(w, m, 2, (20 + 2 * m, 30))
        rolls[m] = I.ITEM_WRIGGLET_CARD
    post = _place(w, rolls)["drops"]
    top = jnp.full((MAX_MONSTERS,), -1, jnp.int32)
    ev = _events(pre, post, w["monsters"], top, w["players"])
    assert int(ev["valid"].sum()) == cards.MAX_CARD_EVENTS and ev["item"].shape == (cards.MAX_CARD_EVENTS,)
    stack = lambda t: jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), t)  # noqa: E731
    vev = jax.vmap(cards.card_events)(stack(pre), stack(post), stack(w["monsters"]), stack(top), stack(w["players"]))
    assert vev["valid"].shape == (2, cards.MAX_CARD_EVENTS) and int(vev["valid"].sum()) == 2 * cards.MAX_CARD_EVENTS


def test_loot_phase_card_roll_is_announced_end_to_end(monkeypatch):
    """A real loot.loot pass with a forced card roll (every entry hits) gives exactly the kind's card event."""
    w = _player(_world(), 4, (POS[0], POS[1] + 1))
    pre_drops = w["drops"]
    w = _died(w, 11, M.KIND_BY_NAME["Snarlfang"], POS)
    # every drop entry hits (the roll reads the module's DROP_CHANCE, whichever loot path rolls)
    monkeypatch.setattr(ro_drops, "DROP_CHANCE", jnp.full_like(D.DROP_CHANCE, 10000))
    out = loot.loot(w)
    ev = jax.device_get(_events(pre_drops, out["drops"], out["monsters"],
                                jnp.full((MAX_MONSTERS,), -1, jnp.int32), out["players"]))
    assert [int(i) for i in ev["item"][ev["valid"]]] == [I.ITEM_SNARLFANG_CARD]
    assert int(ev["finder"][0]) == 4
    assert MAX_DROPS >= cards.MAX_CARD_EVENTS and MAX_PLAYERS > 4
