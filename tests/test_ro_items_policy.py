"""Tests for bots/ro_items_policy.py: job-aware weapons, armor upgrades, potions, shop hooks."""
import jax
import jax.numpy as jnp

from bots import ro_items_policy as P
from game import ro_stats
from game.config import MAX_PLAYERS
from game.data import items as I
from game.data.skills import JOB_ARCHER, JOB_MAGE, JOB_NOVICE, JOB_SWORDSMAN
from game.state import make_empty_world

KEY = jax.random.PRNGKey(0)


def world(inv, job=JOB_NOVICE, worn=(), hp_pct=100, ryn=0, pz=False):
    """Slot 0 active at (100, 100) holding `inv` (item ids, one each); worn = [(equip slot, inv slot)]."""
    w = make_empty_world(KEY)
    p, it = w["players"], w["items"]
    p = {**p, "pos": p["pos"].at[0].set(jnp.array([100, 100])), "active": p["active"].at[0].set(True),
         "inv": p["inv"].at[0, :len(inv)].set(jnp.asarray(inv, jnp.int32))}
    it = {**it, "inv_count": it["inv_count"].at[0, :len(inv)].set(1), "ryn": it["ryn"].at[0].set(ryn)}
    for e, s in worn:
        it = {**it, "equip": it["equip"].at[0, e].set(s)}
    w = {**w, "players": p, "items": it, "stats": {**w["stats"], "job": w["stats"]["job"].at[0].set(job)}}
    mx = int(ro_stats.derived(w)["max_hp"][0])
    w["players"] = {**p, "hp": p["hp"].at[0].set(mx * hp_pct // 100)}
    if pz:
        w["pz"] = w["pz"].at[100, 100].set(True)
    if pz:
        from game import ro_shop
        tile = ro_shop.T_SHOP if ro_shop.T_SHOP is not None else 9
        w["terrain"] = w["terrain"].at[101, 100].set(tile)  # a shop tile next to the bot
    return w


_items = jax.jit(P.item_inputs)


def test_equips_the_best_weapon_the_job_may_wield():
    inv = [I.ITEM_BOW, I.ITEM_BLADE, I.ITEM_KNIFE]
    assert int(_items(world(inv, JOB_SWORDSMAN), KEY)["equip"][0]) == 1    # blade (53) over knife; no bows
    assert int(_items(world(inv, JOB_ARCHER), KEY)["equip"][0]) == 2       # knife 17 beats bow 15
    assert int(_items(world([I.ITEM_BOW], JOB_ARCHER), KEY)["equip"][0]) == 0
    assert int(_items(world([I.ITEM_BLADE], JOB_MAGE), KEY)["equip"][0]) == -1  # mages cannot use swords


def test_mage_scores_matk():
    inv = [I.ITEM_KNIFE, I.ITEM_ROD]  # knife ATK 17 vs rod ATK 15 + MATK 15
    assert int(_items(world(inv, JOB_MAGE), KEY)["equip"][0]) == 1


def test_no_swap_when_the_worn_weapon_is_as_good():
    w = world([I.ITEM_BLADE, I.ITEM_KNIFE], JOB_SWORDSMAN, worn=[(I.EQ_WEAPON, 0)])
    assert int(_items(w, KEY)["equip"][0]) == -1


def test_armor_upgrade_by_def_and_free_slots():
    sw = JOB_SWORDSMAN
    assert int(_items(world([I.ITEM_HAT, I.ITEM_HELM], sw), KEY)["equip"][0]) == 1  # helm (6) fills the free slot
    assert int(_items(world([I.ITEM_HAT, I.ITEM_HELM], sw, worn=[(I.EQ_HEAD, 0)]), KEY)["equip"][0]) == 1
    assert int(_items(world([I.ITEM_HAT, I.ITEM_HELM], sw, worn=[(I.EQ_HEAD, 1)]), KEY)["equip"][0]) == -1
    if hasattr(I, "EQUIP_JOBS") and not bool(I.EQUIP_JOBS[I.ITEM_HELM, JOB_NOVICE]):
        assert int(_items(world([I.ITEM_HAT, I.ITEM_HELM]), KEY)["equip"][0]) == 0  # a Novice wears the hat


def test_potion_below_potion_pct_never_meat():
    w = world([I.ITEM_MEAT, I.ITEM_RED_POTION], hp_pct=P.POTION_PCT - 10)
    assert int(_items(w, KEY)["use_item"][0]) == 1
    w = world([I.ITEM_MEAT, I.ITEM_RED_POTION], hp_pct=P.POTION_PCT + 10)
    assert int(_items(w, KEY)["use_item"][0]) == -1


def test_shop_keys_only_when_the_engine_has_them(monkeypatch):
    out = _items(world([I.ITEM_JELLY_BEAD], ryn=5000, pz=True), KEY)
    keys = P.engine_inputs()
    assert ("buy" in out) == ("buy" in keys) and ("sell" in out) == ("sell" in keys)
    monkeypatch.setattr(P, "engine_inputs", lambda: {"buy", "sell"})
    out = P.item_inputs(world([I.ITEM_JELLY_BEAD], ryn=5000, pz=True), KEY)
    assert int(out["buy"][0]) == P.BUY_ITEM and int(out["sell"][0]) == 0
    out = P.item_inputs(world([I.ITEM_JELLY_BEAD], ryn=5000, pz=False), KEY)
    assert int(out["buy"][0]) == -1 and int(out["sell"][0]) == -1
    out = P.item_inputs(world([I.ITEM_JELLY_BEAD], ryn=10, pz=True), KEY)
    assert int(out["buy"][0]) == -1


def test_shapes_jit_vmap_and_never_drops():
    w = world([I.ITEM_BLADE], JOB_SWORDSMAN)
    ws = jax.tree.map(lambda a: jnp.stack([a, a]), w)
    out = jax.jit(jax.vmap(P.item_inputs))(ws, jax.random.split(KEY, 2))
    for k in ("use_item", "equip", "refine", "drop"):
        assert out[k].shape == (2, MAX_PLAYERS) and out[k].dtype == jnp.int32
    assert bool((out["drop"] == -1).all())
    assert int(out["equip"][1, 0]) == 0
