import jax
import numpy as np

from game.config import INV_SLOTS, ITEM_MEAT
from game.data import items as item_data
from game.data.skills import N_SKILLS
from game.state import make_empty_world
from net import characters, char_fields

SLOT = 5


def _geared(world):
    """Give SLOT a refined, carded, worn weapon, potions, ryn, stats, skills and craft progress."""
    w = jax.device_get(world)
    w = jax.tree_util.tree_map(np.array, w)
    weapon = next(i for i in range(item_data.N_ITEMS) if int(item_data.KIND[i]) == item_data.KIND_WEAPON
                  and int(item_data.SLOTS[i]) > 0)
    card = next(i for i in range(item_data.N_ITEMS) if int(item_data.KIND[i]) == item_data.KIND_CARD)
    w["players"]["inv"][SLOT, 0] = weapon
    w["players"]["inv"][SLOT, 1] = ITEM_MEAT
    w["items"]["inv_count"][SLOT, :2] = (1, 25)
    w["items"]["inv_refine"][SLOT, 0] = 4
    w["items"]["inv_cards"][SLOT, 0, 0] = card
    w["items"]["equip"][SLOT, 0] = 0
    w["items"]["ryn"][SLOT] = 4321
    w["players"]["pos"][SLOT] = (30, 40); w["players"]["kills"][SLOT] = 7
    w["stats"]["str"][SLOT] = 30; w["stats"]["base_level"][SLOT] = 9; w["stats"]["job"][SLOT] = 1
    w["stats"]["base_exp"][SLOT] = 5; w["stats"]["was_dead"][SLOT] = True
    sk = int(np.argmax(np.asarray(__import__("game.data.skills", fromlist=["x"]).SKILL_MAX_LEVEL))); w["skills"]["learned"][SLOT, sk] = 2; w["skills"]["sp"][SLOT] = 17
    w["craft"]["exp"][SLOT, 2] = 300; w["craft"]["level"][SLOT, 2] = 3
    w["craft"]["gathers"][SLOT] = 11; w["craft"]["crafts"][SLOT] = 2
    return w


def _eq(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)


def test_apply_then_extract_equals_saved(tmp_path):
    src = _geared(make_empty_world(None))
    fields = char_fields.character_fields(src, SLOT, town=2)
    fields["storage"] = np.zeros(100, np.int32)
    characters.save(tmp_path, "gear", fields)
    loaded = characters.load(tmp_path, "gear")
    assert loaded is not None
    dst = char_fields.apply_character(make_empty_world(None), 9, loaded)
    back = char_fields.character_fields(jax.device_get(dst), 9, town=2, storage=loaded["storage"])
    _eq(back, fields)
    assert loaded["items_equip"][0] == 0 and loaded["items_ryn"] == 4321 and loaded["stats_was_dead"] == 1


def test_apply_missing_fields_keeps_fresh():
    w = make_empty_world(None)
    out = char_fields.apply_character(w, 3, {"hp": 12})
    assert int(out["players"]["hp"][3]) == 12
    assert int(out["stats"]["base_level"][3]) == 1 and int(out["items"]["ryn"][3]) == 0
    other = char_fields.apply_character(w, 4, {"stats_str": 50})
    assert int(other["stats"]["str"][4]) == 50 and int(other["stats"]["str"][3]) == 1


def test_fresh_character_extracts_to_fresh_fields():
    w = jax.device_get(make_empty_world(None))
    got = char_fields.character_fields(w, 0)
    fresh = char_fields.fresh_ro_fields()
    for k in char_fields.RO_KEYS:
        np.testing.assert_array_equal(got[k], fresh[k], err_msg=k)


def test_corrupt_equip_and_cards_are_repaired():
    inv = np.zeros(16, np.int32)
    raw = char_fields.fresh_ro_fields(inv)
    raw["items_equip"] = np.array([3, 3, -1, -1, -1, -1, -1, -1], np.int32)  # points at an empty slot
    raw["items_inv_cards"][0, 0] = 40
    out = char_fields.validate_ro(raw, inv)
    assert (out["items_equip"] == -1).all() and not out["items_inv_cards"].any()


def test_short_skills_array_is_zero_padded():
    inv = np.zeros(INV_SLOTS, np.int32)
    raw = char_fields.fresh_ro_fields(inv)
    old = np.zeros(33, np.int32)
    old[4] = 2
    raw["skills_learned"] = old
    out = char_fields.validate_ro(raw, inv)
    assert out["skills_learned"].shape == (N_SKILLS,)
    assert out["skills_learned"][4] == 2
    assert not out["skills_learned"][33:].any()
    fresh = char_fields.fresh_ro_fields(inv)
    for k in char_fields.RO_KEYS:
        if k != "skills_learned":
            np.testing.assert_array_equal(out[k], fresh[k], err_msg=k)


def test_long_or_2d_skills_array_still_falls_back():
    raw = {"skills_learned": np.zeros(50, np.int32), "skills_sp": 0}
    assert char_fields._skills(raw) is None
    raw2d = {"skills_learned": np.zeros((2, 21), np.int32), "skills_sp": 0}
    assert char_fields._skills(raw2d) is None
    inv = np.zeros(INV_SLOTS, np.int32)
    full = char_fields.fresh_ro_fields(inv)
    full["skills_learned"] = np.zeros(50, np.int32)
    out = char_fields.validate_ro(full, inv)
    np.testing.assert_array_equal(out["skills_learned"], char_fields.fresh_group("skills")["skills_learned"])
