import numpy as np

from game.config import INV_SLOTS, ITEM_MEAT
from game.data import craft as craft_data
from game.data import items as item_data
from net import characters
from net.waybell import STORAGE_SLOTS, character_storage, new_storage, transfer, valid_ids


def _inv():
    inv = np.zeros(INV_SLOTS, np.int32)
    inv[2] = ITEM_MEAT
    return inv


def test_round_trip():
    inv, st, ok = transfer(_inv(), new_storage(), 2, "in")
    assert ok and inv[2] == 0 and st[0] == ITEM_MEAT
    inv, st, ok = transfer(inv, st, 0, "out")
    assert ok and inv[0] == ITEM_MEAT and st[0] == 0


def test_refusals():
    inv, st = _inv(), new_storage()
    for slot, d in [(0, "in"), (-1, "in"), (INV_SLOTS, "in"), (2, "sideways"), (0, "out"), (STORAGE_SLOTS, "out")]:
        i2, s2, ok = transfer(inv, st, slot, d)
        assert not ok and (i2 == inv).all() and (s2 == st).all()
    full = np.full(STORAGE_SLOTS, ITEM_MEAT, np.int32)
    assert not transfer(inv, full, 2, "in")[2]
    assert not transfer(np.full(INV_SLOTS, ITEM_MEAT, np.int32), full, 0, "out")[2]


def test_purity():
    inv, st = _inv(), new_storage()
    i0, s0 = inv.copy(), st.copy()
    i2, s2, ok = transfer(inv, st, 2, "in")
    assert ok and (inv == i0).all() and (st == s0).all()
    assert i2 is not inv and s2 is not st


def test_character_storage_validation():
    assert (character_storage({}) == new_storage()).all()
    good = new_storage()
    good[5] = item_data.ITEM_BRAMBLET_CARD  # an RO item id far above the old NUM_ITEMS = 4
    good[6] = craft_data.ITEM_IRON_SWORD  # a craft id
    good[7] = item_data.N_BASE_ITEMS - 1
    assert (character_storage({"storage": good}) == good).all()
    reserved = new_storage(); reserved[0] = 150  # inside the id space but no item there
    for bad in (np.zeros(5, np.int32), np.zeros(STORAGE_SLOTS, np.int64),
                np.full(STORAGE_SLOTS, item_data.N_ITEMS, np.int32), np.full(STORAGE_SLOTS, -1, np.int32), reserved):
        assert (character_storage({"storage": bad}) == new_storage()).all()


def test_valid_ids_covers_every_real_item_and_craft_id():
    assert valid_ids(np.arange(item_data.N_BASE_ITEMS))
    assert valid_ids(np.asarray(craft_data.CRAFT_ITEM_IDS))
    assert not valid_ids([item_data.N_BASE_ITEMS]) or item_data.N_BASE_ITEMS >= item_data.CRAFT_ID_BASE
    assert not valid_ids([item_data.N_ITEMS]) and not valid_ids([-1]) and not valid_ids(np.array([1.0]))


def test_save_load_round_trip(tmp_path):
    ch = characters.new_character(0, (10, 10))
    st = new_storage(); st[7] = ITEM_MEAT
    characters.save(tmp_path, "tester", {**ch, "storage": st})
    got = characters.load(tmp_path, "tester")
    assert got is not None and (character_storage(got) == st).all()
