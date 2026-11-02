import numpy as np

from game.config import INV_SLOTS, ITEM_MEAT, NUM_ITEMS
from net import characters
from net.kafra import STORAGE_SLOTS, character_storage, new_storage, transfer


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
    good = new_storage(); good[5] = NUM_ITEMS - 1
    assert (character_storage({"storage": good}) == good).all()
    for bad in (np.zeros(5, np.int32), np.zeros(STORAGE_SLOTS, np.int64), np.full(STORAGE_SLOTS, NUM_ITEMS, np.int32)):
        assert (character_storage({"storage": bad}) == new_storage()).all()


def test_save_load_round_trip(tmp_path):
    ch = characters.new_character(0, (10, 10))
    st = new_storage(); st[7] = ITEM_MEAT
    characters.save(tmp_path, "tester", {**ch, "storage": st})
    got = characters.load(tmp_path, "tester")
    assert got is not None and (character_storage(got) == st).all()
