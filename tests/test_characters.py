import numpy as np
from game.config import INV_SLOTS, MAP, PLAYER_MAX_HP, ITEM_MEAT
from game.world_layout import TOWNS
from net.characters import (load, load_roster, new_character, next_town, save, save_roster, valid_name)

def test_valid_name():
    assert valid_name("Curtis_1") == "curtis_1"
    for bad in ("ab", "a" * 17, "bad name", "bad-name", "", "ünïcode"):
        assert valid_name(bad) is None

def test_new_character_defaults():
    c = new_character(2, (10, 20))
    assert c["town"] == 2 and c["hp"] == PLAYER_MAX_HP and c["kills"] == 0 and c["deaths"] == 0
    assert c["pos"].tolist() == [10, 20] and c["spawn"].tolist() == [10, 20]
    assert c["inv"].shape == (INV_SLOTS,) and c["inv"].dtype == np.int32 and not c["inv"].any()

def test_save_load_exact(tmp_path):
    c = new_character(1, (5, 6)); c["hp"] = 77; c["inv"][3] = ITEM_MEAT; c["kills"] = 9
    save(tmp_path, "bob", c)
    r = load(tmp_path, "bob")
    assert set(r) == set(c)
    for k in c:
        np.testing.assert_array_equal(np.asarray(r[k]), np.asarray(c[k]))
    assert isinstance(r["town"], int) and isinstance(r["hp"], int)
    assert load(tmp_path, "nobody") is None

def test_load_corrupt_file_returns_none(tmp_path):
    (tmp_path / "eve.npz").write_bytes(b"not an npz")
    assert load(tmp_path, "eve") is None
    (tmp_path / "empty.npz").write_bytes(b"")
    assert load(tmp_path, "empty") is None

def test_roster_round_robin(tmp_path):
    assert load_roster(tmp_path) == {}
    r = {}
    towns = []
    for i in range(len(TOWNS) + 1):
        towns.append(next_town(r))
        r[f"p{i}"] = towns[-1]
    assert towns == list(range(len(TOWNS))) + [0]
    save_roster(tmp_path, r)
    assert load_roster(tmp_path) == r

def test_save_is_atomic_leaves_no_tmp(tmp_path):
    save(tmp_path, "amy", new_character(0, (5, 6)))
    save_roster(tmp_path, {"amy": 0})
    assert not list(tmp_path.glob("*.tmp"))
    assert load(tmp_path, "amy") is not None and load_roster(tmp_path) == {"amy": 0}

def test_load_validates(tmp_path):
    good = new_character(0, (5, 6))
    save(tmp_path, "ok", good)
    assert load(tmp_path, "ok") is not None
    missing = {k: v for k, v in good.items() if k != "inv"}
    save(tmp_path, "nokey", missing)
    assert load(tmp_path, "nokey") is None
    save(tmp_path, "badpos", {**good, "pos": np.array([MAP, 0], np.int32)})
    assert load(tmp_path, "badpos") is None
    save(tmp_path, "badtown", {**good, "town": 99})
    assert load(tmp_path, "badtown") is None
    save(tmp_path, "badinv", {**good, "inv": np.zeros(3, np.int32)})
    assert load(tmp_path, "badinv") is None
