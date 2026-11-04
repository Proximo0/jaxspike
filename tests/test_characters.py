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
    assert c["inv"].shape == (INV_SLOTS,) and c["inv"].dtype == np.int32
    assert c["inv"][0] > 0 and c["inv"][1] > 0 and not c["inv"][2:].any()  # the Novice starter kit, worn (round 6)
    from game.data.items import EQ_BODY, EQ_WEAPON
    assert c["items_equip"][EQ_WEAPON] == 0 and c["items_equip"][EQ_BODY] == 1

def test_save_load_exact(tmp_path):
    c = new_character(1, (5, 6)); c["hp"] = 77; c["inv"][3] = ITEM_MEAT; c["items_inv_count"][3] = 1; c["kills"] = 9
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


def test_ro_fields_round_trip(tmp_path):
    from net import char_fields
    c = new_character(1, (5, 6))
    assert set(char_fields.RO_KEYS) <= set(c) and "storage" in c
    c["stats_str"] = 40; c["stats_base_level"] = 12; c["stats_sp"] = 3; c["skills_learned"][4] = 2
    c["items_ryn"] = 1234; c["craft_exp"][0] = 120
    save(tmp_path, "ro", c)
    r = load(tmp_path, "ro")
    assert r["stats_str"] == 40 and r["stats_base_level"] == 12 and r["items_ryn"] == 1234
    assert r["skills_learned"][4] == 2 and r["craft_level"][0] == 2
    assert r["storage"].shape == (100,)


def test_old_format_file_loads_with_fresh_ro(tmp_path):
    from net import char_fields
    c = new_character(0, (5, 6)); c["inv"][2] = ITEM_MEAT
    old = {k: c[k] for k in ("town", "pos", "spawn", "hp", "inv", "kills", "deaths")}
    save(tmp_path, "old", old)
    r = load(tmp_path, "old")
    assert r is not None
    fresh = char_fields.fresh_ro_fields(c["inv"])
    for k in char_fields.RO_KEYS:
        np.testing.assert_array_equal(r[k], fresh[k])
    assert r["items_inv_count"][2] == 1 and not r["storage"].any()


def test_bad_ro_values_fall_back_or_clamp(tmp_path):
    c = new_character(0, (5, 6))
    save(tmp_path, "a", {**c, "stats_str": 500, "stats_sp": 10**6, "stats_job": 99})
    r = load(tmp_path, "a")
    assert r["stats_str"] == 99 and r["stats_job"] == 6 and r["stats_sp"] <= r["stats_max_sp"]
    save(tmp_path, "b", {**c, "stats_base_exp": 10**9})
    assert load(tmp_path, "b")["stats_base_exp"] == 0
    save(tmp_path, "c", {**c, "items_inv_cards": np.zeros(3, np.int32), "craft_level": np.zeros(2, np.int32)})
    r = load(tmp_path, "c")
    assert r["items_inv_cards"].shape == (INV_SLOTS, 4) and r["craft_level"].shape == (6,)
    save(tmp_path, "d", {**c, "inv": np.full(INV_SLOTS, 999, np.int32)})
    assert load(tmp_path, "d") is None
    part = {k: v for k, v in c.items() if k != "items_ryn"}
    save(tmp_path, "e", part)
    assert load(tmp_path, "e")["items_ryn"] == 0
