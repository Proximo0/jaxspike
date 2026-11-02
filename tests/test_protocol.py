import jax, json, numpy as np, pytest
from game.config import INV_SLOTS, MAP, MONSTER_HP, PLAYER_MAX_HP, PROTOCOL_VERSION, VIEW_H, VIEW_W
from game.state import make_empty_world
from game.view import changed_mask, entity_table, view_update
from net.protocol import (decode, decode_map, encode, encode_map, entity_rows, snapshot, validate_client,
                          view_window, welcome_message)

def test_encode_decode_round_trip_and_rejects():
    m = decode(encode({"t": "input", "move": 2, "attack": False, "eat": True}))
    assert m == {"v": PROTOCOL_VERSION, "t": "input", "move": 2, "attack": False, "eat": True}
    for bad in ("not json", "[1,2]", '{"t": "x"}', '{"v": 99, "t": "x"}', '{"v": 1, "t": "x"}', '{"v": 2}', '{"v": 2, "t": 5}'):
        assert decode(bad) is None

def test_validate_client():
    assert validate_client({"t": "input", "move": 4, "attack": True, "eat": False, "junk": 1}) == \
        {"t": "input", "move": 4, "attack": True, "eat": False}
    assert validate_client({"t": "input", "move": 5, "attack": True, "eat": False}) is None
    assert validate_client({"t": "input", "move": 1, "attack": 1, "eat": False}) is None
    assert validate_client({"t": "walk_to", "x": MAP, "y": 0}) is None
    assert validate_client({"t": "walk_to", "x": 3, "y": 4}) == {"t": "walk_to", "x": 3, "y": 4}
    assert validate_client({"t": "login", "name": "bob"}) == {"t": "login", "name": "bob"}
    assert validate_client({"t": "bogus"}) is None

def _world():
    w = make_empty_world(jax.random.PRNGKey(0))
    p, m, d = w["players"], w["monsters"], w["drops"]
    p = {**p, "active": p["active"].at[0].set(True).at[7].set(True).at[9].set(True),
         "pos": p["pos"].at[0].set(np.array([100, 100])).at[7].set(np.array([103, 99])).at[9].set(np.array([300, 300])),
         "dead": p["dead"].at[7].set(False), "hp": p["hp"].at[0].set(50), "kills": p["kills"].at[0].set(3)}
    m = {**m, "active": m["active"].at[0].set(True), "pos": m["pos"].at[0].set(np.array([102, 102])),
         "kind": m["kind"].at[0].set(1), "hp": m["hp"].at[0].set(MONSTER_HP[1])}
    d = {**d, "active": d["active"].at[0].set(True), "pos": d["pos"].at[0].set(np.array([100, 101])), "item": d["item"].at[0].set(2)}
    w = {**w, "players": p, "monsters": m, "drops": d, "pz": w["pz"].at[100, 100].set(True)}
    return w

def test_view_window_centres_on_pos():
    assert view_window((100, 50)) == (100 - VIEW_W // 2, 50 - VIEW_H // 2)

def test_welcome_round_trip_terrain_and_pz():
    w = _world()
    terrain = np.asarray(w["terrain"]).copy()
    terrain[3, 7] = 5  # asymmetric cell so a transposed index would show
    pz = np.asarray(w["pz"])
    msg = decode(encode(welcome_message(4, 120, (encode_map(terrain, np.int8), encode_map(pz, np.uint8)))))
    assert msg["t"] == "welcome" and msg["slot"] == 4 and msg["map"] == MAP and msg["tick_ms"] == 120
    assert (msg["view_w"], msg["view_h"]) == (VIEW_W, VIEW_H)
    got = decode_map(msg["terrain"], np.int8)
    assert got.shape == (MAP, MAP) and (got == terrain).all() and got[3, 7] == 5 and got[7, 3] != 5
    assert (decode_map(msg["pz"], np.uint8).astype(bool) == pz).all() and decode_map(msg["pz"], np.uint8)[100, 100] == 1
    import base64, zlib
    assert zlib.decompress(base64.b64decode(msg["terrain"]))[7 * MAP + 3] == 5  # wire index y * MAP + x

def test_decode_map_rejects_bad_blobs():
    for bad in ("!!!", "AAAA", encode_map(np.zeros((MAP, MAP)), np.int8)[:-8]):
        with pytest.raises(ValueError):
            decode_map(bad, np.int8)

def _pull(w, viewers):
    """What the server pulls each tick, built with the real view functions."""
    tab = entity_table(w)
    z = np.zeros((entity_table(w)["pos"].shape[0],), bool)
    known = np.zeros((512, z.shape[0]), bool)
    vs = np.full(512, -1, np.int32)
    for v in viewers:
        vs[v] = v
    prev = {**tab, "pos": tab["pos"] + 1}  # everything counts as changed
    send, gone, _ = view_update(tab, changed_mask(prev, tab), vs, known)
    p = w["players"]
    return jax.device_get({"send": send, "gone": gone, "tab": tab,
                           "players": {k: p[k] for k in ("pos", "spawn", "hp", "dead", "inv", "kills", "deaths")},
                           "safe": w["pz"][p["pos"][:, 0], p["pos"][:, 1]]})

def test_snapshot_contents_v2():
    pull = _pull(_world(), [0])
    s = snapshot(pull, 0, {0: "me", 7: "bob", 9: "far"}, notice="hi", tick=5, town=0, full=True)
    assert s["t"] == "snap" and s["tick"] == 5 and s["notice"] == "hi" and s["full"] is True
    assert "terrain" not in s and s["gone"] == []
    you = s["you"]
    assert you["pos"] == [100, 100] and you["hp"] == 50 and you["max_hp"] == PLAYER_MAX_HP
    assert you["kills"] == 3 and you["town"] == 0 and you["safe"] is True and len(you["inv"]) == INV_SLOTS
    by_id = {e["id"]: e for e in s["ents"]}
    assert set(by_id) == {0, 7, 512, 512 + 4096}  # player 0, player 7 (in view), monster 0, drop 0; player 9 is far
    assert by_id[0]["name"] == "me" and by_id[0]["kind"] == 0 and by_id[7]["name"] == "bob"
    orc = by_id[512]
    assert orc["kind"] == 2 and orc["mkind"] == 1 and orc["max_hp"] == MONSTER_HP[1] and orc["pos"] == [102, 102]
    drop = by_id[512 + 4096]
    assert drop["kind"] == 102 and drop["item"] == 2 and drop["pos"] == [100, 101]
    assert "name" not in orc and "mkind" not in drop

def test_snapshot_is_json_serialisable_and_gone():
    pull = _pull(_world(), [0])
    pull["gone"] = np.array(pull["gone"]); pull["gone"][0][:2] = [3, 4]
    s = snapshot(pull, 0, {}, None, 1, 0, False)
    assert s["gone"] == [3, 4] and s["full"] is False
    assert decode(encode(s))["ents"] == s["ents"]
    json.dumps(s)

def test_entity_rows_bot_name_default():
    pull = _pull(_world(), [0])
    rows = entity_rows(np.array([7]), pull["tab"], {})
    assert rows[0]["name"] == "bot"
