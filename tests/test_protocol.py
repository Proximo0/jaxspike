import jax, numpy as np
from game.config import (INV_SLOTS, MAP, MONSTER_HP, PLAYER_MAX_HP, PROTOCOL_VERSION, T_GRASS, T_ROCK,
                         VIEW_H, VIEW_W)
from game.state import make_empty_world
from net.protocol import decode, encode, host_view, snapshot, terrain_window, validate_client, view_window

def test_encode_decode_round_trip_and_rejects():
    m = decode(encode({"t": "input", "move": 2, "attack": False, "eat": True}))
    assert m == {"v": PROTOCOL_VERSION, "t": "input", "move": 2, "attack": False, "eat": True}
    for bad in ("not json", "[1,2]", '{"t": "x"}', '{"v": 99, "t": "x"}', '{"v": 1}', '{"v": 1, "t": 5}'):
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

def test_terrain_window_clips_at_origin():
    t = np.full((MAP, MAP), T_GRASS, np.int8)
    x0, y0 = view_window((0, 0))
    rows = terrain_window(t, x0, y0)
    assert len(rows) == VIEW_W and all(len(r) == VIEW_H for r in rows)
    assert rows[0][0] == T_ROCK and rows[-1][-1] == T_GRASS
    assert rows[VIEW_W // 2][VIEW_H // 2] == T_GRASS  # the player's own tile

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

def test_snapshot_contents():
    hv = host_view(_world())
    s = snapshot(hv, 0, {0: "me", 7: "bob", 9: "far"}, send_terrain=True, notice="hi", tick=5, town=0)
    assert s["t"] == "snap" and s["tick"] == 5 and s["notice"] == "hi"
    you = s["you"]
    assert you["pos"] == [100, 100] and you["hp"] == 50 and you["max_hp"] == PLAYER_MAX_HP
    assert you["kills"] == 3 and you["town"] == 0 and you["safe"] is True and len(you["inv"]) == INV_SLOTS
    kinds = sorted((e["kind"], tuple(e["pos"])) for e in s["ents"])
    assert kinds == [("d", (100, 101)), ("m", (102, 102)), ("p", (100, 100)), ("p", (103, 99))]
    names = {e["name"] for e in s["ents"] if e["kind"] == "p"}
    assert names == {"me", "bob"} and not any("slot" in e for e in s["ents"])
    orc = next(e for e in s["ents"] if e["kind"] == "m")
    assert orc["mkind"] == 1 and orc["max_hp"] == MONSTER_HP[1]
    assert s["terrain"]["x0"] == 100 - VIEW_W // 2 and len(s["terrain"]["rows"]) == VIEW_W
    assert snapshot(hv, 0, {}, send_terrain=False, notice=None, tick=6, town=0)["terrain"] is None

def test_snapshot_is_json_serialisable():
    import json
    json.dumps(snapshot(host_view(_world()), 0, {0: "me"}, True, None, 1, 0))
