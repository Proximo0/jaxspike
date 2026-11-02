import jax, json, numpy as np, pytest
from game.config import INV_SLOTS, MAP, MONSTER_HP, N_ENTITIES, PLAYER_MAX_HP, PROTOCOL_VERSION, VIEW_H, VIEW_W
from game.state import make_empty_world
from game.view import changed_mask, entity_table, view_update
from net.protocol import (build_welcome_static, decode, decode_map, encode, encode_map, entity_rows, snapshot,
                          validate_client, view_window, welcome_message)

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
                           "players": {k: p[k] for k in ("pos", "spawn", "hp", "dead", "active", "inv", "kills", "deaths")},
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


# ----- RO messages, welcome tables and the RO `you` block (docs/*_wire.md) -----

def test_validate_login_password_and_create():
    assert validate_client({"t": "login", "name": "bob", "password": "pw"}) == {"t": "login", "name": "bob", "password": "pw"}
    assert validate_client({"t": "login", "name": "bob", "password": "pw", "create": True}) == \
        {"t": "login", "name": "bob", "password": "pw", "create": True}
    assert validate_client({"t": "login", "name": "bob", "password": 5}) is None
    assert validate_client({"t": "login", "name": "bob", "password": "x", "create": 1}) is None
    assert validate_client({"t": "login", "name": 7}) is None

@pytest.mark.parametrize("msg,want", [
    ({"t": "allot", "stat": 0}, {"t": "allot", "stat": 0}),
    ({"t": "allot", "stat": 5, "junk": 1}, {"t": "allot", "stat": 5}),
    ({"t": "job", "job": 6}, {"t": "job", "job": 6}),
    ({"t": "skill", "id": 4, "target": -1}, {"t": "skill", "id": 4, "target": -1}),
    ({"t": "skill", "id": 32, "target": N_ENTITIES - 1}, {"t": "skill", "id": 32, "target": N_ENTITIES - 1}),
    ({"t": "learn", "id": 1}, {"t": "learn", "id": 1}),
    ({"t": "use_item", "slot": 3}, {"t": "use_item", "slot": 3}),
    ({"t": "equip", "slot": 99}, {"t": "equip", "slot": 99}),  # any int: the engine ignores slots out of range
    ({"t": "refine", "slot": -7}, {"t": "refine", "slot": -7}),
    ({"t": "drop", "slot": 0}, {"t": "drop", "slot": 0}),
    ({"t": "gather"}, {"t": "gather"}),
    ({"t": "craft", "recipe": 1}, {"t": "craft", "recipe": 1}),
    ({"t": "party", "op": "invite", "target": "bob"}, {"t": "party", "op": "invite", "target": "bob"}),
    ({"t": "party", "op": "create", "target": None}, {"t": "party", "op": "create", "target": None}),
    ({"t": "chat", "kind": "whisper", "text": "hi", "to": "bob"}, {"t": "chat", "kind": "whisper", "text": "hi", "to": "bob"}),
    ({"t": "chat", "kind": "say", "text": "hi"}, {"t": "chat", "kind": "say", "text": "hi", "to": None}),
    ({"t": "storage", "op": "in", "slot": 2}, {"t": "storage", "op": "in", "slot": 2}),
    ({"t": "save"}, {"t": "save"}),
])
def test_validate_ro_messages_accept(msg, want):
    assert validate_client(msg) == want

@pytest.mark.parametrize("msg", [
    {"t": "allot", "stat": 6}, {"t": "allot", "stat": -1}, {"t": "allot", "stat": True}, {"t": "allot"},
    {"t": "allot", "stat": "1"}, {"t": "job", "job": 0}, {"t": "job", "job": 7}, {"t": "job", "job": 1.0},
    {"t": "skill", "id": 0, "target": -1}, {"t": "skill", "id": 33, "target": -1},
    {"t": "skill", "id": 4, "target": -2}, {"t": "skill", "id": 4, "target": N_ENTITIES}, {"t": "skill", "id": 4},
    {"t": "skill", "id": True, "target": -1}, {"t": "learn", "id": 0}, {"t": "learn", "id": 33},
    {"t": "use_item", "slot": "1"}, {"t": "equip", "slot": True}, {"t": "refine"}, {"t": "drop", "slot": 1.5},
    {"t": "craft", "recipe": 0}, {"t": "craft", "recipe": 99}, {"t": "craft"},
    {"t": "party", "op": "kick", "target": None}, {"t": "party", "op": "invite", "target": 5}, {"t": "party"},
    {"t": "chat", "kind": "yell", "text": "x"}, {"t": "chat", "kind": "say", "text": 5},
    {"t": "chat", "kind": "whisper", "text": "x"}, {"t": "chat", "kind": "say", "text": "x", "to": 3},
    {"t": "storage", "op": "swap", "slot": 1}, {"t": "storage", "op": "in", "slot": True}, {"t": "storage", "op": "in"},
])
def test_validate_ro_messages_reject(msg):
    assert validate_client(msg) is None

def test_welcome_static_tables():
    from game.data import items as item_data
    st = build_welcome_static()
    w = decode(encode(welcome_message(2, 120, ("a", "b"), st)))  # JSON round trip
    for k in ("world", "items", "drops", "skills", "elements", "buffs", "stat_names", "job_names", "stat_cost",
              "max_base_level", "max_job_level", "base_exp_len", "job_exp_len", "mvp_kinds", "recipes", "nodes",
              "craft_items", "craft_skills", "elem_mult"):
        assert k in w, k
    assert w["terrain"] == "a" and w["slot"] == 2
    world = w["world"]
    assert [t["name"] for t in world["towns"]][0] == "Prontera" and len(world["towns"]) == 5
    assert world["towns"][0]["kafra"] == [264, 256] and world["towns"][0]["station"] == [248, 256]
    assert world["dungeons"][0]["box"] == [228, 18, 284, 74] and len(world["warps"]) == 4
    assert world["warps"][0] == {"from": [256, 77], "to": [256, 71]}
    assert world["terrain_names"][8:] == ["warp", "kafra", "station", "cave"] and len(world["terrain_names"]) == 12
    assert len(world["monsters"]) == 20 and world["monsters"][18]["mvp"] is True and world["monsters"][0]["name"] == "Poring"
    assert len(w["items"]) == item_data.N_ITEMS and w["items"][5]["id"] == 5
    assert w["drops"][0]["kind"] == 0 and "drops" in w["drops"][0]
    assert len(w["skills"]["names"]) == 33 and len(w["skills"]["list"]) == 32 and w["skills"]["jobs"][1] == "Swordsman"
    bash = next(s for s in w["skills"]["list"] if s["name"] == "Bash")
    assert len(bash["levels"]) == bash["max_level"] and set(bash["levels"][0]) == \
        {"sp", "cast", "cooldown", "range", "power", "buff_ticks"}
    assert len(w["elements"]) == 10 and len(w["elem_mult"]) == 10 and len(w["buffs"]) == 8
    assert w["stat_names"] == ["str", "agi", "vit", "int", "dex", "luk"] and len(w["job_names"]) == 7
    assert len(w["max_job_level"]) == 7 and w["max_base_level"] == 99 and w["base_exp_len"] == 100 and w["job_exp_len"] == 51
    assert len(w["stat_cost"]) == 100
    assert w["mvp_kinds"][18] == "Golden Thief Bug" and w["mvp_kinds"][19] == "Eddga" and w["mvp_kinds"][0] is None
    assert len(w["mvp_kinds"]) == 20
    assert all({"id", "inputs", "output", "skill", "level", "exp", "station"} <= set(r) for r in w["recipes"])
    assert len(w["nodes"]) == 8 and w["craft_skills"]["names"][0] == "mining" and len(w["craft_skills"]["level_exp"]) == 50

def _ro_pull(w, viewers):
    from game import ro_stats
    from net.server import _ro_blocks
    pull = _pull(w, viewers)
    d = ro_stats.derived(w)
    pull["ro"] = jax.device_get({**_ro_blocks(w), "derived": {"max_hp": d["max_hp"], "max_sp": d["max_sp"]}})
    return pull

def test_snapshot_ro_you_block():
    from game.data import stats as sd
    w = _world()
    st, sk, it, cr, pa = w["stats"], w["skills"], w["items"], w["craft"], w["party"]
    st = {**st, "base_level": st["base_level"].at[0].set(5), "job": st["job"].at[0].set(2), "job_level": st["job_level"].at[0].set(3),
          "str": st["str"].at[0].set(9), "stat_points": st["stat_points"].at[0].set(4), "skill_points": st["skill_points"].at[0].set(2),
          "base_exp": st["base_exp"].at[0].set(7), "sp": st["sp"].at[0].set(11)}  # stats.sp is the SP pool
    sk = {**sk, "casting": sk["casting"].at[0].set(4), "cast_left": sk["cast_left"].at[0].set(3),
          "buffs": sk["buffs"].at[0, 2].set(40).at[0, 5].set(9), "buff_level": sk["buff_level"].at[0, 2].set(3).at[0, 5].set(1),
          "cooldown": sk["cooldown"].at[0, 4].set(6), "learned": sk["learned"].at[0, 2].set(1).at[0, 9].set(5)}
    p = w["players"]
    p = {**p, "inv": p["inv"].at[0, 1].set(7)}
    it = {**it, "inv_count": it["inv_count"].at[0, 1].set(3), "inv_refine": it["inv_refine"].at[0, 1].set(4),
          "inv_cards": it["inv_cards"].at[0, 1, 0].set(9), "equip": it["equip"].at[0, 0].set(1), "zeny": it["zeny"].at[0].set(1234)}
    cr = {**cr, "level": cr["level"].at[0, 1].set(3), "exp": cr["exp"].at[0, 1].set(55)}
    pa = {**pa, "party_id": pa["party_id"].at[0].set(2), "leader": pa["leader"].at[0].set(True)}
    w = {**w, "stats": st, "skills": sk, "items": it, "craft": cr, "party": pa, "players": p}
    pull = _ro_pull(w, [0])
    s = snapshot(pull, 0, {0: "me"}, None, 1, 0, True, {"members": ["me"]})
    you = decode(encode(s))["you"]
    assert you["level"] == 5 and you["job"] == 2 and you["job_level"] == 3
    assert you["stats"] == [9, 1, 1, 1, 1, 1] and you["stat_points"] == 4 and you["skill_points"] == 2
    assert you["sp"] == 11 and you["max_sp"] > 0 and you["max_hp"] == int(pull["ro"]["derived"]["max_hp"][0]) != 0
    assert you["base_exp"] == 7 and you["base_exp_next"] == sd.BASE_EXP_TABLE[5] and you["job_exp_next"] == sd.JOB_EXP_TABLE[2][3]
    assert you["buffs"] == [[2, 40, 3], [5, 9, 1]] and you["cooldowns"] == [[4, 6]] and you["learned"] == [[2, 1], [9, 5]]
    assert you["casting"] == 4 and you["cast_left"] == 3
    assert you["zeny"] == 1234 and len(you["inv"]) == INV_SLOTS and you["equip"] == [1, -1, -1, -1, -1, -1, -1, -1]
    assert you["inv"][1] == {"item": 7, "count": 3, "refine": 4, "cards": [9, 0, 0, 0]}
    assert you["inv"][0] == {"item": 0, "count": 0, "refine": 0, "cards": [0, 0, 0, 0]}
    assert you["craft"]["level"][1] == 3 and you["craft"]["exp"][1] == 55 and len(you["craft"]["level"]) == 6
    assert you["party"] == 2 and you["leader"] is True and you["members"] == ["me"]
    me = next(e for e in s["ents"] if e["id"] == 0)
    assert me["max_hp"] == you["max_hp"]  # players' max_hp in ents comes from the derived stats too

def test_exp_next_is_zero_at_the_caps():
    w = _world()
    st = {**w["stats"], "base_level": w["stats"]["base_level"].at[0].set(99), "job_level": w["stats"]["job_level"].at[0].set(10)}
    s = snapshot(_ro_pull({**w, "stats": st}, [0]), 0, {}, None, 1, 0, True)
    assert s["you"]["base_exp_next"] == 0 and s["you"]["job_exp_next"] == 0  # Novice job cap is 10
