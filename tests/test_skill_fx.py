"""ro6-fx2: the per-player "skill resolved this tick" record (skills.fx_skill / fx_target) and the snap head "fx"."""
import jax
import jax.numpy as jnp
import numpy as np

from game.data import skills as sk
from net import protocol
from tests.test_ro_skills import MP, PHASE, _derived, _inp, _learn, _mon, _sp, _tick, _world


def _fx(w, slot=0):
    return int(w["skills"]["fx_skill"][slot]), np.asarray(w["skills"]["fx_target"][slot]).tolist()


def test_instant_bash_records_skill_and_target_tile_then_clears():
    w = _sp(_learn(_world(monster_pos=(21, 20)), sk.SK_BASH), 50)
    out = _tick(w, skill=sk.SK_BASH, target=_mon(0))
    assert _fx(out) == (sk.SK_BASH, [21, 20])
    assert int(jnp.sum(out["skills"]["fx_skill"] != 0)) == 1
    nxt = _tick(out)
    assert _fx(nxt) == (0, [0, 0])


def test_fire_bolt_fx_only_on_the_resolve_tick_at_the_targets_current_tile():
    s = sk.SK_FIRE_BOLT
    cast = int(sk.SKILL_CAST[s, 1])
    assert cast > 0
    w = _tick(_sp(_learn(_world(kind=1), s), 50), skill=s, target=_mon(0))
    assert _fx(w)[0] == 0  # cast started, nothing resolved
    m = w["monsters"]
    w = {**w, "monsters": {**m, "pos": m["pos"].at[0].set(jnp.array([22, 21]))}}  # the target walked
    for _ in range(cast):
        assert _fx(w)[0] == 0
        w = _tick(w)
    assert _fx(w) == (s, [22, 21])


def test_magnum_break_records_the_casters_tile():
    s = sk.SK_MAGNUM_BREAK
    out = _tick(_sp(_learn(_world(player_pos=(20, 20), monster_pos=(21, 20)), s), 50), skill=s, target=_mon(0))
    assert _fx(out) == (s, [20, 20])


def test_heal_on_an_ally_records_the_allys_tile():
    s = sk.SK_HEAL
    w = _sp(_learn(_world(), s), 50)
    p = w["players"]
    w = {**w, "players": {**p, "pos": p["pos"].at[1].set(jnp.array([22, 22])), "active": p["active"].at[1].set(True),
                          "hp": p["hp"].at[1].set(10)}}
    w = _tick(w, skill=s, target=1)
    for _ in range(30):  # heal may carry a cast time: tick until it resolves
        if int(w["skills"]["casts"][0]) == 1:
            break
        w = _tick(w)
    assert int(w["skills"]["casts"][0]) == 1
    assert _fx(w) == (s, [22, 22])


def test_rejected_cast_records_nothing_and_shapes_hold_under_vmap():
    w = _sp(_learn(_world(monster_pos=(40, 40)), sk.SK_BASH), 50)  # out of range
    out = _tick(w, skill=sk.SK_BASH, target=_mon(0))
    assert int(jnp.sum(out["skills"]["fx_skill"])) == 0
    ws = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), w)
    i = _inp(skill=sk.SK_BASH, target=_mon(0))
    out2 = jax.vmap(PHASE, in_axes=(0, None, None))(ws, i, _derived())
    assert out2["skills"]["fx_skill"].shape == (2, MP) and out2["skills"]["fx_target"].shape == (2, MP, 2)
    assert out2["skills"]["fx_target"].dtype == jnp.int32


# ----- wire -----
def _fx_pull(rows, viewer_pos=(100, 100)):
    """A minimal pull: viewer slot 0 at viewer_pos, `rows` = [(caster, skill, x, y)] resolved this tick."""
    n_ent = 8
    skill = np.zeros(MP, np.int32)
    target = np.zeros((MP, 2), np.int32)
    for c, s, x, y in rows:
        skill[c], target[c] = s, (x, y)
    pos = np.zeros((MP, 2), np.int32)
    pos[0] = viewer_pos
    tab = {"kind": np.zeros(n_ent, np.int32), "pos": np.zeros((n_ent, 2), np.int32), "hp": np.zeros(n_ent, np.int32),
           "max_hp": np.zeros(n_ent, np.int32)}
    return {"send": np.full((1, 4), -1), "gone": np.full((1, 4), -1), "tab": tab, "players": {"pos": pos},
            "fx": {"skill": skill, "target": target}}


def test_fx_head_in_json_and_bin_filtered_to_the_view_window():
    pull = _fx_pull([(3, sk.SK_FIRE_BOLT, 105, 98), (4, sk.SK_HEAL, 300, 300), (0, sk.SK_MAGNUM_BREAK, 100, 100)])
    views = [(0, {}, None, True, "bin"), (0, {}, None, True, "json")]
    data, text = protocol.build_snaps(pull, views, 7, {})
    b, j = protocol.decode_snap_bin(data), protocol.decode(text)
    want = [[0, sk.SK_MAGNUM_BREAK, 100, 100], [3, sk.SK_FIRE_BOLT, 105, 98]]
    assert b["fx"] == want and j["fx"] == want and b == j


def test_fx_absent_when_nothing_resolved_and_capped_at_32():
    data, text = protocol.build_snaps(_fx_pull([]), [(0, {}, None, True, "bin"), (0, {}, None, True, "json")], 1, {})
    assert "fx" not in protocol.decode_snap_bin(data) and "fx" not in protocol.decode(text)
    many = [(c, sk.SK_BASH, 100, 100) for c in range(1, 41)]
    data, text = protocol.build_snaps(_fx_pull(many), [(0, {}, None, True, "bin"), (0, {}, None, True, "json")], 1, {})
    assert len(protocol.decode_snap_bin(data)["fx"]) == protocol.FX_PER_SNAP == 32
    assert len(protocol.decode(text)["fx"]) == 32
