import jax
import jax.numpy as jnp
import numpy as np

from bots.ro_party_policy import party_inputs
from game import ro_party
from game.config import MAX_PLAYERS
from game.state import empty_inputs, make_empty_world
from game.world_layout import TOWNS

N = MAX_PLAYERS
phase = jax.jit(ro_party.phase)
share = jax.jit(ro_party.share_exp)


def world(active=(), level=None):
    w = make_empty_world(jax.random.PRNGKey(0))
    act = np.zeros(N, bool)
    act[list(active)] = True
    w = {**w, "party": ro_party.make_state()}
    w["players"] = {**w["players"], "active": jnp.asarray(act)}
    if level is not None:
        w["stats"] = {"base_level": jnp.asarray(level, jnp.int32)}
    return w


def inp(ops=None, targets=None):
    i = {**empty_inputs(), "party": np.zeros(N, np.int32), "target": np.full(N, -1, np.int32)}
    for s, o in (ops or {}).items():
        i["party"][s] = o
    for s, t in (targets or {}).items():
        i["target"][s] = t
    return {k: jnp.asarray(v) for k, v in i.items()}


def run(w, ops=None, targets=None):
    return phase(w, inp(ops, targets))


def pid(w):
    return np.asarray(w["party"]["party_id"])


def test_create_invite_accept_leave():
    w = world([3, 5, 7])
    w = run(w, {3: 1})
    assert pid(w)[3] == 0 and bool(w["party"]["leader"][3]) and int(w["party"]["member_count"][0]) == 1
    w = run(w, {3: 3}, {3: 5})
    assert int(w["party"]["invite_from"][5]) == 3
    w = run(w, {5: 4})
    assert pid(w)[5] == 0 and int(w["party"]["member_count"][0]) == 2
    assert int(w["party"]["invite_from"][5]) == -1
    w = run(w, {5: 2})
    assert pid(w)[5] == -1 and int(w["party"]["member_count"][0]) == 1


def test_two_creators_get_distinct_ids():
    w = run(world([2, 9]), {2: 1, 9: 1})
    assert list(pid(w)[[2, 9]]) == [0, 1]


def test_invite_tiebreak_lowest_leader():
    w = run(world([1, 4, 8]), {1: 1, 4: 1})
    w = run(w, {1: 3, 4: 3}, {1: 8, 4: 8})
    assert int(w["party"]["invite_from"][8]) == 1


def test_13th_member_rejected_and_slot_order():
    w = world(range(1, 16))
    w = run(w, {1: 1})
    st = dict(w["party"])
    inv = np.full(N, -1, np.int32)
    inv[2:16] = 1
    st["invite_from"] = jnp.asarray(inv)
    st["invite_ttl"] = jnp.where(jnp.asarray(inv) >= 0, 50, 0).astype(jnp.int32)
    w = run({**w, "party": st}, {s: 4 for s in range(2, 16)})
    p = pid(w)
    assert int(w["party"]["member_count"][0]) == ro_party.PARTY_MAX_MEMBERS
    assert (p[2:13] == 0).all() and (p[13:16] == -1).all()
    assert (np.asarray(w["party"]["invite_from"]) == -1).all()


def test_invite_expiry():
    w = world([1, 2])
    w = run(w, {1: 1})
    w = run(w, {1: 3}, {1: 2})
    for _ in range(ro_party.INVITE_TTL - 1):
        w = run(w)
    assert int(w["party"]["invite_from"][2]) == 1
    w = run(w)
    assert int(w["party"]["invite_from"][2]) == -1
    w = run(w, {2: 4})
    assert pid(w)[2] == -1


def test_leader_handoff_and_free():
    w = world([2, 6, 4])
    w = run(w, {2: 1})
    st = dict(w["party"])
    st["party_id"] = st["party_id"].at[jnp.array([6, 4])].set(0)
    w = run({**w, "party": st})
    w = run(w, {2: 2})
    lead = np.asarray(w["party"]["leader"])
    assert lead[4] and not lead[6] and not lead[2]
    w = run(w, {4: 2, 6: 2})
    assert int(w["party"]["member_count"].sum()) == 0
    assert not np.asarray(w["party"]["leader"]).any()
    w = run(w, {6: 1})
    assert pid(w)[6] == 0


def test_logout_cleans():
    w = world([1, 2])
    w = run(w, {1: 1})
    pl = {**w["players"], "active": w["players"]["active"].at[1].set(False)}
    w = run({**w, "players": pl})
    assert pid(w)[1] == -1 and int(w["party"]["member_count"][0]) == 0


def _party_world(members, levels=None, pos=None, dead=()):
    w = world(members, level=levels)
    st = dict(w["party"])
    st["party_id"] = st["party_id"].at[jnp.array(members)].set(0)
    st["member_count"] = st["member_count"].at[0].set(len(members))
    w["party"] = st
    pl = dict(w["players"])
    if pos is not None:
        pl["pos"] = jnp.asarray(pos, jnp.int32)
    d = np.zeros(N, bool)
    d[list(dead)] = True
    pl["dead"] = jnp.asarray(d)
    w["players"] = pl
    return w


def gain(**kw):
    g = np.zeros(N, np.int32)
    for k, v in kw.items():
        g[int(k[1:])] = v
    return jnp.asarray(g)


def test_share_equal_split_with_party_bonus_and_remainder():
    w = _party_world([1, 2, 3])
    out = np.asarray(share(w, gain(s1=100)))  # 100 * 120% = 120, 40 each
    assert out[1] == 40 and out[2] == 40 and out[3] == 40 and out.sum() == 120
    out = np.asarray(share(w, gain(s1=101)))  # 101 * 120% = 121: 41 to the reference killer, 40, 40
    assert out[1] == 41 and out[2] == 40 and out[3] == 40 and out.sum() == 121


def test_share_excludes_level_other_zone_and_dead():
    lv = np.ones(N, np.int32)
    lv[2] = 40
    pos = np.zeros((N, 2), np.int32)
    pos[3] = [200, 0]  # across the first zone split line (192): another field zone
    w = _party_world([1, 2, 3, 4, 5], levels=lv, pos=pos, dead=[5])
    out = np.asarray(share(w, gain(s1=90)))  # 2 eligible: 90 * 110% = 99 -> 50 (ref) + 49
    assert out[2] == 0 and out[3] == 0 and out[5] == 0
    assert out[1] == 50 and out[4] == 49


def test_share_same_zone_ignores_distance():
    pos = np.zeros((N, 2), np.int32)
    pos[1] = [10, 10]
    pos[2] = [180, 180]  # 340 cells away, same field zone
    w = _party_world([1, 2], pos=pos)
    out = np.asarray(share(w, gain(s1=100)))
    assert out[1] == 55 and out[2] == 55


def test_share_single_eligible_member_gets_no_bonus():
    pos = np.zeros((N, 2), np.int32)
    pos[2] = [200, 0]
    w = _party_world([1, 2], pos=pos)
    out = np.asarray(share(w, gain(s1=100)))
    assert out[1] == 100 and out[2] == 0


def test_share_solo_and_conservation():
    w = world([1, 2])
    out = np.asarray(share(w, gain(s1=50, s2=7)))
    assert out[1] == 50 and out[2] == 7
    w = _party_world([1, 2, 3, 4])
    g = gain(s1=101, s3=10, s4=5)
    out = np.asarray(share(w, g))  # pooled 116 * 130% = 150, split 37 x 4 + 2 to the reference
    assert out.dtype == np.int32 and out.sum() == 150 and out[1] == 39 and out[2] == 37


def test_jit_vmap():
    w = world([3])
    w2 = jax.tree_util.tree_map(lambda *x: jnp.stack(x), w, w)
    i = inp({3: 1})
    i2 = jax.tree_util.tree_map(lambda *x: jnp.stack(x), i, i)
    out = jax.jit(jax.vmap(ro_party.phase))(w2, i2)
    assert (np.asarray(out["party"]["party_id"])[:, 3] == 0).all()


def test_bot_policy_parties_of_up_to_four_gatherers_solo():
    from bots.ro_party_policy import PARTY_SIZE, _IS_GATHERER
    nt = len(TOWNS)
    members = list(range(20))
    w = world(members)
    stepf = jax.jit(lambda w, k: ro_party.phase(w, party_inputs(w, k)))
    key = jax.random.PRNGKey(1)
    for _ in range(30):
        w = stepf(w, key)
    p = np.asarray(pid(w))
    g = np.asarray(_IS_GATHERER)[:20]
    assert (p[:20][g] == -1).all()                     # gatherers stay solo
    assert (p[:20][~g] >= 0).all()                     # every hunter is in a party
    mc = np.asarray(w["party"]["member_count"])
    assert mc.max() <= PARTY_SIZE and mc.sum() == int((~g).sum())
    for i in set(p[:20][~g].tolist()):                 # a party never mixes towns
        assert len(set((np.nonzero(p == i)[0] % nt).tolist())) == 1
