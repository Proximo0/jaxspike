"""Tests for bots/ro_party_policy.py: parties of up to 4, gatherers solo, leader_of."""
import jax
import jax.numpy as jnp
import numpy as np

from bots import ro_party_policy as pp
from game import ro_party
from game.config import MAX_PLAYERS
from game.state import empty_inputs, make_empty_world

KEY = jax.random.PRNGKey(5)
NT = pp.NT
_inputs = jax.jit(pp.party_inputs)
_leader = jax.jit(pp.leader_of)
T0 = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
GATH = [5, 25, 45]
NONG = [s for s in T0 if s not in GATH]


def world(active=(), pos=None):
    w = make_empty_world(KEY)
    p = dict(w["players"])
    p["active"] = p["active"].at[jnp.array(list(active), jnp.int32)].set(True) if active else p["active"]
    w["players"] = p
    return w


def setparty(w, **kw):
    st = dict(w["party"])
    for k, (idx, v) in kw.items():
        st[k] = st[k].at[jnp.array(idx, jnp.int32)].set(v)
    return {**w, "party": st}


def test_one_creator_per_town_gatherers_silent():
    w = world(T0[:8])
    out = _inputs(w, KEY)
    party = np.asarray(out["party"])
    assert [i for i in range(MAX_PLAYERS) if party[i] == 1] == [0]
    assert party[GATH[0]] == 0 and (party != 0).sum() == 1


def test_leader_invites_only_with_room_and_never_gatherers():
    w = world(T0[:8])
    full = setparty(w, party_id=([0, 10, 15, 20], 0), leader=([0], True),
                    member_count=([0], 4))
    assert int(_inputs(full, KEY)["party"][0]) == 0
    part = setparty(w, party_id=([0, 10, 15], 0), leader=([0], True), member_count=([0], 3))
    out = _inputs(part, KEY)
    assert int(out["party"][0]) == 3
    assert int(out["target"][0]) == 20  # 5 is a gatherer
    assert int(out["target"][0]) not in GATH


def test_accept_rules():
    w = world(T0[:8])
    w = setparty(w, invite_from=([5, 20], 0))
    party = np.asarray(_inputs(w, KEY)["party"])
    assert party[5] == 0 and party[20] == 4


def test_leader_of():
    w = world([0, 5, 10, 15, 20])
    pos = w["players"]["pos"].at[0].set(jnp.array([7, 9], jnp.int32)).at[10].set(jnp.array([1, 2], jnp.int32))
    w["players"] = {**w["players"], "pos": pos}
    w = setparty(w, party_id=([0, 10, 15], 3), leader=([0], True), member_count=([3], 3))
    lpos, has, lslot = (np.asarray(x) for x in _leader(w))
    assert has[10] and has[15] and lslot[10] == 0 and tuple(lpos[10]) == (7, 9)
    assert not has[0] and not has[5] and not has[20]
    dead = {**w["players"], "dead": w["players"]["dead"].at[0].set(True)}
    has2 = np.asarray(_leader({**w, "players": dead})[1])
    assert not has2.any()


def test_fallback_lowest_member_when_no_flag():
    w = world([10, 15])
    w = setparty(w, party_id=([10, 15], 2), member_count=([2], 2))
    _, has, lslot = (np.asarray(x) for x in _leader(w))
    assert not has[10] and has[15] and lslot[15] == 10


def test_shapes_dtypes_jit_vmap():
    ws = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), world(T0[:4]), world(T0[:6]))
    out = jax.jit(jax.vmap(pp.party_inputs))(ws, jnp.stack([KEY, KEY]))
    assert out["party"].shape == (2, MAX_PLAYERS) and out["party"].dtype == jnp.int32
    assert out["target"].dtype == jnp.int32 and out["save"].dtype == jnp.bool_
    lp, has, ls = jax.jit(jax.vmap(pp.leader_of))(ws)
    assert lp.shape == (2, MAX_PLAYERS, 2) and lp.dtype == jnp.int32
    assert has.shape == (2, MAX_PLAYERS) and has.dtype == jnp.bool_ and ls.dtype == jnp.int32
    assert set(out) == {"party", "target", "save"}


def test_engine_loop_forms_parties_of_four():
    w = world(T0)
    ph = jax.jit(lambda w, k: ro_party.phase(w, {**empty_inputs(), **pp.party_inputs(w, k)}))
    for _ in range(12):
        w = ph(w, KEY)
    pid = np.asarray(w["party"]["party_id"])
    assert all(pid[s] >= 0 for s in NONG)
    assert all(pid[s] == -1 for s in GATH)
    counts = np.bincount(pid[pid >= 0])
    assert counts.max() <= 4 and (counts > 0).sum() >= 2
