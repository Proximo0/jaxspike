"""Tests for game/view.py: entity table, change mask, per-viewer visibility and deltas."""
import jax
import jax.numpy as jnp
import numpy as np

from game.config import (MAX_DROPS, MAX_MONSTERS, MAX_PLAYERS, MAX_VIEWERS, N_ENTITIES, VIEW_H, VIEW_K,
                         VIEW_W)
from game import ro_stats
from game.state import make_empty_world
from game.view import changed_mask, entity_table, view_update

HW, HH = VIEW_W // 2, VIEW_H // 2
M0 = MAX_PLAYERS            # first monster id
D0 = MAX_PLAYERS + MAX_MONSTERS


def world_with(players=(), monsters=(), drops=()):
    """players: [(slot, x, y)], monsters: [(i, x, y, kind)], drops: [(i, x, y, item)]."""
    w = make_empty_world(jax.random.PRNGKey(0))
    p, m, d = dict(w["players"]), dict(w["monsters"]), dict(w["drops"])
    for s, x, y in players:
        p["pos"] = p["pos"].at[s].set(jnp.array([x, y], jnp.int32))
        p["active"] = p["active"].at[s].set(True)
    for i, x, y, k in monsters:
        m["pos"] = m["pos"].at[i].set(jnp.array([x, y], jnp.int32))
        m["kind"] = m["kind"].at[i].set(k)
        m["hp"] = m["hp"].at[i].set(40)
        m["active"] = m["active"].at[i].set(True)
    for i, x, y, it in drops:
        d["pos"] = d["pos"].at[i].set(jnp.array([x, y], jnp.int32))
        d["item"] = d["item"].at[i].set(it)
        d["active"] = d["active"].at[i].set(True)
    return {**w, "players": p, "monsters": m, "drops": d}


def run(world, slots, known=None, prev=None):
    now = entity_table(world)
    changed = jnp.ones((N_ENTITIES,), bool) if prev is None else changed_mask(prev, now)
    vs = jnp.full((MAX_VIEWERS,), -1, jnp.int32)
    for v, s in enumerate(slots):
        vs = vs.at[v].set(s)
    if known is None:
        known = jnp.zeros((MAX_VIEWERS, N_ENTITIES), bool)
    s, g, k = view_update(now, changed, vs, known)
    return np.asarray(s), np.asarray(g), np.asarray(k), now


def ids(row):
    return [int(i) for i in row if i >= 0]


def test_entity_table_kinds_and_vis():
    w = world_with(players=[(3, 10, 11)], monsters=[(5, 20, 21, 1)], drops=[(7, 30, 31, 2)])
    w["players"]["dead"] = w["players"]["dead"].at[4].set(True)
    w["players"]["active"] = w["players"]["active"].at[4].set(True)
    t = entity_table(w)
    assert t["pos"].shape == (N_ENTITIES, 2) and t["vis"].shape == (N_ENTITIES,)
    assert N_ENTITIES == MAX_PLAYERS + MAX_MONSTERS + MAX_DROPS
    assert bool(t["vis"][3]) and not bool(t["vis"][4])        # dead player invisible
    assert bool(t["vis"][M0 + 5]) and bool(t["vis"][D0 + 7])
    assert int(t["kind"][3]) == 0 and int(t["kind"][M0 + 5]) == 2 and int(t["kind"][D0 + 7]) == 102
    assert t["pos"][M0 + 5].tolist() == [20, 21]
    assert int(t["hp"][M0 + 5]) == 40 and int(t["max_hp"][M0 + 5]) > 0


def test_player_max_hp_is_derived():
    w = world_with(players=[(0, 5, 5), (1, 6, 5)])
    st = dict(w["stats"])
    st["base_level"] = st["base_level"].at[1].set(30)
    st["vit"] = st["vit"].at[1].set(40)
    w = {**w, "stats": st}
    d = np.asarray(ro_stats.derived(w)["max_hp"])
    t = entity_table(w)
    assert (np.asarray(t["max_hp"][:MAX_PLAYERS]) == d).all() and d[1] > d[0]


def test_changed_mask_fields():
    a = entity_table(world_with(players=[(0, 5, 5)], monsters=[(0, 9, 9, 0)]))
    assert not bool(changed_mask(a, a).any())

    def diff(w):
        return np.nonzero(np.asarray(changed_mask(a, entity_table(w))))[0].tolist()

    assert diff(world_with(players=[(0, 6, 5)], monsters=[(0, 9, 9, 0)])) == [0]
    assert diff(world_with(players=[(0, 5, 5)], monsters=[(0, 9, 9, 1)])) == [M0]
    assert diff(world_with(players=[(0, 5, 5)])) == [M0]
    w = world_with(players=[(0, 5, 5)], monsters=[(0, 9, 9, 0)])
    w["players"]["hp"] = w["players"]["hp"].at[0].set(1)
    assert diff(w) == [0]


def test_window_edges_and_self():
    cx, cy = 100, 100
    mons = [(0, cx + HW, cy + HH, 0), (1, cx + HW + 1, cy, 0), (2, cx - HW, cy - HH, 0),
            (3, cx, cy - HH - 1, 0)]
    w = world_with(players=[(0, cx, cy)], monsters=mons)
    s, g, k, _ = run(w, [0])
    got = set(ids(s[0]))
    assert got == {0, M0 + 0, M0 + 2}
    assert (g[0] == -1).all()
    assert set(np.nonzero(k[0])[0].tolist()) == got
    assert (s[1] == -1).all() and (g[1] == -1).all() and not k[1].any()


def test_self_sent_even_when_dead():
    w = world_with(players=[(0, 50, 50)])
    w["players"]["dead"] = w["players"]["dead"].at[0].set(True)
    s, _, _, _ = run(w, [0])
    assert ids(s[0]) == [0]


def test_nearest_first_ties_by_id_and_padding():
    mons = [(0, 105, 100, 0), (1, 101, 100, 0), (2, 99, 100, 0), (3, 100, 103, 0)]
    w = world_with(players=[(0, 100, 100)], monsters=mons)
    s, _, _, _ = run(w, [0])
    assert ids(s[0]) == [0, M0 + 1, M0 + 2, M0 + 3, M0 + 0]
    assert s.shape == (MAX_VIEWERS, VIEW_K) and (s[0][5:] == -1).all()


def test_overflow_retry():
    n = VIEW_K + 40
    mons = [(i, 100 + (i % 20) - 10, 100 + (i // 20) - 6, 0) for i in range(n)]
    w = world_with(players=[(0, 100, 100)], monsters=mons)
    s1, _, k1, now = run(w, [0])
    assert len(ids(s1[0])) == VIEW_K and k1[0].sum() == VIEW_K
    s2, g2, k2, _ = run(w, [0], known=jnp.asarray(k1), prev=now)
    sent2 = ids(s2[0])
    assert len(sent2) == n + 1 - VIEW_K
    assert not set(sent2) & set(ids(s1[0]))
    assert k2[0].sum() == n + 1 and (g2[0] == -1).all()


def test_no_resend_when_unchanged_and_resend_on_change():
    w = world_with(players=[(0, 100, 100)], monsters=[(0, 101, 100, 0)])
    _, _, k, now = run(w, [0])
    s, g, k2, _ = run(w, [0], known=jnp.asarray(k), prev=now)
    assert ids(s[0]) == [] and ids(g[0]) == [] and (k2 == k).all()
    w2 = world_with(players=[(0, 100, 100)], monsters=[(0, 102, 100, 0)])
    s, _, _, _ = run(w2, [0], known=jnp.asarray(k), prev=now)
    assert ids(s[0]) == [M0]


def test_gone_after_death_and_after_leaving_window():
    w = world_with(players=[(0, 100, 100)], monsters=[(0, 101, 100, 0), (1, 102, 100, 0)])
    _, _, k, now = run(w, [0])
    w2 = world_with(players=[(0, 100, 100)], monsters=[(1, 100 + HW + 1, 100, 0)])
    s, g, k2, _ = run(w2, [0], known=jnp.asarray(k), prev=now)
    assert ids(g[0]) == [M0, M0 + 1] and ids(s[0]) == []
    assert set(np.nonzero(k2[0])[0].tolist()) == {0}
    _, g3, _, _ = run(w2, [0], known=jnp.asarray(k2), prev=entity_table(w2))
    assert ids(g3[0]) == []


def test_gone_overflow_retries():
    n = VIEW_K + 5
    known = np.zeros((MAX_VIEWERS, N_ENTITIES), bool)
    known[0, M0:M0 + n] = True
    known[0, 0] = True
    w = world_with(players=[(0, 100, 100)])
    s, g, k, _ = run(w, [0], known=jnp.asarray(known))
    assert len(ids(g[0])) == VIEW_K
    s2, g2, k2, _ = run(w, [0], known=jnp.asarray(k))
    assert len(ids(g2[0])) == 5 and not k2[0, M0:].any()


def test_two_viewers_and_unused_rows():
    w = world_with(players=[(0, 100, 100), (1, 300, 300)], monsters=[(0, 301, 300, 0)])
    s, _, k, _ = run(w, [0, 1, -1, 1])
    assert ids(s[0]) == [0] and ids(s[1]) == [1, M0] and ids(s[3]) == [1, M0]
    assert (s[2] == -1).all() and not k[2].any()
    known = np.ones((MAX_VIEWERS, N_ENTITIES), bool)
    _, g, k2, _ = run(w, [-1], known=jnp.asarray(known))
    assert (g[0] == -1).all() and not k2[0].any()


def test_jit_and_shapes():
    w = world_with(players=[(0, 100, 100)], monsters=[(0, 101, 101, 1)])

    @jax.jit
    def f(world, prev, known, vs):
        now = entity_table(world)
        return view_update(now, changed_mask(prev, now), vs, known)

    prev = entity_table(make_empty_world(jax.random.PRNGKey(0)))
    known = jnp.zeros((MAX_VIEWERS, N_ENTITIES), bool)
    vs = jnp.full((MAX_VIEWERS,), -1, jnp.int32).at[0].set(0)
    s, g, k = f(w, prev, known, vs)
    assert s.shape == (MAX_VIEWERS, VIEW_K) and s.dtype == jnp.int32
    assert g.shape == (MAX_VIEWERS, VIEW_K) and g.dtype == jnp.int32
    assert k.shape == (MAX_VIEWERS, N_ENTITIES) and k.dtype == jnp.bool_
    assert ids(np.asarray(s)[0]) == [0, M0]


def test_dead_player_gone_for_others_but_self_still_sees_self():
    w = world_with(players=[(0, 100, 100), (1, 102, 100)])
    _, _, k, now = run(w, [0, 1])
    assert k[0, 1] and k[1, 0]
    w2 = world_with(players=[(0, 100, 100), (1, 102, 100)])
    w2["players"]["dead"] = w2["players"]["dead"].at[1].set(True)
    s, g, k2, _ = run(w2, [0, 1], known=jnp.asarray(k), prev=now)
    assert ids(g[0]) == [1] and not k2[0, 1]          # other viewer: dead player is gone
    assert 1 not in ids(s[0])
    assert ids(g[1]) == [0] or ids(g[1]) == []        # the dead viewer keeps itself
    assert k2[1, 1] and 1 not in ids(g[1])
