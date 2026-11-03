"""RO monster modes (game/ro_monster_modes.py): chase by sight, give up and return home, random walk, assist,
looting, immovable kinds, per-kind speed and respawn."""
import jax
import jax.numpy as jnp
import numpy as np

from game import ro_monster_modes as modes
from game import ro_move
from game.config import TICK_MS, ITEM_MEAT, ITEM_NONE, MAX_MONSTERS, MONSTER_HP
from game.data import monsters as mdata
from game.data.monsters import KIND_BY_NAME
from game.monsters import _occupancy, monster_ai
from tests.test_monsters import build, mpos

ORC = KIND_BY_NAME["Orc Warrior"]
PORING = KIND_BY_NAME["Poring"]
WOLF = KIND_BY_NAME["Wolf"]
THIEF = KIND_BY_NAME["Thief Bug"]
EDDGA = KIND_BY_NAME["Eddga"]


def ro(w, wander_in=1000, state=None):
    """Give the world a monai block (no wander for `wander_in` ticks) and set every monster to full hp."""
    m = w["monsters"]
    m["hp"] = jnp.asarray(MONSTER_HP, jnp.int32)[m["kind"]]
    w["monai"] = {**modes.make_state(), "wander_in": jnp.full((MAX_MONSTERS,), wander_in, jnp.int32)}
    if state is not None:
        for i, s in state.items():
            w["monai"]["state"] = w["monai"]["state"].at[i].set(s)
    return w


def decide(w, tables=modes.TABLES):
    return modes.decide(w, w["monsters"]["active"], _occupancy(w), tables)


def st(w, i):
    return int(w["monai"]["state"][i])


def test_aggressive_chases_within_sight_only():
    sight = int(mdata.SIGHT[ORC])
    w = monster_ai(ro(build(players=[(20, 20)], monsters=[(20 + sight, 20)], kind=ORC)))
    assert st(w, 0) == modes.CHASE and mpos(w, 0) == (19 + sight, 20)
    w = monster_ai(ro(build(players=[(20, 20)], monsters=[(21 + sight, 20)], kind=ORC)))
    assert st(w, 0) == modes.IDLE and mpos(w, 0) == (21 + sight, 20)


def test_passive_ignores_until_hurt():
    w = ro(build(players=[(20, 20)], monsters=[(23, 20)], kind=PORING))
    out = monster_ai(w)
    assert st(out, 0) == modes.IDLE and mpos(out, 0) == (23, 20)
    w["monsters"]["hp"] = w["monsters"]["hp"].at[0].set(MONSTER_HP[PORING] - 1)
    out = monster_ai(w)
    assert st(out, 0) == modes.CHASE and mpos(out, 0) == (22, 20)


def test_stops_at_attack_range_and_takes_orthogonal_neighbour():
    out = monster_ai(ro(build(players=[(20, 20)], monsters=[(21, 20)], kind=ORC)))
    assert mpos(out, 0) == (21, 20)
    out = monster_ai(ro(build(players=[(20, 20)], monsters=[(21, 21)], kind=ORC)))
    assert mpos(out, 0) == (20, 21)  # diagonal neighbour steps along x to an orthogonal neighbour


def test_gives_up_beyond_chase_radius_and_walks_home():
    chase = int(mdata.CHASE[ORC])
    w = ro(build(players=[(40, 20)], monsters=[(20, 20)], kind=ORC), state={0: modes.CHASE})
    w["monsters"]["pos"] = w["monsters"]["pos"].at[0].set(jnp.array([21 + chase, 20]))
    w["players"]["pos"] = w["players"]["pos"].at[0].set(jnp.array([23 + chase, 20]))
    out = monster_ai(w)
    assert st(out, 0) == modes.RETURN and mpos(out, 0) == (20 + chase, 20)
    for _ in range(chase * int(mdata.SPEED_MS[ORC]) // TICK_MS + 2):  # walks home at its kind speed, then idles
        out = monster_ai(out)
    assert st(out, 0) == modes.IDLE
    assert max(abs(a - b) for a, b in zip(mpos(out, 0), (20, 20))) <= 1


def test_idle_outside_wander_box_returns_home():
    w = ro(build(monsters=[(20, 20)], kind=PORING))
    w["monsters"]["pos"] = w["monsters"]["pos"].at[0].set(jnp.array([20 + mdata.WANDER_BOX + 2, 20]))
    out = monster_ai(w)
    assert st(out, 0) == modes.RETURN and mpos(out, 0) == (21 + mdata.WANDER_BOX, 20)


def test_random_walk_stays_near_spawn_and_out_of_pz():
    spawns = [(10 + 4 * (i % 8), 10 + 4 * (i // 8)) for i in range(32)]
    pz = [(x, y) for x in range(8, 46) for y in range(8, 46) if (x + y) % 7 == 0 and (x, y) not in spawns]
    w = ro(build(monsters=spawns, kind=PORING, pz=pz), wander_in=0)
    w = {**w, "move": ro_move.make_state()}
    step = jax.jit(monster_ai)
    pzn = np.asarray(w["pz"])
    moved, stood = 0, 0
    first = np.asarray(w["monsters"]["pos"][:32])
    for t in range(150):
        w = step({**w, "tick": jnp.int32(t)})
        pos = np.asarray(w["monsters"]["pos"][:32])
        assert (np.abs(pos - np.asarray(spawns)).max(axis=1) <= mdata.WANDER_BOX).all()
        assert not pzn[pos[:, 0], pos[:, 1]].any()
        assert len({tuple(p) for p in pos}) == 32
        c = modes.counts(w)
        moved, stood = moved + int(c["wander"]), stood + int(c["stand"])
    assert (np.asarray(w["monsters"]["pos"][:32]) != first).any(axis=1).sum() >= 24  # nearly all wandered
    assert stood > moved > 0  # wanders are short walks between standing spells


def test_wander_is_deterministic_under_jit():
    w = ro(build(monsters=[(30, 30)], kind=PORING), wander_in=0)
    a = jax.jit(monster_ai)(w)
    b = monster_ai(w)
    assert mpos(a, 0) == mpos(b, 0)
    assert tuple(np.asarray(a["monai"]["goal"][0])) == tuple(np.asarray(b["monai"]["goal"][0]))
    g = np.asarray(a["monai"]["goal"][0])
    assert 1 <= np.abs(g - 30).max() <= mdata.WANDER_MAX_CELLS or tuple(g) == (30, 30)
    lo, hi = modes.WANDER_MIN_TICKS, modes.WANDER_MAX_TICKS
    assert lo <= int(a["monai"]["wander_in"][0]) <= hi


def test_assist_pulls_kind_mate_into_the_fight():
    passive = {**modes.TABLES, "aggro": jnp.zeros_like(mdata.AGGRO)}
    w = ro(build(players=[(20, 20)], monsters=[(22, 20), (25, 20)], kind=WOLF))
    w["monsters"]["hp"] = w["monsters"]["hp"].at[0].set(1)  # wolf 0 is fighting (hurt)
    out, pd = decide(w, passive)
    assert st(out, 1) == modes.CHASE and int(pd[1]) == int(ro_move.dir_toward(jnp.array([-1, 0])))
    no_assist = {**passive, "assist": jnp.zeros_like(mdata.ASSIST)}
    out, pd = decide(w, no_assist)
    assert st(out, 1) == modes.IDLE and int(pd[1]) == -1
    w2 = ro(build(players=[(20, 20)], monsters=[(22, 20)], kind=WOLF))  # another kind fighting does not pull
    w2 = {**w2, "monsters": {**w2["monsters"]}}
    m = w2["monsters"]
    m["pos"], m["spawn"] = m["pos"].at[1].set(jnp.array([25, 20])), m["spawn"].at[1].set(jnp.array([25, 20]))
    m["active"], m["kind"] = m["active"].at[1].set(True), m["kind"].at[1].set(ORC)
    m["hp"] = m["hp"].at[1].set(1).at[0].set(MONSTER_HP[WOLF])
    out, _ = decide(w2, passive)
    assert st(out, 0) == modes.IDLE


def _drop(w, slot, pos, item=ITEM_MEAT):
    d = w["drops"]
    d["pos"] = d["pos"].at[slot].set(jnp.array(pos, jnp.int32))
    d["item"] = d["item"].at[slot].set(item)
    d["active"] = d["active"].at[slot].set(True)
    d["ttl"] = d["ttl"].at[slot].set(100)
    return w


def test_looter_walks_to_drop_and_picks_it_up():
    w = _drop(ro(build(monsters=[(20, 20)], kind=PORING)), 3, (24, 20))
    w = monster_ai(w)
    assert mpos(w, 0) == (21, 20) and int(w["monai"]["held_item"][0]) == ITEM_NONE
    for _ in range(3 * int(mdata.SPEED_MS[PORING]) // TICK_MS + 1):  # three more cells at Poring speed
        w = monster_ai(w)
    assert int(w["monai"]["held_item"][0]) == ITEM_MEAT
    assert not bool(w["drops"]["active"][3]) and int(w["drops"]["item"][3]) == ITEM_NONE
    orc = _drop(ro(build(monsters=[(20, 20)], kind=ORC)), 3, (21, 20))  # not a looter
    assert int(monster_ai(orc)["monai"]["held_item"][0]) == ITEM_NONE


def test_lowest_index_looter_wins_a_drop():
    w = _drop(ro(build(monsters=[(20, 20), (22, 20)], kind=PORING)), 0, (21, 20))
    w = monster_ai(w)
    assert int(w["monai"]["held_item"][0]) == ITEM_MEAT and int(w["monai"]["held_item"][1]) == ITEM_NONE


def test_looted_item_falls_where_the_monster_died_and_kind_respawn_applies():
    w = ro(build(monsters=[(20, 20)], kind=PORING))
    w["monai"]["held_item"] = w["monai"]["held_item"].at[0].set(ITEM_MEAT)
    w["monai"]["was_active"] = w["monai"]["was_active"].at[0].set(True)
    m = w["monsters"]
    m["active"] = m["active"].at[0].set(False)
    m["pos"] = m["pos"].at[0].set(jnp.array([27, 21]))
    m["respawn_in"] = m["respawn_in"].at[0].set(60)  # combat's generic timer
    out = monster_ai(w)
    d = out["drops"]
    slot = int(np.argmax(np.asarray(d["active"])))
    assert bool(d["active"][slot]) and int(d["item"][slot]) == ITEM_MEAT
    assert tuple(int(v) for v in d["pos"][slot]) == (27, 21)
    assert int(out["monai"]["held_item"][0]) == ITEM_NONE
    assert int(out["monsters"]["respawn_in"][0]) == int(modes.RESPAWN_TICKS[PORING]) - 1


def test_mvp_respawn_timer_untouched():
    w = ro(build(monsters=[(20, 20)], kind=EDDGA))
    w["monai"]["was_active"] = w["monai"]["was_active"].at[0].set(True)
    m = w["monsters"]
    m["active"] = m["active"].at[0].set(False)
    m["respawn_in"] = m["respawn_in"].at[0].set(60)
    assert int(monster_ai(w)["monsters"]["respawn_in"][0]) == 59


def test_immovable_kind_never_moves():
    rooted = {**modes.TABLES, "immovable": jnp.ones_like(mdata.IMMOVABLE)}
    w = ro(build(players=[(20, 20)], monsters=[(23, 20)], kind=ORC), wander_in=0)
    out, pd = decide(w, rooted)
    assert int(pd[0]) == -1 and st(out, 0) == modes.CHASE


def test_kind_speed_through_move_block():
    """Returning home in a straight line: cells walked in 12 ticks follow SPEED_MS[kind]."""
    from game.config import TICK_MS
    walked = {}
    for k in (PORING, THIEF):
        w = ro(build(monsters=[(10, 20)], kind=k), state={0: modes.RETURN})
        w["monsters"]["pos"] = w["monsters"]["pos"].at[0].set(jnp.array([40, 20]))
        w = {**w, "move": ro_move.make_state()}
        for t in range(12):
            w = monster_ai({**w, "tick": jnp.int32(t)})
        walked[k] = 40 - mpos(w, 0)[0]
    for k, n in walked.items():
        assert n == 12 * TICK_MS // int(mdata.SPEED_MS[k]) + 1, (k, walked)  # rested: the first tick steps


def test_fallback_without_block_still_chases_and_returns():
    w = build(players=[(20, 20)], monsters=[(24, 20)], kind=ORC)
    w = {k: v for k, v in w.items() if k not in ("monai", "move")}  # the legacy path: neither block
    assert "monai" not in w
    out = monster_ai(w)
    assert mpos(out, 0) == (23, 20) and "monai" not in out
    far = build(monsters=[(20, 20)], kind=PORING)
    far = {k: v for k, v in far.items() if k not in ("monai", "move")}
    far["monsters"]["hp"] = far["monsters"]["hp"].at[0].set(MONSTER_HP[PORING])
    far["monsters"]["pos"] = far["monsters"]["pos"].at[0].set(jnp.array([30, 20]))
    assert mpos(monster_ai(far), 0) == (29, 20)


def test_jit_vmap_with_block():
    w1 = ro(build(players=[(20, 20)], monsters=[(24, 20)], kind=ORC))
    w2 = ro(build(players=[(20, 20)], monsters=[(20, 25)], kind=ORC))
    batch = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), w1, w2)
    out = jax.jit(jax.vmap(monster_ai))(batch)
    assert tuple(int(v) for v in out["monsters"]["pos"][0, 0]) == (23, 20)
    assert tuple(int(v) for v in out["monsters"]["pos"][1, 0]) == (20, 24)
    assert out["monai"]["state"].shape == (2, MAX_MONSTERS)
