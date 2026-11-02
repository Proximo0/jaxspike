"""Acceptance tests for the M2 skills phase (docs/m2_skills_brief.md). One jitted phase is reused by every test."""
import jax
import jax.numpy as jnp
import numpy as np

from game import ro_skills
from game.config import (
    MAX_PLAYERS,
    MONSTER_HP,
    MONSTER_RESPAWN_TICKS,
    PLAYER_ATTACK_COOLDOWN,
    PLAYER_MAX_HP,
)
from game.data import elements as el
from game.data import skills as sk
from game.state import empty_inputs, make_empty_world

MP = MAX_PLAYERS
PHASE = jax.jit(ro_skills.phase)


# ---------------------------------------------------------------- helpers
def _world(player_pos=(20, 20), monster_pos=(21, 20), kind=0, job=None, points=None):
    w = make_empty_world(jax.random.PRNGKey(0))
    p, m = w["players"], w["monsters"]
    p = {**p, "pos": p["pos"].at[0].set(jnp.array(player_pos)),
         "spawn": p["spawn"].at[0].set(jnp.array([30, 30])),
         "active": p["active"].at[0].set(True)}
    m = {**m, "pos": m["pos"].at[0].set(jnp.array(monster_pos)),
         "hp": m["hp"].at[0].set(MONSTER_HP[kind]),
         "kind": m["kind"].at[0].set(kind),
         "active": m["active"].at[0].set(True)}
    w = {**w, "players": p, "monsters": m, "skills": ro_skills.make_state()}
    if job is not None:
        w["stats"] = {"job": jnp.full((MP,), job, jnp.int32),
                      "skill_points": jnp.full((MP,), 0 if points is None else points, jnp.int32)}
    return w


def _derived(**over):
    d = {"atk": 10, "matk": 20, "hit": 0, "flee": 0, "def": 0, "mdef": 0, "aspd_ticks": PLAYER_ATTACK_COOLDOWN,
         "cast_mult": 1000, "max_hp": PLAYER_MAX_HP, "max_sp": 50, "crit": 0}
    d.update(over)
    return {k: jnp.full((MP,), v, jnp.int32) for k, v in d.items()}


def _inp(skill=0, target=-1, learn=0, slot=0):
    i = empty_inputs()
    return {**i, "skill": jnp.zeros((MP,), jnp.int32).at[slot].set(skill),
            "target": jnp.full((MP,), -1, jnp.int32).at[slot].set(target),
            "learn": jnp.zeros((MP,), jnp.int32).at[slot].set(learn)}


def _tick(w, d=None, **kw):
    return PHASE(w, _inp(**kw), _derived() if d is None else d)


def _run(w, n, d=None):
    for _ in range(n):
        w = _tick(w, d)
    return w


def _set(w, key, idx, val):
    return {**w, "skills": {**w["skills"], key: w["skills"][key].at[idx].set(val)}}


def _learn(w, skill, lvl=1, slot=0):
    return _set(w, "learned", (slot, skill), lvl)


def _sp(w, v=50, slot=0):
    """Set the authoritative SP pool (stats.sp when the world has a stats block)."""
    blk, key = ro_skills._sp_pool(w)
    return {**w, blk: {**w[blk], key: w[blk][key].at[slot].set(v)}}


def _nostats(w):
    return {k: v for k, v in w.items() if k != "stats"}


def _mon(i):
    return MP + i


def _dmg(base, skill, lvl, defend_elem):
    """Expected damage from the brief's formula using table values."""
    return (base * int(sk.SKILL_POWER[skill, lvl]) // 1000 * int(sk.SKILL_HITS[skill])
            * int(el.ELEM_MULT[int(sk.SKILL_ELEMENT[skill]), defend_elem]) // 100)


def _hp(w, i=0):
    return int(w["monsters"]["hp"][i])


# ---------------------------------------------------------------- state and tables
def test_make_state_shapes_and_dtypes():
    s = ro_skills.make_state()
    shapes = {"learned": (MP, sk.N_SKILLS), "cooldown": (MP, sk.N_SKILLS), "cast_left": (MP,), "casting": (MP,),
              "cast_target": (MP,), "cast_tile": (MP, 2), "buffs": (MP, sk.N_BUFFS), "buff_level": (MP, sk.N_BUFFS),
              "sp": (MP,), "casts": (MP,), "skill_dmg": (MP,)}
    assert set(s) == set(shapes)
    for k, shp in shapes.items():
        assert s[k].shape == shp, k
        assert s[k].dtype == jnp.int32, k
    assert int(s["sp"][0]) == sk.SP_START
    assert int(s["cast_target"][0]) == -1
    assert int(s["learned"].sum()) == 0 and int(s["buffs"].sum()) == 0
    assert ro_skills.active_buffs({"skills": s}).shape == (MP, sk.N_BUFFS)


def test_table_shapes():
    assert sk.N_SKILLS == 33 and sk.N_BUFFS == 8 and sk.MAX_SKILL_LEVEL == 10
    for name in ("SKILL_JOB", "SKILL_MAX_LEVEL", "SKILL_EFFECT", "SKILL_ELEMENT", "SKILL_TARGET", "SKILL_AOE",
                 "SKILL_BUFF", "SKILL_HITS", "SKILL_FLAT_HEAL"):
        assert getattr(sk, name).shape == (sk.N_SKILLS,), name
    for name in ("SKILL_SP", "SKILL_CAST", "SKILL_COOLDOWN", "SKILL_RANGE", "SKILL_POWER", "SKILL_BUFF_TICKS"):
        assert getattr(sk, name).shape == (sk.N_SKILLS, sk.MAX_SKILL_LEVEL + 1), name
    assert len(sk.SKILL_NAME) == sk.N_SKILLS
    assert el.ELEM_MULT.shape == (el.N_ELEMENTS, el.N_ELEMENTS) and len(el.ELEM_NAME) == el.N_ELEMENTS


def test_elem_mult_spot_checks():
    assert int(el.elem_mult(el.ELEM_NEUTRAL, el.ELEM_GHOST)) == 25
    assert int(el.elem_mult(el.ELEM_NEUTRAL, el.ELEM_NEUTRAL)) == 100
    assert int(el.elem_mult(el.ELEM_HOLY, el.ELEM_UNDEAD)) in (125, 150)
    assert int(el.elem_mult(el.ELEM_FIRE, el.ELEM_WATER)) == int(el.ELEM_MULT[el.ELEM_FIRE, el.ELEM_WATER])
    assert (np.asarray(el.ELEM_MULT) >= 0).all() and (np.asarray(el.ELEM_MULT) <= 200).all()
    assert int(el.MONSTER_ELEMENT[0]) == el.ELEM_WATER and int(el.MONSTER_ELEMENT[1]) == el.ELEM_EARTH  # Poring, Orc Warrior
    out = el.elem_mult(jnp.array([el.ELEM_NEUTRAL, el.ELEM_HOLY]), jnp.array([el.ELEM_GHOST, el.ELEM_UNDEAD]))
    assert out.shape == (2,)


# ---------------------------------------------------------------- learn
def test_learn_job_gate_and_points():
    w = _world(job=sk.JOB_NOVICE, points=3)
    w1 = _tick(w, learn=sk.SK_BASH)  # swordsman skill, novice cannot
    assert int(w1["skills"]["learned"][0, sk.SK_BASH]) == 0
    assert int(w1["stats"]["skill_points"][0]) == 3
    w2 = _tick(w, learn=sk.SK_FIRST_AID)  # novice skill, any job
    assert int(w2["skills"]["learned"][0, sk.SK_FIRST_AID]) == 1
    assert int(w2["stats"]["skill_points"][0]) == 2
    ws = _world(job=sk.JOB_SWORDSMAN, points=2)
    w3 = _tick(ws, learn=sk.SK_BASH)
    assert int(w3["skills"]["learned"][0, sk.SK_BASH]) == 1
    assert int(w3["stats"]["skill_points"][0]) == 1
    w4 = _tick(w3, learn=sk.SK_BASH)
    assert int(w4["skills"]["learned"][0, sk.SK_BASH]) == 2
    assert int(w4["stats"]["skill_points"][0]) == 0
    w5 = _tick(w4, learn=sk.SK_BASH)  # no points left
    assert int(w5["skills"]["learned"][0, sk.SK_BASH]) == 2
    assert int(w5["stats"]["skill_points"][0]) == 0
    assert int(w5["skills"]["learned"][1:].sum()) == 0  # other slots untouched


def test_learn_stops_at_max_level():
    mx = int(sk.SKILL_MAX_LEVEL[sk.SK_BASH])
    w = _learn(_world(job=sk.JOB_SWORDSMAN, points=5), sk.SK_BASH, mx)
    w = _tick(w, learn=sk.SK_BASH)
    assert int(w["skills"]["learned"][0, sk.SK_BASH]) == mx
    assert int(w["stats"]["skill_points"][0]) == 5
    mx = int(sk.SKILL_MAX_LEVEL[sk.SK_FIRST_AID])
    w = _learn(_world(job=sk.JOB_NOVICE, points=5), sk.SK_FIRST_AID, mx)
    w = _tick(w, learn=sk.SK_FIRST_AID)
    assert int(w["skills"]["learned"][0, sk.SK_FIRST_AID]) == mx


def test_learn_free_without_stats_block():
    w = _world()
    w.pop("stats")  # make_empty_world carries the stats block since the merge; learning is free without it
    assert "stats" not in w
    w = _tick(w, learn=sk.SK_BASH)  # everyone is a novice: Bash is a swordsman skill
    assert int(w["skills"]["learned"][0, sk.SK_BASH]) == 0
    mx = int(sk.SKILL_MAX_LEVEL[sk.SK_FIRST_AID])
    for n in (1, 2, 3):
        w = _tick(w, learn=sk.SK_FIRST_AID)
        assert int(w["skills"]["learned"][0, sk.SK_FIRST_AID]) == min(n, mx)


def test_learn_ignores_inactive_player_and_dead_may_learn():
    w = _world(job=sk.JOB_SWORDSMAN, points=3)
    w1 = PHASE(w, _inp(learn=sk.SK_BASH, slot=5), _derived())  # slot 5 is not logged in
    assert int(w1["skills"]["learned"][5, sk.SK_BASH]) == 0
    wd = {**w, "players": {**w["players"], "dead": w["players"]["dead"].at[0].set(True)}}
    wd = _tick(wd, learn=sk.SK_BASH)
    assert int(wd["skills"]["learned"][0, sk.SK_BASH]) == 1


# ---------------------------------------------------------------- cast start
def test_cast_start_sets_sp_cooldown_cast_left():
    s, lvl = sk.SK_FIRE_BOLT, 1
    cast = int(sk.SKILL_CAST[s, lvl])
    assert cast > 0
    w = _sp(_learn(_world(monster_pos=(21, 20), kind=1), s, lvl), 50)
    w = {**w, "tick": jnp.array(1, jnp.int32)}  # off the SP regen schedule
    w1 = _tick(w, skill=s, target=_mon(0))
    assert int(w1["skills"]["casting"][0]) == s
    assert int(w1["stats"]["sp"][0]) == 50 - int(sk.SKILL_SP[s, lvl])
    assert int(w1["skills"]["cooldown"][0, s]) == int(sk.SKILL_COOLDOWN[s, lvl])
    assert int(w1["skills"]["cast_left"][0]) == cast - 1  # countdown runs in the start tick
    assert int(w1["skills"]["cast_target"][0]) == _mon(0)
    assert w1["skills"]["cast_tile"][0].tolist() == [21, 20]
    assert int(w1["skills"]["casts"][0]) == 0
    w2 = _tick(w, d=_derived(cast_mult=2000), skill=s, target=_mon(0))
    assert int(w2["skills"]["cast_left"][0]) == 2 * cast - 1


def test_cooldown_blocks_second_cast_then_releases():
    s = sk.SK_BASH
    cd = int(sk.SKILL_COOLDOWN[s, 1])
    assert cd >= 1
    w = _sp(_learn(_world(kind=1), s), 50)
    w1 = _tick(w, skill=s, target=_mon(0))
    hp1 = _hp(w1)
    assert hp1 < MONSTER_HP[1]
    w2 = _tick(w1, skill=s, target=_mon(0))
    assert _hp(w2) == hp1
    assert int(w2["skills"]["casts"][0]) == 1
    w3 = _run(w1, cd)  # cooldown ticks down to zero
    assert int(w3["skills"]["cooldown"][0, s]) == 0
    w4 = _tick(w3, skill=s, target=_mon(0))
    assert _hp(w4) < hp1


def test_cast_rejections():
    s = sk.SK_BASH
    base = _sp(_learn(_world(kind=1), s), 50)

    def rejected(w, target=_mon(0)):
        out = _tick(w, skill=s, target=target)
        assert int(out["skills"]["casting"][0]) == 0
        assert int(out["skills"]["cooldown"][0, s]) == 0
        assert _hp(out) == _hp(w)
        assert int(out["stats"]["sp"][0]) >= int(w["stats"]["sp"][0])  # nothing deducted (regen may add)

    rejected(_sp(_world(kind=1), 50))  # not learned
    rejected(_sp(base, 0))  # no SP
    rejected(_sp(base, int(sk.SKILL_SP[s, 1]) - 1))
    rejected(base, target=-1)  # no target
    rejected(base, target=3)  # a player slot for an enemy skill
    rejected(base, target=_mon(7))  # empty monster slot
    rejected(_sp(_learn(_world(monster_pos=(25, 20), kind=1), s), 50))  # out of range (Bash range is melee)
    m = base["monsters"]
    rejected({**base, "monsters": {**m, "active": m["active"].at[0].set(False)}})
    p = base["players"]
    rejected({**base, "players": {**p, "dead": p["dead"].at[0].set(True)}})


def test_range_check_accepts_edge_and_rejects_beyond():
    s = sk.SK_FIRE_BOLT
    rng = int(sk.SKILL_RANGE[s, 1])
    assert rng > 1
    near = _sp(_learn(_world(monster_pos=(20 + rng, 20), kind=1), s), 50)
    assert int(_tick(near, skill=s, target=_mon(0))["skills"]["casting"][0]) == s
    far = _sp(_learn(_world(monster_pos=(20 + rng, 21), kind=1), s), 50)  # manhattan rng + 1
    assert int(_tick(far, skill=s, target=_mon(0))["skills"]["casting"][0]) == 0


def test_passive_cannot_be_cast():
    s = sk.SK_SWORD_MASTERY
    assert int(sk.SKILL_EFFECT[s]) == sk.EFFECT_PASSIVE
    w = {**_sp(_learn(_world(), s, 3), 50), "tick": jnp.array(1, jnp.int32)}
    out = _tick(w, skill=s, target=_mon(0))
    assert int(out["skills"]["casting"][0]) == 0
    assert int(out["stats"]["sp"][0]) == 50
    assert int(out["skills"]["cooldown"][0, s]) == 0
    assert _hp(out) == MONSTER_HP[0]


# ---------------------------------------------------------------- damage
def test_instant_bash_damages_rat_same_tick():
    s = sk.SK_BASH
    assert int(sk.SKILL_CAST[s, 1]) == 0
    w = {**_sp(_learn(_world(kind=0), s), 50), "tick": jnp.array(1, jnp.int32)}
    out = _tick(w, skill=s, target=_mon(0))
    expect = 10 * int(sk.SKILL_POWER[s, 1]) // 1000 * int(el.ELEM_MULT[int(sk.SKILL_ELEMENT[s]), el.ELEM_NEUTRAL]) // 100
    assert expect == _dmg(10, s, 1, el.ELEM_NEUTRAL) and expect > 0
    assert _hp(out) == MONSTER_HP[0] - expect
    assert int(out["skills"]["casting"][0]) == 0
    assert int(out["skills"]["casts"][0]) == 1
    assert int(out["skills"]["skill_dmg"][0]) == expect
    assert int(out["stats"]["sp"][0]) == 50 - int(sk.SKILL_SP[s, 1])
    assert not bool(out["monsters"]["active"][1])  # nothing else touched


def test_fire_bolt_takes_cast_time_and_uses_element_multiplier():
    s = sk.SK_FIRE_BOLT
    cast = int(sk.SKILL_CAST[s, 1])
    w = _sp(_learn(_world(kind=1), s), 50)  # orc: earth
    w = _tick(w, skill=s, target=_mon(0))
    for _ in range(cast - 1):
        assert _hp(w) == MONSTER_HP[1]
        w = _tick(w)
    assert _hp(w) == MONSTER_HP[1]  # still casting after `cast` ticks
    w = _tick(w)
    expect = 20 * int(sk.SKILL_POWER[s, 1]) // 1000 * int(el.ELEM_MULT[el.ELEM_FIRE, el.ELEM_EARTH]) // 100
    assert expect == _dmg(20, s, 1, el.ELEM_EARTH)
    assert _hp(w) == MONSTER_HP[1] - expect
    assert int(w["skills"]["casting"][0]) == 0 and int(w["skills"]["casts"][0]) == 1
    assert int(w["skills"]["skill_dmg"][0]) == expect


def test_kill_via_skill_sets_flags_and_credit():
    s = sk.SK_BASH
    w = _sp(_learn(_world(kind=0), s), 50)
    w = {**w, "monsters": {**w["monsters"], "hp": w["monsters"]["hp"].at[0].set(1)}}
    out = _tick(w, skill=s, target=_mon(0))
    assert bool(out["monsters"]["died"][0])
    assert not bool(out["monsters"]["active"][0])
    assert int(out["monsters"]["respawn_in"][0]) == MONSTER_RESPAWN_TICKS
    assert int(out["monsters"]["hp"][0]) <= 0
    assert int(out["players"]["kills"][0]) == 1


def test_magnum_break_hits_chebyshev_one_only():
    s = sk.SK_MAGNUM_BREAK
    assert int(sk.SKILL_AOE[s]) == 1 and int(sk.SKILL_TARGET[s]) == sk.TARGET_SELF
    w = _sp(_learn(_world(monster_pos=(21, 20), kind=0), s), 50)
    spots = [(19, 19), (20, 21), (22, 20), (20, 23)]  # diagonal (in), south (in), distance 2, distance 3
    m = w["monsters"]
    for i, pos in enumerate(spots, start=1):
        m = {**m, "pos": m["pos"].at[i].set(jnp.array(pos)), "hp": m["hp"].at[i].set(MONSTER_HP[0]),
             "kind": m["kind"].at[i].set(0), "active": m["active"].at[i].set(True)}
    w = {**w, "monsters": m}
    out = _tick(w, skill=s, target=_mon(0))  # self-target ignores the target value
    expect = _dmg(10, s, 1, int(el.MONSTER_ELEMENT[0]))  # kind 0 = Poring (Water) since the RO world merge
    assert expect > 0
    hps = [_hp(out, i) for i in range(5)]
    assert hps[:3] == [MONSTER_HP[0] - expect] * 3
    assert hps[3:] == [MONSTER_HP[0]] * 2
    assert int(out["skills"]["skill_dmg"][0]) == 3 * expect
    assert int(out["skills"]["cooldown"][0, s]) == int(sk.SKILL_COOLDOWN[s, 1])


def test_magnum_break_kill_credit_goes_to_lowest_caster():
    s = sk.SK_MAGNUM_BREAK
    w = _sp(_learn(_world(monster_pos=(21, 20)), s), 50)
    p = w["players"]
    p = {**p, "pos": p["pos"].at[1].set(jnp.array([22, 21])), "active": p["active"].at[1].set(True),
         "spawn": p["spawn"].at[1].set(jnp.array([30, 30]))}
    w = _sp(_learn({**w, "players": p}, s, 1, slot=1), 50, slot=1)
    w = {**w, "monsters": {**w["monsters"], "hp": w["monsters"]["hp"].at[0].set(1)}}
    i = _inp(skill=s, target=0, slot=0)
    i = {**i, "skill": i["skill"].at[1].set(s), "target": i["target"].at[1].set(1)}
    out = PHASE(w, i, _derived())
    assert not bool(out["monsters"]["active"][0]) and bool(out["monsters"]["died"][0])
    assert int(out["players"]["kills"][0]) == 1 and int(out["players"]["kills"][1]) == 0


# ---------------------------------------------------------------- heal and buffs
def test_heal_clamps_at_max_hp_and_heals_ally():
    s = sk.SK_HEAL
    heal = (10 + 20) * int(sk.SKILL_POWER[s, 1]) // 1000 + int(sk.SKILL_FLAT_HEAL[s])
    assert heal > 0
    cast = int(sk.SKILL_CAST[s, 1])
    w = _sp(_learn(_world(), s), 50)
    p = w["players"]
    p = {**p, "hp": p["hp"].at[0].set(PLAYER_MAX_HP - 3).at[1].set(1),
         "pos": p["pos"].at[1].set(jnp.array([22, 20])), "active": p["active"].at[1].set(True)}
    w = {**w, "players": p}
    out = _run(_tick(w, skill=s, target=0), cast + 1)  # self-heal clamps
    assert int(out["players"]["hp"][0]) == PLAYER_MAX_HP
    out = _run(_tick(w, skill=s, target=1), cast + 1)  # ally gets the full amount
    assert int(out["players"]["hp"][1]) == min(1 + heal, PLAYER_MAX_HP)
    assert int(out["players"]["hp"][0]) == PLAYER_MAX_HP - 3  # the caster was not healed


def test_heal_cannot_target_dead_ally():
    s = sk.SK_HEAL
    w = _sp(_learn(_world(), s), 50)
    p = w["players"]
    p = {**p, "hp": p["hp"].at[1].set(0), "dead": p["dead"].at[1].set(True),
         "pos": p["pos"].at[1].set(jnp.array([22, 20])), "active": p["active"].at[1].set(True)}
    out = _tick({**w, "players": p}, skill=s, target=1)
    assert int(out["skills"]["casting"][0]) == 0
    assert int(out["skills"]["cooldown"][0, s]) == 0


def test_blessing_sets_buff_and_stat_mods_then_expires():
    s, lvl = sk.SK_BLESSING, 3
    assert int(sk.SKILL_BUFF[s]) == sk.BUFF_BLESSING
    ticks = int(sk.SKILL_BUFF_TICKS[s, lvl])
    cast = int(sk.SKILL_CAST[s, lvl])
    d = _derived(max_sp=500)
    w = _sp(_learn(_world(), s, lvl), 500)
    out = _run(_tick(w, d=d, skill=s, target=0), cast, d)  # resolves on tick cast + 1
    assert int(out["skills"]["buffs"][0, sk.BUFF_BLESSING]) == ticks
    assert int(out["skills"]["buff_level"][0, sk.BUFF_BLESSING]) == lvl
    assert int(ro_skills.active_buffs(out)[0, sk.BUFF_BLESSING]) == ticks
    mods = ro_skills.buff_stat_mods(out)
    assert int(mods["str_flat"][0]) == lvl and int(mods["int_flat"][0]) == lvl and int(mods["dex_flat"][0]) == lvl
    assert int(mods["str_flat"][1]) == 0
    out = _run(out, 5, d)
    assert int(out["skills"]["buffs"][0, sk.BUFF_BLESSING]) == ticks - 5
    out = _set(out, "buffs", (0, sk.BUFF_BLESSING), 2)  # tick down to nothing
    out = _run(out, 2, d)
    assert int(out["skills"]["buffs"][0, sk.BUFF_BLESSING]) == 0
    assert int(out["skills"]["buff_level"][0, sk.BUFF_BLESSING]) == 0
    assert int(ro_skills.buff_stat_mods(out)["str_flat"][0]) == 0


def test_buff_stat_mods_keys_and_passives():
    expected = {"str_flat", "agi_flat", "vit_flat", "int_flat", "dex_flat", "luk_flat", "atk_flat", "atk_pct",
                "matk_pct", "hit_flat", "flee_flat", "def_pct", "mdef_flat", "aspd_pct", "crit_flat",
                "hp_regen_pct"}
    w = _learn(_learn(_world(), sk.SK_SWORD_MASTERY, 5), sk.SK_OWLS_EYE, 4)
    mods = ro_skills.buff_stat_mods(w)
    assert set(mods) == expected
    for v in mods.values():
        assert v.shape == (MP,) and v.dtype == jnp.int32
    assert int(mods["atk_flat"][0]) == 20 and int(mods["dex_flat"][0]) == 4
    w = _learn(w, sk.SK_HP_RECOVERY, 3)
    assert int(ro_skills.buff_stat_mods(w)["hp_regen_pct"][0]) == 30
    w = _set(_set(w, "buffs", (0, sk.BUFF_INC_AGI), 50), "buff_level", (0, sk.BUFF_INC_AGI), 2)  # Inc AGI lvl 2
    mods = ro_skills.buff_stat_mods(w)
    assert int(mods["agi_flat"][0]) == 4 and int(mods["aspd_pct"][0]) == 2


def test_buffs_tick_down_without_casting():
    w = _set(_set(_world(), "buffs", (0, sk.BUFF_ENDURE), 10), "buff_level", (0, sk.BUFF_ENDURE), 1)
    out = _run(w, 4)
    assert int(out["skills"]["buffs"][0, sk.BUFF_ENDURE]) == 6
    assert int(out["skills"]["buff_level"][0, sk.BUFF_ENDURE]) == 1


# ---------------------------------------------------------------- death and SP
def test_dead_caster_cast_cancelled_and_buffs_cleared():
    s = sk.SK_FIRE_BOLT
    cast = int(sk.SKILL_CAST[s, 1])
    w = _sp(_learn(_world(kind=1), s), 50)
    w = _set(_set(w, "buffs", (0, sk.BUFF_BLESSING), 99), "buff_level", (0, sk.BUFF_BLESSING), 2)
    w = _tick(w, skill=s, target=_mon(0))
    assert int(w["skills"]["casting"][0]) == s
    w = {**w, "players": {**w["players"], "dead": w["players"]["dead"].at[0].set(True),
                          "hp": w["players"]["hp"].at[0].set(0)}}
    w = _run(w, cast + 3)
    assert int(w["skills"]["casting"][0]) == 0 and int(w["skills"]["cast_left"][0]) == 0
    assert _hp(w) == MONSTER_HP[1]
    assert int(w["skills"]["casts"][0]) == 0
    assert int(w["skills"]["buffs"][0].sum()) == 0


def test_sp_regen_on_schedule_and_clamp():
    """Fallback pool (no stats block): ro_skills regens skills.sp itself; with stats, ro_stats.phase owns regen."""
    w = _sp(_nostats(_world()), 10)
    step = 1 + 50 // 100
    start = 8 * sk.SP_REGEN_TICKS
    seen = []
    for t in range(start, start + sk.SP_REGEN_TICKS):
        w = _tick({**w, "tick": jnp.array(t, jnp.int32)})
        seen.append(int(w["skills"]["sp"][0]))
    assert seen[-1] == 10 + step  # exactly one regen per SP_REGEN_TICKS ticks
    assert sum(1 for a, b in zip([10] + seen, seen) if b > a) == 1
    assert int(_tick(_sp(_world(), 50))["stats"]["sp"][0]) == 50
    assert int(_tick(_sp(_world(), 400))["stats"]["sp"][0]) == 50  # above max_sp is clamped


def test_no_sp_regen_for_dead():
    w = _sp(_nostats(_world()), 10)
    w = {**w, "players": {**w["players"], "dead": w["players"]["dead"].at[0].set(True)}}
    for t in range(sk.SP_REGEN_TICKS * 2):
        w = _tick({**w, "tick": jnp.array(t, jnp.int32)})
    assert int(w["skills"]["sp"][0]) == 10


# ---------------------------------------------------------------- jit and vmap
def _bash_world():
    return _sp(_learn(_world(kind=0), sk.SK_BASH), 50)


def test_phase_jit_matches_eager():
    w = _bash_world()
    i = _inp(skill=sk.SK_BASH, target=_mon(0))
    d = _derived()
    jit_out = PHASE(w, i, d)
    eager = ro_skills.phase(w, i, d)
    for a, b in zip(jax.tree_util.tree_leaves(jit_out), jax.tree_util.tree_leaves(eager)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert _hp(jit_out) == MONSTER_HP[0] - _dmg(10, sk.SK_BASH, 1, el.ELEM_NEUTRAL)


def test_phase_vmaps_over_two_identical_worlds():
    w = _bash_world()

    def stack(t):
        return jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), t)

    out = jax.jit(jax.vmap(ro_skills.phase))(stack(w), stack(_inp(skill=sk.SK_BASH, target=_mon(0))),
                                              stack(_derived()))
    expect = MONSTER_HP[0] - _dmg(10, sk.SK_BASH, 1, el.ELEM_NEUTRAL)
    assert out["monsters"]["hp"][:, 0].tolist() == [expect, expect]
    for leaf in jax.tree_util.tree_leaves(out):
        assert np.array_equal(np.asarray(leaf[0]), np.asarray(leaf[1]))


def test_fallback_derived_has_contract_keys():
    d = ro_skills.fallback_derived(_world())
    for k in ("atk", "matk", "hit", "flee", "def", "mdef", "aspd_ticks", "cast_mult", "max_hp", "max_sp", "crit"):
        assert d[k].shape == (MP,), k
    assert int(d["max_hp"][0]) == PLAYER_MAX_HP and int(d["cast_mult"][0]) == 1000


def test_sp_spent_from_stats_pool_when_stats_block_exists():
    s, lvl = sk.SK_FIRE_BOLT, 1
    w = _learn(_world(monster_pos=(21, 20), kind=1), s, lvl)
    assert ro_skills._sp_pool(w) == ("stats", "sp")
    w = {**w, "stats": {**w["stats"], "sp": w["stats"]["sp"].at[0].set(40)}, "tick": jnp.array(1, jnp.int32)}
    w = _set(w, "sp", 0, 999)  # the fallback pool is ignored
    w1 = _tick(w, skill=s, target=_mon(0))
    assert int(w1["stats"]["sp"][0]) == 40 - int(sk.SKILL_SP[s, lvl])
    assert int(w1["skills"]["sp"][0]) == 999


def test_sp_pool_falls_back_to_skills_without_stats():
    w = _world()
    w = {k: v for k, v in w.items() if k != "stats"}
    assert ro_skills._sp_pool(w) == ("skills", "sp")
