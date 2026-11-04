import jax
import jax.numpy as jnp
import numpy as np

from game import ro_stats
from game.config import MAX_PLAYERS
from game.data import stats as S
from game.state import empty_inputs, make_empty_world

PHASE = jax.jit(ro_stats.phase)
NEG = jnp.full((MAX_PLAYERS,), -1, jnp.int32)


def _world(active=(0,), tick=1):
    w = make_empty_world(jax.random.PRNGKey(0))
    p = dict(w["players"])
    for i in active:
        p["active"] = p["active"].at[i].set(True)
        p["pos"] = p["pos"].at[i].set(jnp.array([10, 10], jnp.int32))
    return {**w, "players": p, "stats": ro_stats.make_state(), "tick": jnp.int32(tick)}


def _set(w, slot, **kw):
    st = dict(w["stats"])
    for k, v in kw.items():
        st[k] = st[k].at[slot].set(v)
    return {**w, "stats": st}


def _setp(w, slot, **kw):
    p = dict(w["players"])
    for k, v in kw.items():
        p[k] = p[k].at[slot].set(v)
    return {**w, "players": p}


def _inputs(allot=None, job=None):
    i = empty_inputs()
    a, j = NEG, NEG
    for slot, v in (allot or {}).items():
        a = a.at[slot].set(v)
    for slot, v in (job or {}).items():
        j = j.at[slot].set(v)
    return {**i, "allot": a, "job": j}


def _run(w, **kw):
    return PHASE(w, _inputs(**kw))


def _die_monster(w, pos=(10, 11), kind=0, k=0):
    m = dict(w["monsters"])
    m["died"] = m["died"].at[k].set(True)
    m["kind"] = m["kind"].at[k].set(kind)
    m["pos"] = m["pos"].at[k].set(jnp.array(pos, jnp.int32))
    return {**w, "monsters": m}


def _st(out, key, slot=0):
    return int(out["stats"][key][slot])


def _veteran(w, slot, level=20, job=S.JOB_SWORDSMAN, job_level=20):
    # a level where one rat kill cannot level up, so the paid exp is read back exactly
    return _set(w, slot, base_level=level, job=job, job_level=job_level)


def test_fresh_novice_derived():
    d = ro_stats.derived(_world())
    assert int(d["max_hp"][0]) == 40  # RO pre-re Novice level 1 = 40 HP
    assert int(d["hit"][0]) == 2  # RO: level 1 + dex 1
    assert int(d["flee"][0]) == 2  # RO: level 1 + agi 1
    assert int(d["aspd"][0]) == 151 and int(d["aspd_ticks"][0]) == 10  # Novice fist 150, delay 980 ms
    assert 990 <= int(d["cast_mult"][0]) <= 1000
    assert int(d["max_sp"][0]) >= 10
    assert d["max_hp"].dtype == jnp.int32 and d["max_hp"].shape == (MAX_PLAYERS,)


def test_fresh_state_values():
    st = ro_stats.make_state()
    assert int(st["stat_points"][0]) == S.STAT_POINTS_START
    assert int(st["base_level"][0]) == 1 and int(st["job_level"][0]) == 1
    assert int(st["job"][0]) == S.JOB_NOVICE
    assert int(st["sp"][0]) == int(st["max_sp"][0])
    assert not bool(st["was_dead"].any())


def _credit(w, k, player):
    mvp = w["mvp"]
    return {**w, "mvp": {**mvp, "top_damager": mvp["top_damager"].at[k].set(player),
                         "top_damage": mvp["top_damage"].at[k].set(1)}}


def test_kill_pays_the_damage_credit_holder_only():
    w = _veteran(_veteran(_world(active=(0, 1)), 0), 1)
    w = _credit(_die_monster(w, kind=0), 0, 1)  # slot 1 holds the cumulative-damage credit; slot 0 stands adjacent
    w = _setp(w, 0, kills=1)  # players.kills no longer decides the payout
    out = _run(w)
    assert _st(out, "base_exp", 1) == int(S.MONSTER_BASE_EXP_J[0])
    assert _st(out, "job_exp", 1) == int(S.MONSTER_JOB_EXP_J[0])
    assert _st(out, "base_exp", 0) == 0 and _st(out, "job_exp", 0) == 0
    assert _st(out, "kills_seen", 0) == 1


def test_ranged_kill_pays_from_far_away():
    w = _veteran(_world(active=(3,)), 3)
    w = _credit(_die_monster(w, pos=(200, 200), kind=0), 0, 3)  # corpse 190 cells away: a bolt or an arrow
    out = _run(w)
    assert _st(out, "base_exp", 3) == int(S.MONSTER_BASE_EXP_J[0])
    assert _st(out, "job_exp", 3) == int(S.MONSTER_JOB_EXP_J[0])


def test_credit_to_logged_out_player_pays_nobody():
    w = _veteran(_world(active=(0,)), 0)
    w = _credit(_die_monster(w, kind=0), 0, 5)  # slot 5 is not active
    out = _run(w)
    assert int(out["stats"]["base_exp"].sum()) == 0


def test_orc_pays_orc_exp():
    w = _veteran(_world(), 0)
    w = _credit(_die_monster(w, kind=1), 0, 0)
    out = _run(w)
    assert _st(out, "base_exp") == int(S.MONSTER_BASE_EXP_J[1])
    assert _st(out, "job_exp") == int(S.MONSTER_JOB_EXP_J[1])


def test_party_members_in_the_same_zone_share_with_bonus():
    from game import ro_party
    w = _veteran(_veteran(_world(active=(0, 1)), 0), 1)
    w = _credit(_die_monster(w, kind=1), 0, 0)  # orc: 408 base / 160 job
    pt = ro_party.make_state()
    pt = {**pt, "party_id": pt["party_id"].at[0].set(0).at[1].set(0), "member_count": pt["member_count"].at[0].set(2)}
    w = {**w, "party": pt}
    out = _run(w)
    base = int(S.MONSTER_BASE_EXP_J[1]) * 110 // 100
    job = int(S.MONSTER_JOB_EXP_J[1]) * 110 // 100
    assert _st(out, "base_exp", 0) == base // 2 + base % 2 and _st(out, "base_exp", 1) == base // 2
    assert _st(out, "job_exp", 0) == job // 2 + job % 2 and _st(out, "job_exp", 1) == job // 2


def test_no_died_monster_no_exp():
    w = _setp(_veteran(_world(), 0), 0, kills=1)
    out = _run(w)
    assert _st(out, "base_exp") == 0


def test_level_up_one_gains_stat_points():
    w = _set(_world(), 0, base_exp=S.BASE_EXP_TABLE[1] + 1)
    out = _run(w)
    assert _st(out, "base_level") == 2
    assert _st(out, "base_exp") == 1
    assert _st(out, "stat_points") == S.STAT_POINTS_START + S.STAT_POINTS_PER_LEVEL[2]


def test_no_level_up_below_requirement():
    w = _set(_world(), 0, base_exp=S.BASE_EXP_TABLE[1] - 1)
    out = _run(w)
    assert _st(out, "base_level") == 1 and _st(out, "base_exp") == S.BASE_EXP_TABLE[1] - 1


def test_multi_level_in_one_tick():
    need = S.BASE_EXP_TABLE[1] + S.BASE_EXP_TABLE[2] + S.BASE_EXP_TABLE[3]
    w = _set(_world(), 0, base_exp=need + 5)
    out = _run(w)
    assert _st(out, "base_level") == 4
    assert _st(out, "base_exp") == 5
    gained = sum(S.STAT_POINTS_PER_LEVEL[lv] for lv in (2, 3, 4))
    assert _st(out, "stat_points") == S.STAT_POINTS_START + gained


def test_cap_at_max_level():
    top = S.MAX_BASE_LEVEL
    w = _set(_world(), 0, base_level=top - 1, base_exp=S.BASE_EXP_TABLE[top - 1] + 7)
    out = _run(w)
    assert _st(out, "base_level") == top and _st(out, "base_exp") == 0
    w = _set(_world(), 0, base_level=top, base_exp=12345)
    out = _run(w)
    assert _st(out, "base_level") == top and _st(out, "base_exp") == 0


def test_job_level_up_gives_skill_point():
    w = _set(_world(), 0, job_exp=int(S.JOB_EXP_TABLE[0][1]) + 2)
    out = _run(w)
    assert _st(out, "job_level") == 2
    assert _st(out, "job_exp") == 2
    assert _st(out, "skill_points") == S.SKILL_POINTS_PER_JOB_LEVEL


def test_allot_raises_stat_and_charges_cost():
    w = _set(_world(), 0, stat_points=100)
    out = _run(w, allot={0: S.STAT_STR})
    assert _st(out, "str") == 2
    assert _st(out, "stat_points") == 100 - S.STAT_COST[1]
    # rising cost: a stat at 11 costs STAT_COST[11], more than at 1
    w = _set(_world(), 0, stat_points=100, dex=11)
    out = _run(w, allot={0: S.STAT_DEX})
    assert _st(out, "dex") == 12
    assert _st(out, "stat_points") == 100 - S.STAT_COST[11]
    assert S.STAT_COST[11] > S.STAT_COST[1]
    assert _st(out, "str") == 1


def test_allot_refused_at_max_stat():
    w = _set(_world(), 0, stat_points=500, luk=S.MAX_STAT)
    out = _run(w, allot={0: S.STAT_LUK})
    assert _st(out, "luk") == S.MAX_STAT and _st(out, "stat_points") == 500


def test_allot_refused_without_points():
    w = _set(_world(), 0, stat_points=S.STAT_COST[1] - 1)
    out = _run(w, allot={0: S.STAT_AGI})
    assert _st(out, "agi") == 1 and _st(out, "stat_points") == S.STAT_COST[1] - 1


def test_allot_none_ignored():
    w = _set(_world(), 0, stat_points=100)
    out = _run(w)
    for k in ("str", "agi", "vit", "int", "dex", "luk"):
        assert _st(out, k) == 1
    assert _st(out, "stat_points") == 100


def test_job_change_at_novice_job_10():
    w = _set(_world(), 0, job_level=S.MAX_JOB_LEVEL_NOVICE, job_exp=0, skill_points=7)
    out = _run(w, job={0: S.JOB_ARCHER})
    assert _st(out, "job") == S.JOB_ARCHER
    assert _st(out, "job_level") == 1 and _st(out, "job_exp") == 0
    assert _st(out, "skill_points") == 7


def test_job_change_refused_below_job_10_or_when_not_novice():
    w = _set(_world(), 0, job_level=S.MAX_JOB_LEVEL_NOVICE - 1)
    out = _run(w, job={0: S.JOB_MAGE})
    assert _st(out, "job") == S.JOB_NOVICE and _st(out, "job_level") == S.MAX_JOB_LEVEL_NOVICE - 1
    w = _set(_world(), 0, job=S.JOB_SWORDSMAN, job_level=10)
    out = _run(w, job={0: S.JOB_MAGE})
    assert _st(out, "job") == S.JOB_SWORDSMAN and _st(out, "job_level") == 10
    w = _set(_world(), 0, job_level=S.MAX_JOB_LEVEL_NOVICE)
    out = _run(w, job={0: S.JOB_NOVICE})
    assert _st(out, "job") == S.JOB_NOVICE


def test_death_penalty_at_level_10():
    lvl = S.DEATH_PENALTY_MIN_LEVEL
    loss = S.BASE_EXP_TABLE[lvl] * S.DEATH_PENALTY_PERMILLE // 1000
    start = S.BASE_EXP_TABLE[lvl] // 2
    w = _setp(_set(_world(), 0, base_level=lvl, base_exp=start, job=S.JOB_SWORDSMAN), 0, deaths=1, dead=True)
    out = _run(w)
    assert loss > 0
    assert _st(out, "base_exp") == start - loss
    assert _st(out, "base_level") == lvl
    assert _st(out, "deaths_seen") == 1
    # a second tick with no new death costs nothing more
    out2 = PHASE(out, _inputs())
    assert _st(out2, "base_exp") == start - loss


def test_death_penalty_floors_at_zero():
    lvl = S.DEATH_PENALTY_MIN_LEVEL
    w = _setp(_set(_world(), 0, base_level=lvl, base_exp=0, job=S.JOB_SWORDSMAN), 0, deaths=1, dead=True)
    assert _st(_run(w), "base_exp") == 0
    w = _setp(_set(_world(), 0, base_level=lvl, base_exp=1, job=S.JOB_SWORDSMAN), 0, deaths=1, dead=True)
    out = _run(w)
    assert _st(out, "base_exp") == 0 and _st(out, "base_level") == lvl


def test_no_death_penalty_for_novice():
    start = S.BASE_EXP_TABLE[20] // 2
    w = _setp(_set(_world(), 0, base_level=20, base_exp=start, job_exp=5), 0, deaths=1, dead=True)
    out = _run(w)
    assert _st(out, "base_exp") == start and _st(out, "job_exp") == 5


def test_death_penalty_costs_job_exp_for_swordsman():
    w = _set(_world(), 0, base_level=20, base_exp=1000, job=S.JOB_SWORDSMAN, job_level=20, job_exp=1000)
    out = _run(_setp(w, 0, deaths=1, dead=True))
    assert _st(out, "base_exp") == 1000 - S.BASE_EXP_TABLE[20] * 10 // 1000
    assert _st(out, "job_exp") == 1000 - S.JOB_EXP_TABLE[S.JOB_SWORDSMAN][20] * 10 // 1000
    assert _st(out, "job_exp") < 1000


def test_swordsman_sword_aspd_from_weapon_column():
    from game.data import items as I
    from game.ro_items import make_state as make_items
    w = _set(_world(), 0, job=S.JOB_SWORDSMAN)
    bare = ro_stats.derived(w)
    it = make_items()
    it["equip"] = it["equip"].at[0, I.EQ_WEAPON].set(0)
    it["inv_count"] = it["inv_count"].at[0, 0].set(1)
    w = {**w, "items": it, "players": {**w["players"], "inv": w["players"]["inv"].at[0, 0].set(I.ITEM_SWORD)}}
    d = ro_stats.derived(w)
    assert int(bare["aspd"][0]) == 200 - 40 * 995 // 1000  # fist 160
    assert int(d["aspd"][0]) == 200 - 55 * 995 // 1000  # sword column 145


def test_hp_regen_cadence():
    w = _set(_world(tick=S.HP_REGEN_TICKS), 0, base_level=1)
    mh = int(ro_stats.derived(w)["max_hp"][0])
    want = 1 + mh // 200 + 1 // 5
    nosit = jnp.zeros((MAX_PLAYERS,), bool)
    assert int(ro_stats.hp_regen(w, nosit)[0]) == want
    assert int(ro_stats.hp_regen({**w, "tick": jnp.int32(S.HP_REGEN_TICKS + 1)}, nosit)[0]) == 0
    half = {**w, "tick": jnp.int32(S.HP_REGEN_TICKS // 2)}
    assert int(ro_stats.hp_regen(half, nosit)[0]) == 0
    sit = jnp.zeros((MAX_PLAYERS,), bool).at[0].set(True)
    assert int(ro_stats.hp_regen(half, sit)[0]) == want


def test_sp_regen_on_regen_tick_only():
    max_sp = int(ro_stats.derived(_world())["max_sp"][0])
    gain = 1 + max_sp // 100 + 1 // 6
    w = _set(_world(tick=S.SP_REGEN_TICKS), 0, sp=0, max_sp=max_sp)
    assert _st(_run(w), "sp") == min(gain, max_sp)
    w = _set(_world(tick=S.SP_REGEN_TICKS + 1), 0, sp=0, max_sp=max_sp)
    assert _st(_run(w), "sp") == 0


def test_sp_full_on_respawn():
    max_sp = int(ro_stats.derived(_world())["max_sp"][0])
    w = _setp(_set(_world(tick=1), 0, sp=0), 0, dead=True)
    mid = _run(w)
    assert bool(mid["stats"]["was_dead"][0])
    assert _st(mid, "sp") == 0
    mid = {**mid, "players": {**mid["players"], "dead": mid["players"]["dead"].at[0].set(False)}}
    out = PHASE(mid, _inputs())
    assert _st(out, "sp") == max_sp
    assert not bool(out["stats"]["was_dead"][0])


def test_sp_never_above_max():
    max_sp = int(ro_stats.derived(_world())["max_sp"][0])
    w = _set(_world(tick=S.SP_REGEN_TICKS), 0, sp=max_sp, max_sp=max_sp)
    assert _st(_run(w), "sp") == max_sp
    w = _set(_world(tick=1), 0, sp=max_sp + 50)
    assert _st(_run(w), "sp") == max_sp


def test_inactive_rows_untouched():
    w = _world(active=(0,))
    w = _set(w, 1, base_exp=S.BASE_EXP_TABLE[1] + 3, stat_points=50, job_level=10, sp=0)
    w = _setp(_setp(w, 1, kills=1), 1, deaths=1)
    w = _die_monster(w)
    before = {k: np.array(v[1]) for k, v in w["stats"].items()}
    out = _run(w, allot={1: S.STAT_STR}, job={1: S.JOB_MAGE})
    for k, v in before.items():
        assert np.array_equal(np.array(out["stats"][k][1]), v), k


def test_jit_matches_eager():
    w = _veteran(_world(active=(0, 1)), 0)
    w = _set(w, 1, stat_points=40, base_exp=S.BASE_EXP_TABLE[1] + 1)
    w = _setp(_die_monster(w), 0, kills=1)
    inp = _inputs(allot={1: S.STAT_VIT})
    eager = ro_stats.phase(w, inp)
    jitted = PHASE(w, inp)
    for k in eager["stats"]:
        assert np.array_equal(np.array(eager["stats"][k]), np.array(jitted["stats"][k])), k
    assert int(jitted["stats"]["vit"][1]) == 2


def test_vmap_over_two_worlds():
    a = _set(_world(), 0, stat_points=100)
    b = _set(_world(), 0, base_exp=S.BASE_EXP_TABLE[1] + 1)
    wb = jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y]), a, b)
    ib = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), _inputs(allot={0: S.STAT_STR}))
    out = jax.jit(jax.vmap(ro_stats.phase))(wb, ib)
    assert out["stats"]["str"].shape == (2, MAX_PLAYERS)
    assert int(out["stats"]["str"][0, 0]) == 2
    assert int(out["stats"]["base_level"][1, 0]) == 2
