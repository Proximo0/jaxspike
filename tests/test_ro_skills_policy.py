"""Tests for bots/ro_skills_policy.py: learn plans, heals, buffs, attack skills by job."""
import jax
import jax.numpy as jnp

from bots import ro_skills_policy as P
from game import ro_party, ro_stats
from game.config import MAX_PLAYERS
from game.data import skills as S
from game.state import make_empty_world

KEY = jax.random.PRNGKey(0)


def world(job=S.JOB_SWORDSMAN, pos=(100, 100), hp_pct=100, sp=200, points=0, learned=(), monster=None, slot=0):
    w = make_empty_world(KEY)
    st = {**w["stats"], "job": w["stats"]["job"].at[slot].set(job), "sp": w["stats"]["sp"].at[slot].set(sp),
          "skill_points": w["stats"]["skill_points"].at[slot].set(points)}
    sk = w["skills"]
    for sid, lv in learned:
        sk = {**sk, "learned": sk["learned"].at[slot, sid].set(lv)}
    p = w["players"]
    p = {**p, "pos": p["pos"].at[slot].set(jnp.array(pos)), "active": p["active"].at[slot].set(True)}
    w = {**w, "stats": st, "skills": sk, "players": p}
    mx = int(ro_stats.derived(w)["max_hp"][slot])
    w["players"] = {**p, "hp": p["hp"].at[slot].set(mx * hp_pct // 100)}
    if monster is not None:
        m = w["monsters"]
        w["monsters"] = {**m, "pos": m["pos"].at[0].set(jnp.array(monster)), "active": m["active"].at[0].set(True)}
    return w


_cast = jax.jit(P.cast_inputs)
_learn = jax.jit(P.learn_input)


def test_learn_follows_the_job_plan_while_points_last():
    assert int(_learn(world(S.JOB_SWORDSMAN, points=3))[0]) == S.SK_BASH
    assert int(_learn(world(S.JOB_SWORDSMAN, points=3, learned=[(S.SK_BASH, 5)]))[0]) == S.SK_ENDURE
    assert int(_learn(world(S.JOB_MAGE, points=1, learned=[(S.SK_FIRE_BOLT, 5)]))[0]) == S.SK_COLD_BOLT
    assert int(_learn(world(S.JOB_NOVICE, points=1))[0]) == S.SK_FIRST_AID
    assert int(_learn(world(S.JOB_SWORDSMAN, points=0))[0]) == 0


def test_learn_without_a_points_ledger_caps_at_the_first_step():
    w = world(S.JOB_ARCHER)
    w["stats"] = {k: v for k, v in w["stats"].items() if k != "skill_points"}
    assert int(_learn(w)[0]) == S.SK_DOUBLE_STRAFE
    w["skills"] = {**w["skills"], "learned": w["skills"]["learned"].at[0, S.SK_DOUBLE_STRAFE].set(5)}
    assert int(_learn(w)[0]) == 0


def test_melee_skill_on_an_adjacent_monster():
    skill, target = _cast(world(S.JOB_SWORDSMAN, learned=[(S.SK_BASH, 3)], monster=(101, 100)))
    assert int(skill[0]) == S.SK_BASH and int(target[0]) == MAX_PLAYERS + 0
    skill, _ = _cast(world(S.JOB_SWORDSMAN, learned=[(S.SK_BASH, 3)], monster=(104, 100)))
    assert int(skill[0]) == 0


def test_mage_casts_its_best_learned_bolt_and_archer_strafes():
    skill, target = _cast(world(S.JOB_MAGE, learned=[(S.SK_COLD_BOLT, 2)], monster=(100, 101)))
    assert int(skill[0]) == S.SK_COLD_BOLT and int(target[0]) == MAX_PLAYERS
    skill, _ = _cast(world(S.JOB_ARCHER, learned=[(S.SK_DOUBLE_STRAFE, 1)], monster=(99, 100)))
    assert int(skill[0]) == S.SK_DOUBLE_STRAFE


def test_no_cast_without_sp_or_on_cooldown_or_in_town():
    assert int(_cast(world(S.JOB_SWORDSMAN, sp=0, learned=[(S.SK_BASH, 3)], monster=(101, 100)))[0][0]) == 0
    w = world(S.JOB_SWORDSMAN, learned=[(S.SK_BASH, 3)], monster=(101, 100))
    w["skills"] = {**w["skills"], "cooldown": w["skills"]["cooldown"].at[0, S.SK_BASH].set(2)}
    assert int(_cast(w)[0][0]) == 0
    w = world(S.JOB_SWORDSMAN, learned=[(S.SK_BASH, 3)], monster=(101, 100))
    w["pz"] = w["pz"].at[100, 100].set(True)
    assert int(_cast(w)[0][0]) == 0


def test_sp_comes_from_the_stats_pool():
    w = world(S.JOB_SWORDSMAN, sp=0, learned=[(S.SK_BASH, 3)], monster=(101, 100))
    w["skills"] = {**w["skills"], "sp": w["skills"]["sp"].at[0].set(999)}  # the stale pool must not count
    assert int(_cast(w)[0][0]) == 0


def test_acolyte_heals_itself_when_hurt():
    skill, target = _cast(world(S.JOB_ACOLYTE, hp_pct=30, learned=[(S.SK_HEAL, 3)]))
    assert int(skill[0]) == S.SK_HEAL and int(target[0]) == 0


def test_acolyte_heals_a_hurt_party_member_in_range():
    w = world(S.JOB_ACOLYTE, learned=[(S.SK_HEAL, 3)])
    p = w["players"]
    p = {**p, "pos": p["pos"].at[3].set(jnp.array([104, 100])), "active": p["active"].at[3].set(True),
         "hp": p["hp"].at[3].set(10)}
    w["players"] = p
    w["party"] = {**w["party"], "party_id": w["party"]["party_id"].at[0].set(2).at[3].set(2)}
    skill, target = _cast(w)
    assert int(skill[0]) == S.SK_HEAL and int(target[0]) == 3
    w["party"] = {**w["party"], "party_id": w["party"]["party_id"].at[3].set(5)}  # not a member: no heal
    assert int(_cast(w)[0][0]) != S.SK_HEAL


def test_acolyte_keeps_blessing_up_outside_town_and_swordsman_endures_in_a_fight():
    skill, target = _cast(world(S.JOB_ACOLYTE, learned=[(S.SK_HEAL, 1), (S.SK_BLESSING, 1)]))
    assert int(skill[0]) == S.SK_BLESSING and int(target[0]) == 0
    w = world(S.JOB_ACOLYTE, learned=[(S.SK_HEAL, 1), (S.SK_BLESSING, 1)])
    w["skills"] = {**w["skills"], "buffs": w["skills"]["buffs"].at[0, S.BUFF_BLESSING].set(50)}
    assert int(_cast(w)[0][0]) == 0
    assert int(_cast(world(S.JOB_SWORDSMAN, learned=[(S.SK_ENDURE, 1)]))[0][0]) == 0
    skill, _ = _cast(world(S.JOB_SWORDSMAN, learned=[(S.SK_ENDURE, 1)], monster=(101, 100)))
    assert int(skill[0]) == S.SK_ENDURE


def test_novice_first_aid_when_hurt():
    skill, target = _cast(world(S.JOB_NOVICE, hp_pct=30, learned=[(S.SK_FIRST_AID, 1)]))
    assert int(skill[0]) == S.SK_FIRST_AID and int(target[0]) == 0


def test_bot_inputs_shapes_jit_vmap():
    w = world(S.JOB_SWORDSMAN, learned=[(S.SK_BASH, 3)], monster=(101, 100))
    ws = jax.tree.map(lambda a: jnp.stack([a, a]), w)
    out = jax.jit(jax.vmap(P.bot_inputs))(ws, jax.random.split(KEY, 2))
    for k in ("learn", "skill", "target"):
        assert out[k].shape == (2, MAX_PLAYERS) and out[k].dtype == jnp.int32
    assert int(out["skill"][1, 0]) == S.SK_BASH


def test_party_block_is_present_in_the_empty_world():
    assert "party" in make_empty_world(KEY) and ro_party.MAX_PARTIES > 0
