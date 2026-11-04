"""Round 3 (ro3-exp): kills by skill from range pay exp and zeny to the caster through the damage credit.

Runs the real phases in step order (skills -> combat -> loot -> stats -> mvp) on a world with every block, so the
test covers the wiring: ro_skills_effects reports hits through ro_mvp.record_damage, loot and ro_stats read
ro_mvp.credit_of, ro_mvp.phase clears the credit afterwards.
"""
import jax
import jax.numpy as jnp

from game import combat, loot, ro_mvp, ro_skills, ro_stats
from game.config import MAX_PLAYERS
from game.data import skills as sk
from game.data import stats as S
from game.data.items import ITEM_GOLD_COIN
from game.state import empty_inputs, make_empty_world

MP = MAX_PLAYERS
CASTER, OTHER = 3, 0
MON = 0


def _world():
    w = make_empty_world(jax.random.PRNGKey(1))
    p, m = w["players"], w["monsters"]
    for slot, pos in ((CASTER, (20, 20)), (OTHER, (26, 21))):
        p = {**p, "pos": p["pos"].at[slot].set(jnp.array(pos, jnp.int32)), "active": p["active"].at[slot].set(True),
             "hp": p["hp"].at[slot].set(40)}
    m = {**m, "pos": m["pos"].at[MON].set(jnp.array([26, 20], jnp.int32)), "hp": m["hp"].at[MON].set(1),
         "kind": m["kind"].at[MON].set(0), "active": m["active"].at[MON].set(True)}
    st = w["stats"]
    st = {**st, "sp": st["sp"].at[CASTER].set(50), "max_sp": st["max_sp"].at[CASTER].set(50),
          "base_level": st["base_level"].at[CASTER].set(20).at[OTHER].set(20),
          "job": st["job"].at[CASTER].set(S.JOB_MAGE), "job_level": st["job_level"].at[CASTER].set(20),
          "int": st["int"].at[CASTER].set(50)}  # matk ~100 so a level-1 bolt deals damage
    skl = w["skills"]
    skl = {**skl, "learned": skl["learned"].at[CASTER, sk.SK_FIRE_BOLT].set(1)}
    return {**w, "players": p, "monsters": m, "stats": st, "skills": skl, "tick": jnp.int32(1)}


def _inputs(skill=0):
    i = empty_inputs()
    return {**i, "skill": i["skill"].at[CASTER].set(skill),
            "target": i["target"].at[CASTER].set(MP + MON)}


def _phases(w, i):
    w = ro_skills.phase(w, i)
    w = combat.combat(w, i)
    w = loot.loot(w)
    w = ro_stats.phase(w, i)
    return ro_mvp.phase(w)


def test_fire_bolt_kill_from_range_pays_exp_and_zeny_to_the_caster():
    run = jax.jit(_phases)
    w = _world()
    w = run(w, _inputs(sk.SK_FIRE_BOLT))
    cast = int(sk.SKILL_CAST[sk.SK_FIRE_BOLT, 1])
    for _ in range(cast + 1):
        if bool(w["monsters"]["died"][MON]):
            break
        w = {**w, "monsters": {**w["monsters"], "died": jnp.zeros_like(w["monsters"]["died"])},
             "tick": w["tick"] + 1}
        w = run(w, _inputs())
    assert not bool(w["monsters"]["active"][MON]), "the bolt should have killed the 1-hp Poring"
    assert int(w["stats"]["base_exp"][CASTER]) == S.MONSTER_BASE_EXP[0]
    assert int(w["stats"]["job_exp"][CASTER]) == S.MONSTER_JOB_EXP[0]
    assert int(w["stats"]["base_exp"][OTHER]) == 0  # the adjacent bystander gets nothing
    assert int(w["items"]["zeny"][CASTER]) > 0 and int(w["items"]["zeny"][OTHER]) == 0
    assert int((w["drops"]["item"] == ITEM_GOLD_COIN).sum()) == 0  # the zeny went to the caster, not the ground
    assert int(w["mvp"]["top_damager"][MON]) == -1  # the credit is cleared after the payout
