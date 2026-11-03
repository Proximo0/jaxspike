"""Bot skill policy: learn / skill / target inputs by job. Pure JAX, read-only on the world.

Learning follows a per-job plan (JOB_PLAN: (skill, level) steps in order, one point per tick while skill points
last). Casting, one skill per tick, first match wins:
1. Heal (Acolyte): self below HEAL_PCT of max_hp, else the most hurt living party member in range below HEAL_PCT.
2. First Aid (Novice) below HEAL_PCT.
3. A self buff whose timer ran out (Blessing / Increase AGI for Acolytes outside town; Endure for Swordsmen in a
   fight), keeping SP for one Heal.
4. The job's attack skill on the nearest monster: the highest-learned of its attack skills, when the monster is
   within min(skill range, RANGED_CAST_MAX). RANGED_CAST_MAX is 1 because game/ro_stats.py pays kill exp only for
   corpses at distance <= 1: a bolt that kills from 9 cells away pays nothing (reported defect); raise it to 9 when
   that is fixed and mages/archers will cast from range.
SP comes from stats.sp when the stats block exists (the authoritative pool), else skills.sp.
"""
import jax.numpy as jnp
import numpy as np

from game.config import MAX_PLAYERS, PLAYER_MAX_HP
from game.data import skills as S
from game.data.skills import (JOB_ACOLYTE, JOB_ARCHER, JOB_MAGE, JOB_MERCHANT, JOB_NOVICE, JOB_SWORDSMAN,
                              JOB_THIEF, N_JOBS)
from game.spatial import build_grid, nearest_in_block

LEARN_TARGET_LEVEL = 5  # kept for callers: the first plan step of every job takes its first skill to this level
HEAL_PCT = 60           # heal self or a party member below this percent of max_hp
RANGED_CAST_MAX = 1     # see the module docstring
HEAL_RANGE = 9

# Per job: learn plan, (skill, target level) in order. Passives count: the plan spends every point.
_PLANS = {
    JOB_NOVICE: ((S.SK_FIRST_AID, 1), (S.SK_BASIC_SKILL, 9)),
    JOB_SWORDSMAN: ((S.SK_BASH, 5), (S.SK_ENDURE, 1), (S.SK_HP_RECOVERY, 5), (S.SK_BASH, 10),
                    (S.SK_SWORD_MASTERY, 10), (S.SK_HP_RECOVERY, 10), (S.SK_MAGNUM_BREAK, 10), (S.SK_ENDURE, 10)),
    JOB_MAGE: ((S.SK_FIRE_BOLT, 5), (S.SK_COLD_BOLT, 5), (S.SK_LIGHTNING_BOLT, 5), (S.SK_FIRE_BOLT, 10),
               (S.SK_SOUL_STRIKE, 10), (S.SK_COLD_BOLT, 10), (S.SK_LIGHTNING_BOLT, 10)),
    JOB_ARCHER: ((S.SK_DOUBLE_STRAFE, 5), (S.SK_OWLS_EYE, 5), (S.SK_VULTURES_EYE, 5), (S.SK_DOUBLE_STRAFE, 10),
                 (S.SK_OWLS_EYE, 10), (S.SK_VULTURES_EYE, 10), (S.SK_ARROW_SHOWER, 10)),
    JOB_ACOLYTE: ((S.SK_HEAL, 5), (S.SK_INCREASE_AGI, 3), (S.SK_BLESSING, 3), (S.SK_HEAL, 10),
                  (S.SK_INCREASE_AGI, 10), (S.SK_BLESSING, 10), (S.SK_ANGELUS, 10)),
    JOB_THIEF: ((S.SK_ENVENOM, 5), (S.SK_DOUBLE_ATTACK, 10), (S.SK_ENVENOM, 10)),
    JOB_MERCHANT: ((S.SK_MAMMONITE, 10), (S.SK_DISCOUNT, 10)),
}
# Per job: attack skills, best first (the first learned one is cast).
_ATTACKS = {
    JOB_NOVICE: (), JOB_SWORDSMAN: (S.SK_BASH,),
    JOB_MAGE: (S.SK_FIRE_BOLT, S.SK_COLD_BOLT, S.SK_LIGHTNING_BOLT, S.SK_SOUL_STRIKE),
    JOB_ARCHER: (S.SK_DOUBLE_STRAFE,), JOB_ACOLYTE: (), JOB_THIEF: (S.SK_ENVENOM,), JOB_MERCHANT: (S.SK_MAMMONITE,),
}
# Per job: self buffs kept up, (skill, only while fighting).
_BUFFS = {JOB_SWORDSMAN: ((S.SK_ENDURE, True),),
          JOB_ACOLYTE: ((S.SK_BLESSING, False), (S.SK_INCREASE_AGI, False))}


def _pad(rows, width, fill):
    out = np.full((N_JOBS, width) + np.shape(fill), fill, np.int32)
    for j, r in rows.items():
        if r:
            out[j, :len(r)] = r
    return jnp.asarray(out)


_PLAN_W = max(len(p) for p in _PLANS.values())
JOB_PLAN = _pad(_PLANS, _PLAN_W, (0, 0))                       # [N_JOBS, W, 2]
JOB_ATTACKS = _pad(_ATTACKS, 4, 0)                             # [N_JOBS, 4], 0 = none
JOB_BUFFS = _pad({j: [(s, int(f)) for s, f in b] for j, b in _BUFFS.items()}, 2, (0, 0))  # [N_JOBS, 2, 2]
# First castable skill per job (older callers and tests).
JOB_FIRST_SKILL = jnp.array(
    [S.SK_FIRST_AID, S.SK_BASH, S.SK_FIRE_BOLT, S.SK_DOUBLE_STRAFE, S.SK_HEAL, S.SK_ENVENOM, S.SK_MAMMONITE],
    jnp.int32)

_SLOT = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)


def _jobs(world):
    # Static pytree-structure check, jit-safe.
    if "stats" in world and "job" in world["stats"]:
        return world["stats"]["job"].astype(jnp.int32)
    return _SLOT % N_JOBS


def _sp(world):
    if "stats" in world and "sp" in world["stats"]:
        return world["stats"]["sp"].astype(jnp.int32)
    return world["skills"]["sp"].astype(jnp.int32)


def _max_hp(world):
    if "stats" in world:
        from game import ro_stats
        return jnp.maximum(ro_stats.derived(world)["max_hp"].astype(jnp.int32), 1)
    return jnp.full((MAX_PLAYERS,), PLAYER_MAX_HP, jnp.int32)


def learn_input(world):
    """int32 [MAX_PLAYERS]: the first unfinished plan step's skill (0 when the plan is done or inactive).

    Without a skill_points ledger (early worlds) the engine lets anyone learn; the plan is then capped at the
    first step so bots do not max every skill on tick one.
    """
    pl, sk = world["players"], world["skills"]
    job = _jobs(world)
    plan = JOB_PLAN[job]                                   # [P, W, 2]
    sid, target = plan[..., 0], plan[..., 1]
    lvl = jnp.take_along_axis(sk["learned"], sid, axis=1)  # [P, W]
    todo = (sid > 0) & (lvl < target)
    has_points = "stats" in world and "skill_points" in world["stats"]
    if not has_points:
        todo = todo & (jnp.arange(_PLAN_W)[None] == 0) & (lvl < LEARN_TARGET_LEVEL)
    first = jnp.argmax(todo, axis=1)
    out = jnp.where(todo.any(axis=1), sid[_SLOT, first], 0)
    if has_points:
        out = jnp.where(world["stats"]["skill_points"] > 0, out, 0)
    return jnp.where(pl["active"], out, 0).astype(jnp.int32)


def _ready(world, skill, sp):
    """bool [P]: skill (int32 [P], 0 = none) is learned, off cooldown, affordable, and the caster is idle."""
    sk = world["skills"]
    sc = jnp.clip(skill, 0, S.N_SKILLS - 1)
    lvl = sk["learned"][_SLOT, sc]
    ok = (skill > 0) & (lvl > 0) & (sk["casting"] == 0) & (sk["cooldown"][_SLOT, sc] == 0)
    ok = ok & (sp >= S.SKILL_SP[sc, jnp.clip(lvl, 0, S.MAX_SKILL_LEVEL)])
    if "after_delay" in sk:
        ok = ok & (sk["after_delay"] == 0)
    return ok


def _range(world, skill):
    sc = jnp.clip(skill, 0, S.N_SKILLS - 1)
    return S.SKILL_RANGE[sc, jnp.clip(world["skills"]["learned"][_SLOT, sc], 0, S.MAX_SKILL_LEVEL)]


def _heal_cost(world):
    lvl = jnp.clip(world["skills"]["learned"][:, S.SK_HEAL], 0, S.MAX_SKILL_LEVEL)
    return jnp.where(lvl > 0, S.SKILL_SP[S.SK_HEAL, lvl], 0)


def cast_inputs(world):
    """Return (skill int32 [P], target int32 [P]): 0 / -1 = none."""
    pl, mon, sk = world["players"], world["monsters"], world["skills"]
    live = pl["active"] & ~pl["dead"]
    job = _jobs(world)
    sp = _sp(world)
    hp, mx = pl["hp"].astype(jnp.int32), _max_hp(world)
    low = live & (hp * 100 < HEAL_PCT * mx)
    on_pz = world["pz"][pl["pos"][:, 0], pl["pos"][:, 1]]

    table, _ = build_grid(mon["pos"], mon["active"])
    idx, dist = nearest_in_block(table, pl["pos"], mon["pos"])
    fighting = dist <= 1

    # 1. Heal: self, else the most hurt party member in range (a [P, P] mask; party ids from the party block)
    heal = jnp.full((MAX_PLAYERS,), S.SK_HEAL, jnp.int32)
    can_heal = live & (job == JOB_ACOLYTE) & _ready(world, heal, sp)
    ally_t = jnp.full((MAX_PLAYERS,), -1, jnp.int32)
    if "party" in world:
        pid = world["party"]["party_id"]
        d = jnp.sum(jnp.abs(pl["pos"][:, None, :] - pl["pos"][None, :, :]), axis=2)
        mate = (pid[:, None] >= 0) & (pid[:, None] == pid[None, :]) & low[None, :] & (d <= HEAL_RANGE)
        frac = jnp.where(mate, hp[None, :] * 1000 // mx[None, :], 1 << 30)
        best = jnp.argmin(frac, axis=1).astype(jnp.int32)
        ally_t = jnp.where(mate.any(axis=1), best, -1)
    heal_t = jnp.where(low, _SLOT, ally_t)
    do_heal = can_heal & (heal_t >= 0)

    # 2. First Aid
    aid = jnp.full((MAX_PLAYERS,), S.SK_FIRST_AID, jnp.int32)
    do_aid = live & low & _ready(world, aid, sp)

    # 3. self buffs (keep SP for one Heal)
    reserve = jnp.where(job == JOB_ACOLYTE, _heal_cost(world), 0)
    buffs = JOB_BUFFS[job]  # [P, 2, 2]
    b_skill = jnp.zeros((MAX_PLAYERS,), jnp.int32)
    for k in range(buffs.shape[1]):
        sid, only_fight = buffs[:, k, 0], buffs[:, k, 1] > 0
        bid = S.SKILL_BUFF[jnp.clip(sid, 0, S.N_SKILLS - 1)]
        off = sk["buffs"][_SLOT, bid] == 0
        sc = jnp.clip(sid, 0, S.N_SKILLS - 1)
        cost = S.SKILL_SP[sc, jnp.clip(sk["learned"][_SLOT, sc], 0, S.MAX_SKILL_LEVEL)]
        want = (live & ~on_pz & off & _ready(world, sid, sp) & (sp - cost >= reserve)
                & jnp.where(only_fight, fighting, True))
        b_skill = jnp.where((b_skill == 0) & want, sid, b_skill)
    do_buff = b_skill > 0

    # 4. attack skill on the nearest monster
    atks = JOB_ATTACKS[job]  # [P, 4]
    lv = jnp.take_along_axis(sk["learned"], atks, axis=1)
    has = (atks > 0) & (lv > 0)
    a_skill = jnp.where(has.any(axis=1), atks[_SLOT, jnp.argmax(has, axis=1)], 0).astype(jnp.int32)
    in_range = (dist > 0) & (dist <= jnp.minimum(_range(world, a_skill), RANGED_CAST_MAX))
    a_cost = S.SKILL_SP[a_skill, jnp.clip(sk["learned"][_SLOT, a_skill], 0, S.MAX_SKILL_LEVEL)]
    do_atk = live & ~on_pz & in_range & _ready(world, a_skill, sp) & (sp - a_cost >= reserve)

    skill = jnp.where(do_heal, S.SK_HEAL, jnp.where(do_aid, S.SK_FIRST_AID,
                      jnp.where(do_buff, b_skill, jnp.where(do_atk, a_skill, 0))))
    target = jnp.where(do_heal, heal_t, jnp.where(do_aid | do_buff, _SLOT,
                       jnp.where(do_atk, MAX_PLAYERS + idx, -1)))
    return skill.astype(jnp.int32), target.astype(jnp.int32)


def bot_inputs(world, key):
    """Return the base bot inputs plus int32 learn / skill / target, each [MAX_PLAYERS]."""
    from bots import policies
    out = policies.bot_inputs(world, key)
    skill, target = cast_inputs(world)
    return {**out, "learn": learn_input(world), "skill": skill, "target": target}
