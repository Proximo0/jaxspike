"""RO stats, jobs and exp (milestone M1). Contract: docs/m1_stats_brief.md and docs/ro_contract.md.

A per-player stats block `world["stats"]` (all arrays [MAX_PLAYERS]), the pinned `derived(world)` read used by other
systems, and `phase(world, inputs)` which runs after loot and before survival. Everything is a fixed-shape array op:
masks over `players.active`, int32 math, tables indexed by id, scatters with mode="drop", no Python `if` on traced
values and no loops over entities (the only loops are fixed-count lax.fori_loops over whole arrays).
"""
import jax.numpy as jnp
from jax import lax

from game.config import MAX_MONSTERS, MAX_PLAYERS, TICK_MS
from game.data import items as item_data
from game import ro_mvp, ro_party
from game.data.stats import (  # noqa: F401  (ids and tables re-exported for the other M1 files)
    BASE_EXP_TABLE_J,
    BASE_HP_TABLE_J,
    BASE_SP_TABLE_J,
    DEATH_PENALTY_PERMILLE,
    JOB_ACOLYTE,
    JOB_ARCHER,
    HP_REGEN_SIT_TICKS,
    HP_REGEN_TICKS,
    JOB_BASE_HP_TABLE_J,
    JOB_BASE_SP_TABLE_J,
    JOB_WEAPON_ASPD_J,
    JOB_EXP_TABLE_J,
    JOB_HP_FACTOR_J,
    JOB_MAGE,
    JOB_MERCHANT,
    JOB_NOVICE,
    JOB_SP_FACTOR_J,
    JOB_SWORDSMAN,
    JOB_THIEF,
    MAX_BASE_LEVEL,
    MAX_JOB_LEVEL_J,
    MAX_JOB_LEVEL_NOVICE,
    MAX_STAT,
    MONSTER_BASE_EXP_J,
    MONSTER_JOB_EXP_J,
    NUM_JOBS,
    NUM_STATS,
    SKILL_POINTS_PER_JOB_LEVEL,
    SP_REGEN_TICKS,
    STAT_AGI,
    STAT_COST_J,
    STAT_DEX,
    STAT_INT,
    STAT_LUK,
    STAT_POINTS_PER_LEVEL_J,
    STAT_POINTS_START,
    STAT_STR,
    STAT_VIT,
)

MAX_LEVELS_PER_TICK = 4  # fixed fori_loop trip count for base and job level-ups (more exp just carries over)

_I32 = jnp.int32
_STAT_KEYS = ("str", "agi", "vit", "int", "dex", "luk")  # order == STAT_STR..STAT_LUK


def _zeros():
    return jnp.zeros((MAX_PLAYERS,), _I32)


def _fresh_max_sp():
    """max_sp of a fresh Novice (level 1, int 1), same formula as derived()."""
    return int(JOB_BASE_SP_TABLE_J[JOB_NOVICE, 1]) * (100 + 1) // 100


def make_state():
    """The block stored under world["stats"]: every player slot starts as a fresh Novice."""
    one = jnp.ones((MAX_PLAYERS,), _I32)
    sp0 = jnp.full((MAX_PLAYERS,), _fresh_max_sp(), _I32)
    s = {k: one for k in _STAT_KEYS}
    s.update(
        base_level=one,
        job_level=one,
        base_exp=_zeros(),
        job_exp=_zeros(),
        job=_zeros(),
        stat_points=jnp.full((MAX_PLAYERS,), STAT_POINTS_START, _I32),
        skill_points=_zeros(),
        sp=sp0,
        max_sp=sp0,
        kills_seen=_zeros(),
        deaths_seen=_zeros(),
        was_dead=jnp.zeros((MAX_PLAYERS,), jnp.bool_),
    )
    return s


def fresh_rows(stats, mask):
    """Reset the rows where mask (bool [MAX_PLAYERS]) is True to a fresh Novice (ledgers zeroed too).

    For the server's login/logout. Rows where mask is False are returned unchanged.
    """
    fresh = make_state()
    return {k: jnp.where(mask, fresh[k], stats[k]) for k in stats}


def derived(world):
    """Pinned cross-system read: combat/items/skills use these. Every value is int32 [MAX_PLAYERS].

    Reads world["stats"] only. All math is integer; products stay far below 2^31.
    """
    s = world["stats"]
    lvl = jnp.clip(s["base_level"], 1, MAX_BASE_LEVEL)
    job = jnp.clip(s["job"], 0, NUM_JOBS - 1)
    st, ag, vi, it, dx, lk = (s[k] for k in _STAT_KEYS)

    max_hp = JOB_BASE_HP_TABLE_J[job, lvl] * (100 + vi) // 100
    max_sp = JOB_BASE_SP_TABLE_J[job, lvl] * (100 + it) // 100
    atk = st + (st // 10) ** 2 + dx // 5 + lk // 5
    matk_min = it + (it // 7) ** 2
    matk_max = it + (it // 5) ** 2
    matk = (matk_min + matk_max) // 2
    hit = lvl + dx
    flee = lvl + ag
    soft_def = vi
    mdef = it + vi // 2
    crit = 10 + lk * 3
    cast_mult = jnp.clip(1000 - dx * 1000 // 150, 0, 1000)
    # ASPD by job and worn weapon class (rAthena job_aspd); a class the job cannot use falls back to the fist column.
    if "items" in world:
        equip = world["items"]["equip"][:, item_data.EQ_WEAPON]
        inv = world["players"]["inv"]
        item = jnp.take_along_axis(inv, jnp.clip(equip, 0, inv.shape[1] - 1)[:, None], axis=1)[:, 0]
        wc = jnp.where(equip >= 0, item_data.CLASS[item], item_data.WC_NONE)
    else:
        wc = jnp.zeros((MAX_PLAYERS,), _I32)
    base = JOB_WEAPON_ASPD_J[job, wc]
    base = jnp.where(base > 0, base, JOB_WEAPON_ASPD_J[job, 0])
    aspd = jnp.minimum(190, 200 - (200 - base) * (1000 - (4 * ag + dx)) // 1000)
    aspd_ticks = jnp.maximum(1, (20 * (200 - aspd) + TICK_MS // 2) // TICK_MS)  # attack delay = 2 x amotion
    out = {
        "max_hp": max_hp, "max_sp": max_sp, "atk": atk, "matk": matk, "hit": hit, "flee": flee,
        "def": soft_def, "mdef": mdef, "crit": crit, "cast_mult": cast_mult, "aspd_ticks": aspd_ticks,
        "aspd": aspd,
    }
    return {k: v.astype(_I32) for k, v in out.items()}


def hp_regen(world, sitting):
    """HP each living player regains THIS tick: 1 + max_hp // 200 + vit // 5 on the regen tick, else 0.

    Every HP_REGEN_TICKS ticks standing, every HP_REGEN_SIT_TICKS where `sitting` (bool [MAX_PLAYERS], derived by
    the caller from the anim action). Survival writes hp.
    """
    p, s = world["players"], world["stats"]
    alive = p["active"] & ~p["dead"]
    sit = jnp.asarray(sitting).astype(jnp.bool_)
    tick = world["tick"]
    on = jnp.where(sit, tick % HP_REGEN_SIT_TICKS == 0, tick % HP_REGEN_TICKS == 0)
    amount = 1 + derived(world)["max_hp"] // 200 + s["vit"] // 5
    return jnp.where(alive & on, amount, 0).astype(_I32)


def _level_loop(level, exp, active, need_fn, max_level):
    """Fixed-trip level-up over whole arrays. Returns (level, exp).

    Each iteration: a row with exp >= the requirement of its current level (and below the cap) pays it and rises.
    """
    def body(_, c):
        lv, ex = c
        need = need_fn(lv)
        up = active & (need > 0) & (ex >= need) & (lv < max_level)
        return lv + up.astype(_I32), ex - jnp.where(up, need, 0)

    return lax.fori_loop(0, MAX_LEVELS_PER_TICK, body, (level, exp))


def phase(world, inputs):
    """Kill exp, level-ups, allot, job change, death penalty, SP. Runs after loot, before survival."""
    p, m, s = world["players"], world["monsters"], world["stats"]
    act = p["active"]
    alive = act & ~p["dead"]
    slot = jnp.arange(MAX_PLAYERS, dtype=_I32)
    allot_in = inputs.get("allot", jnp.full((MAX_PLAYERS,), -1, _I32))
    job_in = inputs.get("job", jnp.full((MAX_PLAYERS,), -1, _I32))

    # --- 1. Kill credit and exp -------------------------------------------------------------------------------
    # The killer of each monster that died this tick is the player with the most cumulative damage on it
    # (ro_mvp.top_damager, fed by combat and by skill hits through ro_mvp.record_damage), as in RO: melee, ranged
    # and skill kills all pay. The gain is then shared through ro_party.share_exp (even split among living party
    # members in the same zone within the level range, +10% per extra member); solo players keep their gain.
    # A world without the mvp block (legacy tests) pays nobody. `kills_seen` stays a ledger of players.kills.
    if "mvp" in world:
        killer = ro_mvp.credit_of(world)
    else:
        killer = jnp.full((MAX_MONSTERS,), -1, _I32)
    paid = m["died"] & (killer >= 0)
    kind = jnp.clip(m["kind"], 0, MONSTER_BASE_EXP_J.shape[0] - 1)
    dest = jnp.where(paid, killer, MAX_PLAYERS)  # unpaid monsters scatter out of bounds and are dropped
    base_gain = _zeros().at[dest].add(jnp.where(paid, MONSTER_BASE_EXP_J[kind], 0), mode="drop")
    job_gain = _zeros().at[dest].add(jnp.where(paid, MONSTER_JOB_EXP_J[kind], 0), mode="drop")
    if "party" in world:
        base_gain = ro_party.share_exp(world, base_gain)
        job_gain = ro_party.share_exp(world, job_gain)
    base_exp = s["base_exp"] + jnp.where(act, base_gain, 0)
    job_exp = s["job_exp"] + jnp.where(act, job_gain, 0)

    # --- 2. Base level-up (up to MAX_LEVELS_PER_TICK levels; leftover exp carries to later ticks) --------------
    blv0 = s["base_level"]
    blv, base_exp = _level_loop(
        blv0, base_exp, act, lambda lv: BASE_EXP_TABLE_J[jnp.clip(lv, 0, MAX_BASE_LEVEL)], MAX_BASE_LEVEL)
    # Stat points for each level reached = sum of STAT_POINTS_PER_LEVEL over (blv0, blv], from a prefix table so no
    # per-level loop is needed: cum[L] = sum of the table up to index L.
    cum = jnp.cumsum(STAT_POINTS_PER_LEVEL_J)
    stat_points = s["stat_points"] + jnp.where(act, cum[jnp.clip(blv, 0, MAX_BASE_LEVEL)]
                                               - cum[jnp.clip(blv0, 0, MAX_BASE_LEVEL)], 0)
    base_exp = jnp.where(act & (blv >= MAX_BASE_LEVEL), 0, base_exp)  # at the cap exp is clamped to 0

    # --- 3. Job level-up -------------------------------------------------------------------------------------
    job = jnp.clip(s["job"], 0, NUM_JOBS - 1)
    jlv0 = s["job_level"]
    max_jlv = MAX_JOB_LEVEL_J[job]
    jlv, job_exp = _level_loop(
        jlv0, job_exp, act,
        lambda lv: JOB_EXP_TABLE_J[job, jnp.clip(lv, 0, JOB_EXP_TABLE_J.shape[1] - 1)], max_jlv)
    skill_points = s["skill_points"] + jnp.where(act, jlv - jlv0, 0) * SKILL_POINTS_PER_JOB_LEVEL
    job_exp = jnp.where(act & (jlv >= max_jlv), 0, job_exp)

    # --- 4. Allot: raise one stat by one, paying the stat's current-value cost -------------------------------
    stack = jnp.stack([s[k] for k in _STAT_KEYS])  # [6, P]
    ai = jnp.clip(allot_in, 0, NUM_STATS - 1)
    value = stack[ai, slot]
    cost = STAT_COST_J[jnp.clip(value, 0, MAX_STAT)]
    ok_allot = alive & (allot_in >= 0) & (allot_in < NUM_STATS) & (value < MAX_STAT) & (stat_points >= cost)
    stack = stack.at[jnp.where(ok_allot, ai, NUM_STATS), slot].add(1, mode="drop")
    stat_points = stat_points - jnp.where(ok_allot, cost, 0)

    # --- 5. Job change (Novice at max Novice job level only; skill points are kept) --------------------------
    ok_job = alive & (job_in >= 1) & (job_in < NUM_JOBS) & (job == JOB_NOVICE) & (jlv >= MAX_JOB_LEVEL_NOVICE)
    job = jnp.where(ok_job, job_in, job)
    jlv = jnp.where(ok_job, 1, jlv)
    job_exp = jnp.where(ok_job, 0, job_exp)

    # --- 6. Death penalty (a death since last tick; 1% of the base AND job exp requirement, Novices exempt) ----
    died_now = act & (p["deaths"] - s["deaths_seen"] > 0) & (job != JOB_NOVICE)
    loss_b = BASE_EXP_TABLE_J[jnp.clip(blv, 0, MAX_BASE_LEVEL)] * DEATH_PENALTY_PERMILLE // 1000
    loss_j = JOB_EXP_TABLE_J[job, jnp.clip(jlv, 0, JOB_EXP_TABLE_J.shape[1] - 1)] * DEATH_PENALTY_PERMILLE // 1000
    base_exp = jnp.where(died_now, jnp.maximum(0, base_exp - loss_b), base_exp)
    job_exp = jnp.where(died_now, jnp.maximum(0, job_exp - loss_j), job_exp)

    # --- 7. SP: recompute max_sp from the new stats, regen on schedule, refill on respawn --------------------
    new_s = {**s, **{k: stack[i] for i, k in enumerate(_STAT_KEYS)}, "base_level": blv, "job": job}
    max_sp = derived({**world, "stats": new_s})["max_sp"]
    regen_tick = (world["tick"] % SP_REGEN_TICKS) == 0
    regen = jnp.where(alive & regen_tick, 1 + max_sp // 100 + new_s["int"] // 6, 0)
    respawned = s["was_dead"] & ~p["dead"]
    sp = jnp.clip(jnp.where(respawned, max_sp, s["sp"] + regen), 0, max_sp)

    # --- 8. Write back (inactive rows keep their old values) and update the ledgers -------------------------
    def w(new, old):
        return jnp.where(act, new, old).astype(old.dtype)

    out = {k: w(stack[i], s[k]) for i, k in enumerate(_STAT_KEYS)}
    out.update(
        base_level=w(blv, s["base_level"]), job_level=w(jlv, s["job_level"]),
        base_exp=w(base_exp, s["base_exp"]), job_exp=w(job_exp, s["job_exp"]), job=w(job, s["job"]),
        stat_points=w(stat_points, s["stat_points"]), skill_points=w(skill_points, s["skill_points"]),
        sp=w(sp, s["sp"]), max_sp=w(max_sp, s["max_sp"]),
        kills_seen=w(p["kills"], s["kills_seen"]), deaths_seen=w(p["deaths"], s["deaths_seen"]),
        was_dead=w(p["dead"], s["was_dead"]),
    )
    return {**world, "stats": out}
