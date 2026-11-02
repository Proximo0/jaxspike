"""RO skills phase: learn, cooldowns and buffs, cast state machine, resolve hook, SP regen.

Pure JAX, fixed shapes, masks over [MAX_PLAYERS]. Contract: docs/m2_skills_brief.md. Effects live in
game/ro_skills_effects.py (resolve). Every output dtype and shape equals the input's (the world is scanned).
"""
import jax.numpy as jnp

from game import ro_skills_effects
from game.config import MAX_MONSTERS, MAX_PLAYERS, PLAYER_ATTACK, PLAYER_ATTACK_COOLDOWN, PLAYER_MAX_HP
from game.data import skills as S

I32 = jnp.int32
_DERIVED_KEYS = ("atk", "matk", "hit", "flee", "def", "mdef", "aspd_ticks", "cast_mult", "max_hp", "max_sp", "crit")
_MOD_KEYS = (
    "str_flat", "agi_flat", "vit_flat", "int_flat", "dex_flat", "luk_flat", "atk_flat", "atk_pct", "matk_pct",
    "hit_flat", "flee_flat", "def_pct", "mdef_flat", "aspd_pct", "crit_flat", "hp_regen_pct",
)


def make_state():
    mp = MAX_PLAYERS
    z = lambda *shape: jnp.zeros(shape, I32)
    return {
        "learned": z(mp, S.N_SKILLS),
        "cooldown": z(mp, S.N_SKILLS),
        "cast_left": z(mp),
        "casting": z(mp),
        "cast_target": jnp.full((mp,), -1, I32),
        "cast_tile": z(mp, 2),
        "buffs": z(mp, S.N_BUFFS),
        "buff_level": z(mp, S.N_BUFFS),
        "sp": jnp.full((mp,), S.SP_START, I32),
        "casts": z(mp),
        "skill_dmg": z(mp),
    }


def active_buffs(world):
    return world["skills"]["buffs"]


def fallback_derived(world):
    """Stopgap until ro_stats merges (M1): fixed numbers from config, no stat scaling."""
    full = lambda v: jnp.full((MAX_PLAYERS,), v, I32)
    return {
        "atk": full(PLAYER_ATTACK), "matk": full(10), "hit": full(0), "flee": full(0), "def": full(0),
        "mdef": full(0), "aspd_ticks": full(PLAYER_ATTACK_COOLDOWN), "cast_mult": full(1000),
        "max_hp": full(PLAYER_MAX_HP), "max_sp": full(50), "crit": full(0),
    }


def buff_stat_mods(world):
    """Stat modifiers from active buffs and learned passives; int32 [MAX_PLAYERS] per key."""
    sk = world["skills"]
    lv = lambda b: jnp.where(sk["buffs"][:, b] > 0, sk["buff_level"][:, b], 0).astype(I32)
    pa = lambda s: sk["learned"][:, s].astype(I32)
    zero = jnp.zeros((MAX_PLAYERS,), I32)
    m = {k: zero for k in _MOD_KEYS}
    bless, agi = lv(S.BUFF_BLESSING), lv(S.BUFF_INC_AGI)
    m["str_flat"] = bless
    m["int_flat"] = bless
    m["dex_flat"] = bless + pa(S.SK_OWLS_EYE)
    m["agi_flat"] = jnp.where(agi > 0, 2 + agi, 0).astype(I32)
    m["aspd_pct"] = agi
    m["def_pct"] = 5 * lv(S.BUFF_ANGELUS)
    m["mdef_flat"] = lv(S.BUFF_ENDURE)
    # Hiding / Play Dead: no stat effect in v1. Double Attack: needs a hit-count hook in combat (open defect).
    m["atk_flat"] = 4 * pa(S.SK_SWORD_MASTERY)
    m["hit_flat"] = pa(S.SK_VULTURES_EYE)
    m["hp_regen_pct"] = 10 * pa(S.SK_HP_RECOVERY)
    return m


def _default_derived(world):
    try:
        from game import ro_stats
    except ImportError:
        return fallback_derived(world)
    return ro_stats.derived(world)


def _sp_pool(world):
    """(block_name, key) of the authoritative SP pool: stats.sp when the stats block exists, else skills.sp."""
    if "stats" in world and "sp" in world["stats"]:
        return "stats", "sp"
    return "skills", "sp"


def phase(world, inputs, derived=None):
    if derived is None:
        derived = _default_derived(world)
    mp, mm = MAX_PLAYERS, MAX_MONSTERS
    pl, mo, sk = world["players"], world["monsters"], world["skills"]
    pidx = jnp.arange(mp, dtype=I32)
    alive = pl["active"] & ~pl["dead"]
    has_stats = "stats" in world
    stats = world["stats"] if has_stats else {}
    job = stats["job"] if "job" in stats else jnp.zeros((mp,), I32)
    has_sp = "skill_points" in stats

    # 1. learn
    ls = inputs["learn"].astype(I32)
    lsc = jnp.clip(ls, 0, S.N_SKILLS - 1)
    cur = sk["learned"][pidx, lsc]
    ok = (
        pl["active"] & (ls > 0) & (ls < S.N_SKILLS) & (cur < S.SKILL_MAX_LEVEL[lsc])
        & ((S.SKILL_JOB[lsc] == S.JOB_NOVICE) | (S.SKILL_JOB[lsc] == job))
    )
    if has_sp:
        ok = ok & (stats["skill_points"] > 0)
    learned = sk["learned"].at[pidx, lsc].add(ok.astype(I32))
    if has_sp:
        world = {**world, "stats": {**stats, "skill_points": stats["skill_points"] - ok.astype(I32)}}

    # 2. timers; dead players lose buffs and casts
    cooldown = jnp.maximum(sk["cooldown"] - 1, 0)
    buffs = jnp.maximum(sk["buffs"] - 1, 0)
    buffs = jnp.where(pl["dead"][:, None], 0, buffs)
    buff_level = jnp.where(buffs > 0, sk["buff_level"], 0)
    casting = jnp.where(pl["dead"], 0, sk["casting"])
    cast_left = jnp.where(pl["dead"], 0, sk["cast_left"])
    cast_target, cast_tile = sk["cast_target"], sk["cast_tile"]
    pool_blk, pool_key = _sp_pool(world)
    sp = world[pool_blk][pool_key]

    # 3. start casts
    s = inputs["skill"].astype(I32)
    sc = jnp.clip(s, 0, S.N_SKILLS - 1)
    t = inputs["target"].astype(I32)
    lvl = jnp.clip(learned[pidx, sc], 0, S.MAX_SKILL_LEVEL)
    ttype = S.SKILL_TARGET[sc]
    is_self = ttype == S.TARGET_SELF
    is_pl = (t >= 0) & (t < mp)
    is_mo = (t >= mp) & (t < mp + mm)
    tpl, tmo = jnp.clip(t, 0, mp - 1), jnp.clip(t - mp, 0, mm - 1)
    ally_ok = is_pl & pl["active"][tpl] & ~pl["dead"][tpl]
    enemy_ok = is_mo & mo["active"][tmo]
    tgt_ok = jnp.where(is_self, True, jnp.where(ttype == S.TARGET_ALLY, ally_ok, enemy_ok))
    own = pl["pos"]
    tpos = jnp.where(is_self[:, None], own, jnp.where(is_pl[:, None], pl["pos"][tpl], mo["pos"][tmo]))
    dist = jnp.sum(jnp.abs(tpos - own), axis=1)
    cost = S.SKILL_SP[sc, lvl]
    start = (
        alive & (casting == 0) & (s > 0) & (s < S.N_SKILLS) & (lvl > 0)
        & (S.SKILL_EFFECT[sc] != S.EFFECT_PASSIVE) & (cooldown[pidx, sc] == 0) & (sp >= cost)
        & tgt_ok & (dist <= S.SKILL_RANGE[sc, lvl])
    )
    sp = jnp.where(start, sp - cost, sp)
    cooldown = cooldown.at[pidx, sc].set(jnp.where(start, S.SKILL_COOLDOWN[sc, lvl], cooldown[pidx, sc]))
    casting = jnp.where(start, s, casting)
    cast_target = jnp.where(start, jnp.where(is_self, pidx, t), cast_target)
    cast_tile = jnp.where(start[:, None], tpos, cast_tile)
    cm = derived["cast_mult"].astype(I32)
    cast_left = jnp.where(start, S.SKILL_CAST[sc, lvl] * cm // 1000, cast_left)

    # 4. resolve
    sk = {
        **sk, "learned": learned, "cooldown": cooldown, "buffs": buffs, "buff_level": buff_level,
        "casting": casting, "cast_left": cast_left, "cast_target": cast_target, "cast_tile": cast_tile,
    }
    if pool_blk == "skills":
        sk["sp"] = sp
    else:
        world = {**world, "stats": {**world["stats"], "sp": sp.astype(world["stats"]["sp"].dtype)}}
    resolve = alive & (casting != 0) & (cast_left == 0)
    world = ro_skills_effects.resolve({**world, "skills": sk}, resolve, derived)
    sk = world["skills"]
    casting = jnp.where(resolve, 0, casting)
    casts = sk["casts"] + resolve.astype(I32)

    # 5. count down unresolved casts
    cast_left = jnp.where(casting != 0, jnp.maximum(cast_left - 1, 0), 0)

    # 6. SP cap (+ regen only for the skills.sp fallback pool; ro_stats.phase regens stats.sp)
    max_sp = derived["max_sp"].astype(I32)
    sp = world[pool_blk][pool_key].astype(I32)
    if pool_blk == "skills":
        regen = alive & (world["tick"] % S.SP_REGEN_TICKS == 0)
        sp = jnp.where(regen, sp + S.SP_REGEN_FLAT + max_sp // 100, sp)
    sp = jnp.minimum(sp, max_sp).astype(I32)

    sk = {**sk, "casting": casting.astype(I32), "cast_left": cast_left.astype(I32), "casts": casts.astype(I32)}
    if pool_blk == "skills":
        sk["sp"] = sp
        return {**world, "skills": sk}
    return {**world, "skills": sk, "stats": {**world["stats"], "sp": sp.astype(world["stats"]["sp"].dtype)}}
