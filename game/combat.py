"""Combat phase: respawns, player and monster auto-attacks, deaths. Pure JAX, fixed shapes, randomness via next_key.

RO-flavoured auto-attacks (all integer math, percent-style):

Player -> monster (one swing per `derived["aspd_ticks"]` ticks, a swing costs the cooldown even when it misses)
  atk      = derived atk + equipped weapon_atk + buff atk_flat, then * (100 + atk_pct) / 100
  roll     = atk * U[90, 110] / 100
  hit%     = clip(80 + (hit + hit_flat) - (monster_level + monster_agi), 5, 95)  (a crit always hits)
  crit     = random per-mille < derived crit (+ crit_flat); a crit doubles the damage and ignores both DEFs
  damage   = roll * elem_mult(weapon_elem, monster_element) / 100, then (non-crit)
             * (100 - clip(DEF[kind], 0, 80)) / 100 (hard DEF, percent) - VIT[kind] (flat soft DEF), min 1
             (0 only when the element table says immune)
  cooldown = aspd_ticks (reduced by the buff aspd_pct), so one attack every aspd_ticks ticks.
Monster -> player (cooldown MONSTER_ATTACK_COOLDOWN; a swing costs the cooldown even when it misses)
  hit%     = clip(80 + (monster_level + monster_dex) - flee, 5, 95), flee = derived flee + flee_bonus
  damage   = U[ATK_MIN, ATK_MAX] * (100 - hard) / 100 - soft, min 1
             hard = clip(armor_def, 0, 80) percent; soft = derived def (= VIT) * (100 + def_pct) / 100 flat
Only aggressive kinds (AGGRO) or monsters that were already hit (hp below the kind's max hp) attack: a passive
monster fights back but never starts a fight.
Respawn hp is `derived["max_hp"]` (survival / ro_items still cap hp at config.PLAYER_MAX_HP).
Every landed player hit is reported with ro_mvp.record_damage(world, attacker_idx, monster_idx, dmg); the kill
credit (players.kills) is ro_mvp.credit_of, the same top cumulative damager that gets the exp.
Cards (ro4-pvp): worn card ATK/HIT/CRIT/FLEE/DEF bonuses add to atk, hit, crit (x10 per mille), flee and hard DEF.
A crit deals at least 2 and is flagged in the anim block's `crit` column (the wire sends it as a negative h).
Player -> player (ro4-pvp, `pvp_swings`): with world["pvp"], players on the arena floor (or in a PvP zone with
PK on) swing at the nearest allowed player in weapon range when no monster is in reach, with the player ->
monster formulas against the target's flee, armor element, hard and soft DEF. PvP deaths go to pvp.pvp_deaths
(not players.deaths: no exp penalty) and pvp.pvp_kills of the lowest-slot PvP hitter.
"""
import jax
import jax.numpy as jnp

from game import ro_anim, ro_items, ro_mvp, ro_pvp, ro_skills, ro_stats
from game.config import MAP, MONSTER_ATTACK_COOLDOWN, MONSTER_HP, MONSTER_RESPAWN_TICKS, MAX_PLAYERS, TICK_MS  # noqa: F401
from game.data import items as item_data
from game.data import monsters as mdata
from game.data.elements import MONSTER_ELEMENT, elem_mult
from game.ro_move import dir_toward
from game.spatial import build_grid, nearest_in_block
from game.state import next_key

_MON_HP = jnp.asarray(MONSTER_HP, jnp.int32)
_MAX_DEF = 80  # percent cap on player and monster DEF reduction


def hit_percent_player(hit, mon_level, mon_agi=0):
    """Chance in percent that a player's swing lands on a monster of this level and AGI (RO: hit - flee)."""
    return jnp.clip(80 + hit - (mon_level + mon_agi), 5, 95)


def hit_percent_monster(mon_level, flee, mon_dex=0):
    """Chance in percent that a monster's swing lands on a player with this flee (RO: monster hit - flee)."""
    return jnp.clip(80 + (mon_level + mon_dex) - flee, 5, 95)


def _roll(u, lo, hi):
    """Integer in [lo, hi] from a uniform float u in [0, 1)."""
    return lo + jnp.minimum(jnp.floor(u * (hi - lo + 1)).astype(jnp.int32), hi - lo)


def _respawn_players(p, m, max_hp):
    """Dead players return to spawn with full hp, once the spawn tile holds no monster or living player.

    Dead players are not physical anywhere in the engine (movement and monster pathing ignore
    them too), so a respawn may land on a tile where another dead player is still waiting.
    The invariant is "no two living players share a tile", not "no two active players".
    """
    occ = jnp.zeros((MAP, MAP), jnp.bool_)
    occ = occ.at[m["pos"][:, 0], m["pos"][:, 1]].max(m["active"])
    living = p["active"] & ~p["dead"]
    occ = occ.at[p["pos"][:, 0], p["pos"][:, 1]].max(living)
    r = p["active"] & p["dead"] & ~occ[p["spawn"][:, 0], p["spawn"][:, 1]]
    return {
        **p,
        "pos": jnp.where(r[:, None], p["spawn"], p["pos"]),
        "hp": jnp.where(r, max_hp, p["hp"]),
        "dead": p["dead"] & ~r,
        "attack_cd": jnp.where(r, 0, p["attack_cd"]),
    }


def _card_terms(eq):
    """Flat combat bonuses from worn cards (and item bonuses), int32 [MAX_PLAYERS] each: atk, hit, flee, def (hard,
    percent like armor DEF), crit (per mille: a card's +1 crit is +1 %). Stat bonuses (STR..LUK, max HP/SP) are not
    applied here (they would need ro_stats.derived)."""
    # Lead (round 4): every card and gear bonus is applied once, in ro_stats.derived (atk, hit, flee, def, crit, the six
    # stats, max HP/SP), so combat reads them through `derived` and adds nothing here. Kept as zeros so the three
    # swing paths keep their shape.
    z = jnp.zeros_like(eq["weapon_atk"])
    return {"atk": z, "hit": z, "flee": z, "def": z, "crit": z}


def pvp_swings(world, p, inputs, mswing_p, atk, hit, crit_chance, flee, hard, soft, elem_w, elem_a, key):
    """The player-vs-player swings of one tick (ro4-pvp), with the RO formulas of the player -> monster path.

    Candidates are the living players on the arena floor or in a PvP zone with PK on (game/ro_pvp.areas), compacted
    into a fixed [PVP_SLOTS] list (lowest slots first). A candidate with the attack input, a ready cooldown and no
    monster in reach (mswing_p) swings at the nearest (manhattan, lowest slot on ties) other candidate within its
    weapon range that ro_pvp.pair_ok allows. Per swing: roll = atk * U[90, 110] / 100; hit% = clip(80 + hit -
    target flee, 5, 95) (a crit always hits); crit = per-mille roll < crit chance; damage = roll * elem(weapon,
    target armor) / 100, then (non-crit) * (100 - target hard DEF) / 100 - target soft DEF, min 1; a crit doubles it,
    ignores both DEFs and is at least 2 (so -1 stays the wire's Miss marker); 0 when the element table says immune.
    Returns per player [MAX_PLAYERS]: swing (bool), target slot, damage taken, crit taken (bool), lowest-slot hitter
    (MAX_PLAYERS = none), swung_at (bool)."""
    n_p = MAX_PLAYERS
    arena, field = ro_pvp.areas({**world, "players": p})
    idx = jnp.nonzero(arena | field, size=ro_pvp.PVP_SLOTS, fill_value=n_p)[0].astype(jnp.int32)  # [K] ascending
    valid = idx < n_p
    ic = jnp.minimum(idx, n_p - 1)
    pos = p["pos"][ic]
    dist = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)  # [K, K]
    ar, fl = arena[ic], field[ic]
    ok = (valid[:, None] & valid[None, :] & (idx[:, None] != idx[None, :])
          & ro_pvp.pair_ok(ar[:, None], fl[:, None], ar[None, :], fl[None, :]))
    can = ok & (dist <= ro_items.weapon_range(world)[ic][:, None])
    col = jnp.argmin(jnp.where(can, dist, 1 << 30), axis=1)
    has = jnp.any(can, axis=1)
    t = ic[col]  # target slot per candidate attacker
    swing = valid & has & inputs["attack"][ic] & (p["attack_cd"][ic] == 0) & ~mswing_p[ic]

    r = jax.random.uniform(key, (3, ro_pvp.PVP_SLOTS))
    base = atk[ic] * _roll(r[0], 90, 110) // 100
    crit = _roll(r[1], 0, 999) < crit_chance[ic]
    hit_ok = _roll(r[2], 0, 99) < jnp.clip(80 + hit[ic] - flee[t], 5, 95)
    elem = elem_mult(elem_w[ic], elem_a[t])
    raw = base * elem // 100
    normal = raw * (100 - hard[t]) // 100 - soft[t]
    dmg = jnp.where(crit, jnp.maximum(raw * 2, 2), normal)
    dmg = jnp.where(elem > 0, jnp.maximum(dmg, 1), 0)
    landed = swing & (hit_ok | crit) & (dmg > 0)

    tw = jnp.where(landed, t, n_p)
    a_full = jnp.where(valid, idx, n_p)
    return {
        "swing": jnp.zeros((n_p,), jnp.bool_).at[a_full].set(swing, mode="drop"),
        "target": jnp.zeros((n_p,), jnp.int32).at[a_full].set(t, mode="drop"),
        "dmg": jnp.zeros((n_p,), jnp.int32).at[tw].add(jnp.where(landed, dmg, 0), mode="drop"),
        "crit": jnp.zeros((n_p,), jnp.bool_).at[tw].max(crit, mode="drop"),
        "first": jnp.full((n_p,), n_p, jnp.int32).at[tw].min(idx, mode="drop"),
        "swung_at": jnp.zeros((n_p,), jnp.bool_).at[jnp.where(swing, t, n_p)].max(True, mode="drop"),
    }


def _no_pvp(n_p):
    z = jnp.zeros((n_p,), jnp.bool_)
    return {"swing": z, "target": jnp.zeros((n_p,), jnp.int32), "dmg": jnp.zeros((n_p,), jnp.int32), "crit": z,
            "first": jnp.full((n_p,), n_p, jnp.int32), "swung_at": z}


def combat(world, inputs):
    """Run one combat tick and return the updated world."""
    d = ro_stats.derived(world)
    eq = ro_items.equipped(world)
    mod = ro_skills.buff_stat_mods(world)
    cards = _card_terms(eq)
    k_p, world = next_key(world)
    r = jax.random.uniform(k_p, (5, max(world["players"]["hp"].shape[0], world["monsters"]["hp"].shape[0])))

    p = _respawn_players(world["players"], world["monsters"], d["max_hp"])
    m = world["monsters"]
    n_p = p["hp"].shape[0]
    n_m = m["hp"].shape[0]
    alive = p["active"] & ~p["dead"]
    pidx = jnp.arange(n_p, dtype=jnp.int32)
    kind = m["kind"]

    # Per-player offence and defence: the same terms against monsters and players (cards included).
    atk = d["atk"] + eq["weapon_atk"] + mod["atk_flat"] + cards["atk"]
    atk = atk * (100 + mod["atk_pct"]) // 100
    hit = d["hit"] + mod["hit_flat"] + cards["hit"]
    crit_chance = d["crit"] + mod["crit_flat"] + cards["crit"]
    flee_all = d["flee"] + mod["flee_flat"] + cards["flee"]
    hard_all = jnp.clip(eq["armor_def"] + cards["def"], 0, _MAX_DEF)  # armor DEF is a percent
    soft_all = d["def"] * (100 + mod["def_pct"]) // 100  # derived def (VIT) is flat

    # Player auto-attacks on the nearest adjacent monster.
    mtable, _ = build_grid(m["pos"], m["active"])
    tgt, dist = nearest_in_block(mtable, p["pos"], m["pos"])
    swing = alive & inputs["attack"] & (p["attack_cd"] == 0) & (dist <= ro_items.weapon_range(world)) & m["active"][tgt]
    tkind = kind[tgt]
    base = atk * _roll(r[0, :n_p], 90, 110) // 100
    crit = _roll(r[1, :n_p], 0, 999) < crit_chance
    hit_roll = _roll(r[2, :n_p], 0, 99) < hit_percent_player(hit, mdata.LEVEL[tkind], mdata.AGI[tkind])
    mdef = jnp.clip(mdata.DEF[tkind], 0, _MAX_DEF)
    elem = elem_mult(eq["weapon_elem"], MONSTER_ELEMENT[tkind])
    dmg_hit = base * elem // 100
    soft = dmg_hit * (100 - mdef) // 100 - mdata.VIT[tkind]  # hard DEF percent, then flat VIT DEF
    dmg_hit = jnp.where(crit, jnp.maximum(dmg_hit * 2, 2), soft)  # a crit is at least 2 (-1 is the wire's Miss)
    dmg_hit = jnp.where(elem > 0, jnp.maximum(dmg_hit, 1), 0)
    dmg_p_out = jnp.where(swing & (hit_roll | crit), dmg_hit, 0)  # damage each player deals this tick
    landed = dmg_p_out > 0
    tgt_w = jnp.where(landed, tgt, n_m)  # out of bounds = dropped
    dmg_m = jnp.zeros_like(m["hp"]).at[tgt_w].add(dmg_p_out, mode="drop")
    crit_m = jnp.zeros((n_m,), jnp.bool_).at[tgt_w].max(crit, mode="drop")
    new_m_hp = m["hp"] - dmg_m
    killed = m["active"] & (m["hp"] > 0) & (new_m_hp <= 0)
    # Kill credit = the exp credit: the top cumulative damager (ro_mvp.credit_of) after this tick's hits are
    # recorded. Worlds without the mvp block fall back to the lowest-slot hitter of the tick.
    if "mvp" in world:
        world = ro_mvp.record_damage(world, pidx, tgt, dmg_p_out)
        killer = ro_mvp.credit_of({**world, "players": p})
        credit = jnp.where(killed & (killer >= 0), killer, n_p)
    else:
        first = jnp.full(m["hp"].shape, n_p, jnp.int32).at[tgt_w].min(pidx, mode="drop")
        credit = jnp.where(killed, first, n_p)
    kills = p["kills"].at[credit].add(1, mode="drop")

    # Player-vs-player swings (ro4-pvp): only with the pvp block; their own key, so the PvE rolls above are unchanged.
    pv = _no_pvp(n_p)
    if "pvp" in world:
        pv = pvp_swings(world, p, inputs, swing, atk, hit, crit_chance, flee_all, hard_all, soft_all,
                        eq["weapon_elem"], eq["armor_elem"], jax.random.fold_in(k_p, 1))
    aspd = jnp.maximum(1, d["aspd_ticks"] * 100 // (100 + mod["aspd_pct"]))
    p_cd = jnp.where(swing | pv["swing"], aspd - 1, jnp.maximum(p["attack_cd"] - 1, 0))  # period == aspd ticks

    # Monster attacks on the nearest adjacent living player outside the protection zone (start-of-phase hp).
    targetable = alive & ~world["pz"][p["pos"][:, 0], p["pos"][:, 1]]
    ptable, _ = build_grid(p["pos"], targetable)
    ptgt, pdist = nearest_in_block(ptable, m["pos"], p["pos"])
    provoked = mdata.AGGRO[kind] | (m["hp"] < _MON_HP[kind])  # passive kinds only fight back
    mswing = m["active"] & (m["hp"] > 0) & (m["attack_cd"] == 0) & (pdist <= 1) & alive[ptgt] & provoked
    mswing = mswing & ~world["pz"][p["pos"][ptgt, 0], p["pos"][ptgt, 1]]  # protection zone
    flee = flee_all[ptgt]
    hard = hard_all[ptgt]
    soft = soft_all[ptgt]
    m_hit = _roll(r[3, :n_m], 0, 99) < hit_percent_monster(mdata.LEVEL[kind], flee, mdata.DEX[kind])
    raw = _roll(r[4, :n_m], mdata.ATK_MIN[kind], mdata.ATK_MAX[kind])
    mdmg = jnp.maximum(raw * (100 - hard) // 100 - soft, 1)
    ptgt_w = jnp.where(mswing & m_hit, ptgt, MAX_PLAYERS)
    dmg_p = jnp.zeros_like(p["hp"]).at[ptgt_w].add(mdmg, mode="drop")
    dmg_taken = dmg_p + pv["dmg"]
    new_p_hp = jnp.maximum(p["hp"] - dmg_taken, 0)
    # ro4-balance: RO attack delay per kind (data.monsters.ADELAY_MS) instead of a flat MONSTER_ATTACK_COOLDOWN
    m_period = jnp.maximum(1, (jnp.asarray(mdata.ADELAY_MS)[kind] + TICK_MS // 2) // TICK_MS)
    m_cd = jnp.where(mswing, m_period, jnp.maximum(m["attack_cd"] - 1, 0))

    # Deaths. A death with PvP damage this tick is a PvP death: counted in pvp.pvp_deaths, not players.deaths, so the
    # ro_stats death penalty (keyed on players.deaths) never fires for it; the lowest-slot PvP hitter gets the kill.
    mdead = m["active"] & (new_m_hp <= 0)
    pdead = p["active"] & ~p["dead"] & (new_p_hp <= 0)
    pvp_death = pdead & (pv["dmg"] > 0)

    players = {
        **p,
        "hp": new_p_hp,
        "dead": p["dead"] | pdead,
        "attack_cd": p_cd,
        "kills": kills,
        "deaths": p["deaths"] + (pdead & ~pvp_death).astype(jnp.int32),
    }
    monsters = {
        **m,
        "hp": new_m_hp,
        "active": m["active"] & ~mdead,
        "respawn_in": jnp.where(mdead, MONSTER_RESPAWN_TICKS, m["respawn_in"]),
        "died": m["died"] | mdead,
        "attack_cd": m_cd,
    }
    world = {**world, "players": players, "monsters": monsters}
    if "pvp" in world:
        v = world["pvp"]
        pk_kills = v["pvp_kills"].at[jnp.where(pvp_death, pv["first"], n_p)].add(1, mode="drop")
        world = {**world, "pvp": {**v, "pvp_kills": pk_kills, "pvp_deaths": v["pvp_deaths"] + pvp_death.astype(jnp.int32)}}
    world = ro_anim.on_combat(world, swing, tgt, mswing, ptgt, dmg_m, dmg_taken)  # ro-anim: attack, hurt, hit numbers
    if "anim" in world:  # ro4-pvp: PvP swings (attack facing the target player), PvP misses, the crit column
        no_m = jnp.zeros((n_m,), jnp.bool_)
        face = jnp.concatenate([dir_toward(p["pos"][pv["target"]] - p["pos"]), jnp.full((n_m,), -1, jnp.int32)])
        world = ro_anim.set_action(world, jnp.concatenate([pv["swing"], no_m]), ro_anim.ATTACK,
                                   ro_anim.amotion_ms(world, d), face)
        world = ro_anim.note_miss(world, jnp.concatenate([pv["swung_at"] & (dmg_taken <= 0), no_m]))
        if "crit" in world["anim"]:
            world = {**world, "anim": {**world["anim"], "crit": jnp.concatenate([pv["crit"], crit_m])}}
    return world
