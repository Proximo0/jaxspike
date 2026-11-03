"""Combat phase: respawns, player and monster auto-attacks, deaths. Pure JAX, fixed shapes, randomness via next_key.

RO-flavoured auto-attacks (all integer math, percent-style):

Player -> monster (one swing per `derived["aspd_ticks"]` ticks, a swing costs the cooldown even when it misses)
  atk      = derived atk + equipped weapon_atk + buff atk_flat, then * (100 + atk_pct) / 100
  roll     = atk * U[90, 110] / 100
  hit%     = clip(80 + (hit + hit_flat) - (150 + 3 * monster_level), 5, 95)      (a crit always hits)
  crit     = random per-mille < derived crit (+ crit_flat); a crit doubles the damage and ignores monster DEF
  damage   = roll * elem_mult(weapon_elem, monster_element) / 100 * (100 - DEF[kind]) / 100, min 1
             (0 only when the element table says immune)
  cooldown = aspd_ticks (reduced by the buff aspd_pct), so one attack every aspd_ticks ticks.
Monster -> player (cooldown MONSTER_ATTACK_COOLDOWN; a swing costs the cooldown even when it misses)
  hit%     = clip(80 + (100 + 2 * monster_level) - flee, 5, 95), flee = derived flee + flee_bonus
  damage   = U[ATK_MIN, ATK_MAX] * (100 - def) / 100, min 1, def = (derived def + armor_def) * (100 + def_pct) / 100
             clipped to [0, 80]
Only aggressive kinds (AGGRO) or monsters that were already hit (hp below the kind's max hp) attack: a passive
monster fights back but never starts a fight.
Respawn hp is `derived["max_hp"]` (survival / ro_items still cap hp at config.PLAYER_MAX_HP).
Every landed player hit is reported with ro_mvp.record_damage(world, attacker_idx, monster_idx, dmg).
"""
import jax
import jax.numpy as jnp

from game import ro_anim, ro_items, ro_mvp, ro_skills, ro_stats
from game.config import MAP, MONSTER_ATTACK_COOLDOWN, MONSTER_HP, MONSTER_RESPAWN_TICKS, MAX_PLAYERS
from game.data import monsters as mdata
from game.data.elements import MONSTER_ELEMENT, elem_mult
from game.spatial import build_grid, nearest_in_block
from game.state import next_key

_MON_HP = jnp.asarray(MONSTER_HP, jnp.int32)
_MAX_DEF = 80  # percent cap on player and monster DEF reduction


def hit_percent_player(hit, mon_level):
    """Chance in percent that a player's swing lands on a monster of this level."""
    return jnp.clip(80 + hit - (150 + 3 * mon_level), 5, 95)


def hit_percent_monster(mon_level, flee):
    """Chance in percent that a monster's swing lands on a player with this flee."""
    return jnp.clip(80 + (100 + 2 * mon_level) - flee, 5, 95)


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


def combat(world, inputs):
    """Run one combat tick and return the updated world."""
    d = ro_stats.derived(world)
    eq = ro_items.equipped(world)
    mod = ro_skills.buff_stat_mods(world)
    k_p, world = next_key(world)
    r = jax.random.uniform(k_p, (5, max(world["players"]["hp"].shape[0], world["monsters"]["hp"].shape[0])))

    p = _respawn_players(world["players"], world["monsters"], d["max_hp"])
    m = world["monsters"]
    n_p = p["hp"].shape[0]
    n_m = m["hp"].shape[0]
    alive = p["active"] & ~p["dead"]
    pidx = jnp.arange(n_p, dtype=jnp.int32)
    kind = m["kind"]

    # Player auto-attacks on the nearest adjacent monster.
    mtable, _ = build_grid(m["pos"], m["active"])
    tgt, dist = nearest_in_block(mtable, p["pos"], m["pos"])
    swing = alive & inputs["attack"] & (p["attack_cd"] == 0) & (dist <= ro_items.weapon_range(world)) & m["active"][tgt]
    tkind = kind[tgt]
    atk = d["atk"] + eq["weapon_atk"] + mod["atk_flat"]
    atk = atk * (100 + mod["atk_pct"]) // 100
    base = atk * _roll(r[0, :n_p], 90, 110) // 100
    crit = _roll(r[1, :n_p], 0, 999) < (d["crit"] + mod["crit_flat"])
    hit_roll = _roll(r[2, :n_p], 0, 99) < hit_percent_player(d["hit"] + mod["hit_flat"], mdata.LEVEL[tkind])
    mdef = jnp.clip(mdata.DEF[tkind], 0, _MAX_DEF)
    elem = elem_mult(eq["weapon_elem"], MONSTER_ELEMENT[tkind])
    dmg_hit = base * elem // 100
    dmg_hit = jnp.where(crit, dmg_hit * 2, dmg_hit * (100 - mdef) // 100)
    dmg_hit = jnp.where(elem > 0, jnp.maximum(dmg_hit, 1), 0)
    dmg_p_out = jnp.where(swing & (hit_roll | crit), dmg_hit, 0)  # damage each player deals this tick
    landed = dmg_p_out > 0
    tgt_w = jnp.where(landed, tgt, n_m)  # out of bounds = dropped
    dmg_m = jnp.zeros_like(m["hp"]).at[tgt_w].add(dmg_p_out, mode="drop")
    new_m_hp = m["hp"] - dmg_m
    killed = m["active"] & (m["hp"] > 0) & (new_m_hp <= 0)
    # Lowest-index hitter of each monster gets the kill credit.
    first = jnp.full(m["hp"].shape, n_p, jnp.int32).at[tgt_w].min(pidx, mode="drop")
    credit = jnp.where(killed, first, n_p)
    kills = p["kills"].at[credit].add(1, mode="drop")
    aspd = jnp.maximum(1, d["aspd_ticks"] * 100 // (100 + mod["aspd_pct"]))
    p_cd = jnp.where(swing, aspd - 1, jnp.maximum(p["attack_cd"] - 1, 0))  # period == aspd ticks

    # Monster attacks on the nearest adjacent living player outside the protection zone (start-of-phase hp).
    targetable = alive & ~world["pz"][p["pos"][:, 0], p["pos"][:, 1]]
    ptable, _ = build_grid(p["pos"], targetable)
    ptgt, pdist = nearest_in_block(ptable, m["pos"], p["pos"])
    provoked = mdata.AGGRO[kind] | (m["hp"] < _MON_HP[kind])  # passive kinds only fight back
    mswing = m["active"] & (m["hp"] > 0) & (m["attack_cd"] == 0) & (pdist <= 1) & alive[ptgt] & provoked
    mswing = mswing & ~world["pz"][p["pos"][ptgt, 0], p["pos"][ptgt, 1]]  # protection zone
    flee = d["flee"][ptgt] + mod["flee_flat"][ptgt]
    pdef = (d["def"] + eq["armor_def"]) * (100 + mod["def_pct"]) // 100
    pdef = jnp.clip(pdef[ptgt], 0, _MAX_DEF)
    m_hit = _roll(r[3, :n_m], 0, 99) < hit_percent_monster(mdata.LEVEL[kind], flee)
    raw = _roll(r[4, :n_m], mdata.ATK_MIN[kind], mdata.ATK_MAX[kind])
    mdmg = jnp.maximum(raw * (100 - pdef) // 100, 1)
    ptgt_w = jnp.where(mswing & m_hit, ptgt, MAX_PLAYERS)
    dmg_p = jnp.zeros_like(p["hp"]).at[ptgt_w].add(mdmg, mode="drop")
    new_p_hp = jnp.maximum(p["hp"] - dmg_p, 0)
    m_cd = jnp.where(mswing, MONSTER_ATTACK_COOLDOWN, jnp.maximum(m["attack_cd"] - 1, 0))

    # Deaths.
    mdead = m["active"] & (new_m_hp <= 0)
    pdead = p["active"] & ~p["dead"] & (new_p_hp <= 0)

    players = {
        **p,
        "hp": new_p_hp,
        "dead": p["dead"] | pdead,
        "attack_cd": p_cd,
        "kills": kills,
        "deaths": p["deaths"] + pdead.astype(jnp.int32),
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
    world = ro_anim.on_combat(world, swing, tgt, mswing, ptgt, dmg_m, dmg_p)  # ro-anim: attack, hurt, hit numbers
    return ro_mvp.record_damage(world, pidx, tgt, dmg_p_out)
