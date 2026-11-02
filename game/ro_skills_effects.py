"""Skill effect resolution (M2, worker C). Pure JAX, fixed shapes, no randomness.

`resolve(world, mask, derived)` applies the effects of every caster in `mask` (bool [MAX_PLAYERS]) whose cast
resolves this tick: monster damage (single target and AoE through one [MAX_PLAYERS, 9] gathered-index path), deaths
with lowest-slot kill credit (mirrors game/combat.py), heals, and buffs. Targets are global entity ids: players
0..MAX_PLAYERS-1, monsters MAX_PLAYERS + i.
"""
import jax.numpy as jnp

from game.config import MAP, MAX_MONSTERS, MAX_PLAYERS, MONSTER_RESPAWN_TICKS
from game.data.elements import MONSTER_ELEMENT, elem_mult
from game.data.skills import (
    EFFECT_ATK,
    EFFECT_BUFF,
    EFFECT_HEAL,
    EFFECT_MATK,
    SKILL_AOE,
    SKILL_BUFF,
    SKILL_BUFF_TICKS,
    SKILL_EFFECT,
    SKILL_ELEMENT,
    SKILL_FLAT_HEAL,
    SKILL_HITS,
    SKILL_POWER,
)

R_MAX = 1
_OFFSETS = jnp.array(
    [[dx, dy] for dx in range(-R_MAX, R_MAX + 1) for dy in range(-R_MAX, R_MAX + 1)], jnp.int32
)  # [9, 2], offset 4 is (0, 0)
N_OFF = _OFFSETS.shape[0]
_CENTER = N_OFF // 2


def base_damage(derived, effect):
    """Per-caster base damage: atk for EFFECT_ATK, matk for EFFECT_MATK, else 0. int32 [MP]."""
    return jnp.where(
        effect == EFFECT_ATK, derived["atk"], jnp.where(effect == EFFECT_MATK, derived["matk"], 0)
    ).astype(jnp.int32)


def skill_damage(base, power, hits, attack_elem, defend_elem):
    """Damage of one skill use: base * power // 1000 * hits * elem_mult // 100, minimum 0. Broadcastable."""
    d = base * power // 1000 * hits * elem_mult(attack_elem, defend_elem) // 100
    return jnp.maximum(d, 0).astype(jnp.int32)


def heal_amount(matk, power, flat):
    """Heal for one cast: (10 + matk) * power // 1000 + flat."""
    return ((10 + matk) * power // 1000 + flat).astype(jnp.int32)


def monster_tile_map(m):
    """[MAP, MAP] int32 index of the lowest-slot active monster on each tile, MAX_MONSTERS if none."""
    midx = jnp.arange(MAX_MONSTERS, dtype=jnp.int32)
    return (
        jnp.full((MAP, MAP), MAX_MONSTERS, jnp.int32)
        .at[m["pos"][:, 0], m["pos"][:, 1]]
        .min(jnp.where(m["active"], midx, MAX_MONSTERS))
    )


def hit_slots(m, occ, hits_mons, target_mon, tile, aoe):
    """Monster index hit by each caster in each of the 9 offset slots -> int32 [MP, 9].

    Entries equal to MAX_MONSTERS mean "no hit" (out of range index, dropped by the scatters).
    Single target (aoe == 0): slot 4 (centre) holds the targeted monster if it is active, the rest are empty.
    AoE: gathered from `occ` around `tile`, offsets with Chebyshev length > aoe or off the map are empty.
    Casters not in `hits_mons` have all slots empty.
    """
    pos = tile[:, None, :] + _OFFSETS[None, :, :]  # [MP, 9, 2]
    inb = jnp.all((pos >= 0) & (pos < MAP), axis=-1)
    gathered = occ[jnp.clip(pos[..., 0], 0, MAP - 1), jnp.clip(pos[..., 1], 0, MAP - 1)]
    within = jnp.max(jnp.abs(_OFFSETS), axis=-1)[None, :] <= aoe[:, None]
    aoe_idx = jnp.where(inb & within, gathered, MAX_MONSTERS)
    tm = jnp.clip(target_mon, 0, MAX_MONSTERS - 1)
    tvalid = (target_mon >= 0) & (target_mon < MAX_MONSTERS) & m["active"][tm]
    single = jnp.full((MAX_PLAYERS, N_OFF), MAX_MONSTERS, jnp.int32)
    single = single.at[:, _CENTER].set(jnp.where(tvalid, target_mon, MAX_MONSTERS))
    idx = jnp.where((aoe > 0)[:, None], aoe_idx, single)
    return jnp.where(hits_mons[:, None], idx, MAX_MONSTERS).astype(jnp.int32)


def damage_monsters(m, p_kills, idx, dmg):
    """Scatter [MP, 9] damage onto monsters, resolve deaths and credit. Returns (monsters, kills).

    `idx` [MP, 9] monster indices (MAX_MONSTERS = none), `dmg` [MP, 9] int32 (already zero where idx is none).
    """
    pidx = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
    flat = idx.reshape(-1)
    dmg_m = jnp.zeros_like(m["hp"]).at[flat].add(dmg.reshape(-1), mode="drop")
    new_hp = m["hp"] - dmg_m
    killed = m["active"] & (m["hp"] > 0) & (new_hp <= 0)
    first = (
        jnp.full(m["hp"].shape, MAX_PLAYERS, jnp.int32)
        .at[flat]
        .min(jnp.broadcast_to(pidx[:, None], idx.shape).reshape(-1), mode="drop")
    )
    credit = jnp.where(killed, first, MAX_PLAYERS)
    kills = p_kills.at[credit].add(1, mode="drop")
    monsters = {
        **m,
        "hp": new_hp,
        "active": m["active"] & ~killed,
        "respawn_in": jnp.where(killed, MONSTER_RESPAWN_TICKS, m["respawn_in"]),
        "died": m["died"] | killed,
    }
    return monsters, kills


def heal_players(p, target, do_heal, heal, max_hp):
    """Scatter-add heals onto living player targets, clamp those to max_hp. Returns hp [MP]."""
    t = jnp.clip(target, 0, MAX_PLAYERS - 1)
    ok = do_heal & (target >= 0) & (target < MAX_PLAYERS) & p["active"][t] & ~p["dead"][t]
    tw = jnp.where(ok, target, MAX_PLAYERS)
    added = jnp.zeros_like(p["hp"]).at[tw].add(jnp.where(ok, heal, 0), mode="drop")
    healed = jnp.zeros(p["hp"].shape, jnp.bool_).at[tw].max(ok, mode="drop")
    living = p["active"] & ~p["dead"]
    return jnp.where(healed & living, jnp.minimum(p["hp"] + added, jnp.maximum(max_hp, p["hp"])), p["hp"])


def apply_buffs(p, sk, target, do_buff, buff, ticks, lvl):
    """Apply buffs to living player targets: ticks by scatter-max, level set where applied."""
    t = jnp.clip(target, 0, MAX_PLAYERS - 1)
    ok = do_buff & (buff > 0) & (target >= 0) & (target < MAX_PLAYERS) & p["active"][t] & ~p["dead"][t]
    tw = jnp.where(ok, target, MAX_PLAYERS)
    buffs = sk["buffs"].at[tw, buff].max(jnp.where(ok, ticks, 0), mode="drop")
    applied = jnp.zeros(sk["buffs"].shape, jnp.bool_).at[tw, buff].max(ok, mode="drop")
    lvl_new = jnp.zeros_like(sk["buff_level"]).at[tw, buff].max(jnp.where(ok, lvl, 0), mode="drop")
    return buffs, jnp.where(applied, lvl_new, sk["buff_level"])


def resolve(world, mask, derived):
    """Apply the effects of all casters in `mask`. See the module docstring."""
    p, m, sk = world["players"], world["monsters"], world["skills"]
    pidx = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
    s = sk["casting"]
    lvl = jnp.take_along_axis(sk["learned"], s[:, None], axis=1)[:, 0]
    t = sk["cast_target"]
    tile = sk["cast_tile"]
    eff = SKILL_EFFECT[s]
    mask = mask & (s != 0)

    # Damage to monsters.
    is_dmg = mask & ((eff == EFFECT_ATK) | (eff == EFFECT_MATK))
    aoe = SKILL_AOE[s]
    occ = monster_tile_map(m)
    idx = hit_slots(m, occ, is_dmg, t - MAX_PLAYERS, tile, aoe)
    base = base_damage(derived, eff)
    d_elem = MONSTER_ELEMENT[m["kind"]][jnp.clip(idx, 0, MAX_MONSTERS - 1)]
    dmg = skill_damage(
        base[:, None], SKILL_POWER[s, lvl][:, None], SKILL_HITS[s][:, None], SKILL_ELEMENT[s][:, None], d_elem
    )
    dmg = jnp.where(idx < MAX_MONSTERS, dmg, 0)
    monsters, kills = damage_monsters(m, p["kills"], idx, dmg)

    # Heals.
    heal = heal_amount(derived["matk"], SKILL_POWER[s, lvl], SKILL_FLAT_HEAL[s])
    hp = heal_players(p, t, mask & (eff == EFFECT_HEAL), heal, derived["max_hp"])

    # Buffs.
    buffs, buff_level = apply_buffs(
        p, sk, t, mask & (eff == EFFECT_BUFF), SKILL_BUFF[s], SKILL_BUFF_TICKS[s, lvl], lvl
    )

    skills = {
        **sk,
        "buffs": buffs,
        "buff_level": buff_level,
        "skill_dmg": sk["skill_dmg"] + dmg.sum(axis=1).astype(jnp.int32),
    }
    players = {**p, "hp": hp.astype(p["hp"].dtype), "kills": kills}
    return {**world, "players": players, "monsters": monsters, "skills": skills}
