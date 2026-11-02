"""Combat phase: respawns, player and monster attacks, deaths. Pure JAX."""
import jax.numpy as jnp

from game.config import (
    MAP,
    MONSTER_ATTACK_COOLDOWN,
    MONSTER_DAMAGE,
    MONSTER_RESPAWN_TICKS,
    MAX_PLAYERS,
    PLAYER_ATTACK,
    PLAYER_ATTACK_COOLDOWN,
    PLAYER_MAX_HP,
)
from game.spatial import build_grid, nearest_in_block


def _respawn_players(p, m):
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
        "hp": jnp.where(r, PLAYER_MAX_HP, p["hp"]),
        "dead": p["dead"] & ~r,
        "attack_cd": jnp.where(r, 0, p["attack_cd"]),
    }


def combat(world, inputs):
    """Run one combat tick and return the updated world."""
    p = _respawn_players(world["players"], world["monsters"])
    m = world["monsters"]
    n_p = p["hp"].shape[0]
    alive = p["active"] & ~p["dead"]
    pidx = jnp.arange(n_p, dtype=jnp.int32)

    # Player attacks on the nearest adjacent monster.
    mtable, _ = build_grid(m["pos"], m["active"])
    tgt, dist = nearest_in_block(mtable, p["pos"], m["pos"])
    hit = alive & inputs["attack"] & (p["attack_cd"] == 0) & (dist <= 1) & m["active"][tgt]
    tgt_w = jnp.where(hit, tgt, m["hp"].shape[0])  # out of bounds = dropped
    dmg_m = jnp.zeros_like(m["hp"]).at[tgt_w].add(PLAYER_ATTACK, mode="drop")
    new_m_hp = m["hp"] - dmg_m
    killed = m["active"] & (m["hp"] > 0) & (new_m_hp <= 0)
    # Lowest-index hitter of each monster gets the kill credit.
    first = jnp.full(m["hp"].shape, n_p, jnp.int32).at[tgt_w].min(pidx, mode="drop")
    credit = jnp.where(killed, first, n_p)
    kills = p["kills"].at[credit].add(1, mode="drop")
    p_cd = jnp.where(hit, PLAYER_ATTACK_COOLDOWN, jnp.maximum(p["attack_cd"] - 1, 0))

    # Monster attacks on the nearest adjacent living player outside the protection zone (start-of-phase hp).
    targetable = alive & ~world["pz"][p["pos"][:, 0], p["pos"][:, 1]]
    ptable, _ = build_grid(p["pos"], targetable)
    ptgt, pdist = nearest_in_block(ptable, m["pos"], p["pos"])
    mhit = m["active"] & (m["hp"] > 0) & (m["attack_cd"] == 0) & (pdist <= 1) & alive[ptgt]
    mhit = mhit & ~world["pz"][p["pos"][ptgt, 0], p["pos"][ptgt, 1]]  # protection zone
    dmg_tbl = jnp.array(MONSTER_DAMAGE, jnp.int32)
    ptgt_w = jnp.where(mhit, ptgt, MAX_PLAYERS)
    dmg_p = jnp.zeros_like(p["hp"]).at[ptgt_w].add(dmg_tbl[m["kind"]], mode="drop")
    new_p_hp = jnp.maximum(p["hp"] - dmg_p, 0)
    m_cd = jnp.where(mhit, MONSTER_ATTACK_COOLDOWN, jnp.maximum(m["attack_cd"] - 1, 0))

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
    return {**world, "players": players, "monsters": monsters}
