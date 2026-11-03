"""Gatherer bot policy for the M4 gather/craft phase. Stateless, pure JAX, read-only on the world.

Every GATHER_EVERY-th slot (i % GATHER_EVERY == 1) is a gatherer of node kind (i // GATHER_EVERY) % N_GATHER_KINDS.
Other slots keep the hunting behaviour of bots.policies.bot_inputs. See docs/craft_brief.md (Worker C).
"""
import jax
import jax.numpy as jnp

from bots import policies
from bots.ro_nav_policy import blocked as _wall_blocked
from bots.ro_nav_policy import neighbours
from game import ro_craft
from game.config import HUNT_RETREAT_HP, MAP, MAX_PLAYERS, PLAYER_MAX_HP
from game.data.craft import N_GATHER_KINDS, N_RECIPES, NODE_COPPER, NODE_LEVEL_REQ_A, NODE_SKILL_A, RECIPE_STATION
from game.map_gen import DIST_UNREACHABLE

GATHER_EVERY = 4  # slot i is a gatherer when i % GATHER_EVERY == 1

_IDX = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
_IS_GATHERER = _IDX % GATHER_EVERY == 1
_OWN_KIND = ((_IDX // GATHER_EVERY) % N_GATHER_KINDS).astype(jnp.int32)


def _descend(world, kind, field, enter_pz):
    """Return (move, can): the move code (bots.ro_nav_policy.DELTAS) to the smallest-distance neighbour of field[kind] and whether it lowers it.

    `kind` is a per-player int array; the field is only ever indexed as field[kind, x, y] (never field[kind] per player).
    Walls and tiles holding a living player are unreachable; pz neighbours too unless enter_pz (walking to a station in town) or the player already
    stands on pz (leaving town: the field only decreases, so it never turns back).
    Ties go to the lowest direction code.
    """
    pos = world["players"]["pos"]
    own = field[kind, pos[:, 0], pos[:, 1]].astype(jnp.int32)
    tgt, deltas = neighbours(pos)  # [N, K, 2]
    nbr = field[kind[:, None], tgt[..., 0], tgt[..., 1]].astype(jnp.int32)
    # Tiles holding a living player count as blocked: two gatherers descending different fields can otherwise
    # want each other's tile forever (the shared dist_town field of the hunting bots never has that problem).
    live = world["players"]["active"] & ~world["players"]["dead"]
    occ = jnp.zeros((MAP * MAP,), jnp.bool_).at[jnp.where(live, pos[:, 0] * MAP + pos[:, 1], MAP * MAP)].set(
        True, mode="drop")
    blocked = _wall_blocked(world, pos, deltas) | occ[tgt[..., 0] * MAP + tgt[..., 1]]
    on_pz = world["pz"][pos[:, 0], pos[:, 1]]
    town = world["pz"][tgt[..., 0], tgt[..., 1]] & ~(on_pz[:, None] | enter_pz)
    nbr = jnp.where(blocked | town, DIST_UNREACHABLE, nbr)
    move = (jnp.argmin(nbr, axis=1) + 1).astype(jnp.int32)
    can = (own < DIST_UNREACHABLE) & (nbr.min(axis=1) < own)
    return move, can, own


def craft_bot_inputs(world, key):
    """Return the base bot inputs plus "gather" (bool) and "craft" (int32 recipe id), each [MAX_PLAYERS]."""
    base = policies.bot_inputs(world, key)
    pl, cr = world["players"], world["craft"]
    field = cr["dist_node"]

    live = pl["active"] & ~pl["dead"]
    on_pz = world["pz"][pl["pos"][:, 0], pl["pos"][:, 1]]
    hurt = pl["hp"] < HUNT_RETREAT_HP
    resting = on_pz & (pl["hp"] < PLAYER_MAX_HP)
    eligible = _IS_GATHERER & live & ~hurt & ~resting & ~base["attack"]

    # A gatherer whose skill is below its node kind's requirement works copper until it levels up.
    own_ok = cr["level"][_IDX, NODE_SKILL_A[_OWN_KIND]] >= NODE_LEVEL_REQ_A[_OWN_KIND]
    kind = jnp.where(own_ok, _OWN_KIND, NODE_COPPER).astype(jnp.int32)

    # 1. Craft: the highest recipe whose inputs the bot holds (any chain: a bot gathers whatever it passes, and
    # e.g. a miner holding fish should cook it rather than carry it around full).
    holds = ro_craft.has_inputs(world)
    rid = (N_RECIPES - 1 - jnp.argmax(holds[:, ::-1], axis=1)).astype(jnp.int32)
    has_any = holds.any(axis=1)
    rid = jnp.where(has_any, rid, 0)
    do_craft = has_any & ro_craft.can_craft(world)[_IDX, rid]
    station = jnp.maximum(RECIPE_STATION[rid], 0)
    st_move, st_can, _ = _descend(world, station, field, True)
    # Walk to a station only with a full bag: commuting into town after every single fish or ore wastes the bot.
    room = ro_craft.has_room(pl["inv"])
    walk_station = has_any & ~do_craft & st_can & ~room

    # 2. Gather: stand next to an active node and harvest, else walk down the field of the node kind.
    _, adj_kind, adj_ok = ro_craft.adjacent_node(world)
    adj_lvl_ok = cr["level"][_IDX, NODE_SKILL_A[adj_kind]] >= NODE_LEVEL_REQ_A[adj_kind]
    free = eligible & ~do_craft & ~walk_station & room
    do_gather = free & adj_ok & adj_lvl_ok
    nd_move, nd_can, nd_own = _descend(world, kind, field, False)
    # Within 1 of the field but no active node adjacent (depleted): fall through to the base wander so the bot drifts on.
    walk_node = free & ~do_gather & nd_can & (nd_own > 1)

    move = base["move"]
    move = jnp.where(eligible & walk_node, nd_move, move)
    move = jnp.where(eligible & walk_station, st_move, move)
    move = jnp.where(eligible & (do_craft | do_gather), 0, move)
    return {
        **base,
        "move": move.astype(jnp.int32),
        "gather": eligible & do_gather,
        "craft": jnp.where(eligible & do_craft, rid, 0).astype(jnp.int32),
    }
