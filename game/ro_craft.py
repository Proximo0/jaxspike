"""M4 gathering and crafting phase (pure JAX). Tables live in game/data/craft.py; node placement in ro_craft_nodes."""
import jax
import jax.numpy as jnp

from game.config import ITEM_NONE, MAP, MAX_PLAYERS
from game.data.craft import (
    CRAFT_MAX_LEVEL,
    GATHER_RANGE,
    LEVEL_EXP_A,
    MAX_NODES,
    N_CRAFT_SKILLS,
    N_GATHER_KINDS,
    N_NODE_KINDS,
    N_RECIPES,
    NODE_CHANCE_BASE_A,
    NODE_CHANCE_MAX,
    NODE_CHANCE_PER_LEVEL,
    NODE_EXP_A,
    NODE_LEVEL_REQ_A,
    NODE_PRODUCT_A,
    NODE_RESPAWN_A,
    NODE_SKILL_A,
    RECIPE_EXP,
    RECIPE_INPUTS,
    RECIPE_INPUTS_MAX,
    RECIPE_LEVEL,
    RECIPE_OUTPUT,
    RECIPE_SKILL,
    RECIPE_STATION,
)
from game.ro_craft_nodes import node_at, node_fields, nodes_near, place_nodes, station_tiles  # noqa: F401
from game.state import next_key

assert GATHER_RANGE == 1  # nodes_near covers the own tile and the 4 neighbours

_PIDX = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
# _NEED[r, j]: how many input columns of recipe r hold the same item as column j (0 for padding columns).
_NEED = ((RECIPE_INPUTS[:, :, None] == RECIPE_INPUTS[:, None, :]) & (RECIPE_INPUTS[:, :, None] > 0)).sum(-1)


def make_state():
    """Fresh craft block: no nodes placed, every skill at level 1 with 0 exp."""
    nodes = {
        "pos": jnp.zeros((MAX_NODES, 2), jnp.int32),
        "kind": jnp.zeros((MAX_NODES,), jnp.int32),
        "placed": jnp.zeros((MAX_NODES,), jnp.bool_),
        "active": jnp.zeros((MAX_NODES,), jnp.bool_),
        "respawn_in": jnp.zeros((MAX_NODES,), jnp.int32),
    }
    return {
        "nodes": nodes,
        "dist_node": jnp.zeros((N_NODE_KINDS, MAP, MAP), jnp.int16),
        "level": jnp.ones((MAX_PLAYERS, N_CRAFT_SKILLS), jnp.int32),
        "exp": jnp.zeros((MAX_PLAYERS, N_CRAFT_SKILLS), jnp.int32),
        "gathers": jnp.zeros((MAX_PLAYERS,), jnp.int32),
        "crafts": jnp.zeros((MAX_PLAYERS,), jnp.int32),
    }


def with_craft(world, key):
    """Add a craft block to a terrain world: place nodes, then fill the node distance fields."""
    world = {**world, "craft": make_state()}
    world = place_nodes(world, key)
    craft_block = {**world["craft"], "dist_node": node_fields(world)}
    return {**world, "craft": craft_block}


# ---- inventory helpers -----------------------------------------------------------------------------
def has_room(inv):
    """True where the inventory has an empty slot."""
    return (inv == ITEM_NONE).any(axis=1)


def add_item(inv, item):
    """Put item[i] into player i's first empty slot; ITEM_NONE or a full inventory changes nothing."""
    empty = inv == ITEM_NONE
    first = jnp.argmax(empty, axis=1)
    put = (item != ITEM_NONE) & empty.any(axis=1)
    hit = put[:, None] & (jnp.arange(inv.shape[1])[None, :] == first[:, None])
    return jnp.where(hit, item[:, None], inv)


def count_item(inv, item):
    """Number of slots equal to item[i] (0 for ITEM_NONE)."""
    return ((inv == item[:, None]) & (item[:, None] != ITEM_NONE)).sum(axis=1).astype(jnp.int32)


def remove_item(inv, item):
    """Clear the lowest slot equal to item[i]; ITEM_NONE removes nothing."""
    match = (inv == item[:, None]) & (item[:, None] != ITEM_NONE)
    first = jnp.argmax(match, axis=1)
    hit = match.any(axis=1)[:, None] & (jnp.arange(inv.shape[1])[None, :] == first[:, None])
    return jnp.where(hit, ITEM_NONE, inv)


def level_from_exp(exp):
    """Skill level for total exp, in [1, CRAFT_MAX_LEVEL]."""
    lvl = (exp[..., None] >= LEVEL_EXP_A).sum(-1)
    return jnp.clip(lvl, 1, CRAFT_MAX_LEVEL).astype(jnp.int32)


# ---- adjacency -------------------------------------------------------------------------------------
def _near(world):
    """Candidate node indices [MAX_PLAYERS, 5] around each player (MAX_NODES = none)."""
    pos = jnp.clip(world["players"]["pos"], 0, MAP - 1)
    return nodes_near(node_at(world["craft"]["nodes"]), pos)


def adjacent_node(world):
    """Lowest placed, active, gatherable node within GATHER_RANGE: (idx, kind, ok)."""
    nodes = world["craft"]["nodes"]
    cand = _near(world)
    safe = jnp.minimum(cand, MAX_NODES - 1)
    valid = (cand < MAX_NODES) & nodes["placed"][safe] & nodes["active"][safe] & (nodes["kind"][safe] < N_GATHER_KINDS)
    idx = jnp.where(valid, cand, MAX_NODES).min(axis=1).astype(jnp.int32)
    ok = idx < MAX_NODES
    kind = jnp.where(ok, nodes["kind"][jnp.minimum(idx, MAX_NODES - 1)], 0).astype(jnp.int32)
    return idx, kind, ok


def adjacent_station(world, kind):
    """True where a placed node of `kind` is within GATHER_RANGE; kind == -1 means no station needed."""
    nodes = world["craft"]["nodes"]
    cand = _near(world)
    safe = jnp.minimum(cand, MAX_NODES - 1)
    match = (cand < MAX_NODES) & nodes["placed"][safe] & (nodes["kind"][safe] == kind[:, None])
    return match.any(axis=1) | (kind < 0)


# ---- crafting predicates -----------------------------------------------------------------------------
def has_inputs(world):
    """[MAX_PLAYERS, N_RECIPES]: living player with the skill level and every input (with multiplicity)."""
    pl, cr = world["players"], world["craft"]
    alive = pl["active"] & ~pl["dead"]
    level_ok = cr["level"][:, RECIPE_SKILL] >= RECIPE_LEVEL[None, :]
    # counts[p, r, j] = slots of player p equal to recipe r's input j
    counts = (pl["inv"][:, None, None, :] == RECIPE_INPUTS[None, :, :, None]).sum(-1)
    inputs_ok = ((RECIPE_INPUTS[None] == 0) | (counts >= _NEED[None])).all(-1)
    real = jnp.arange(N_RECIPES) > 0
    return alive[:, None] & level_ok & inputs_ok & real[None, :]


def can_craft(world):
    """has_inputs and the recipe's station (if any) within GATHER_RANGE."""
    nodes = world["craft"]["nodes"]
    cand = _near(world)[:, :, None]                                  # [P, 5, 1]
    safe = jnp.minimum(cand, MAX_NODES - 1)
    at = (cand < MAX_NODES) & nodes["placed"][safe] & (nodes["kind"][safe] == RECIPE_STATION[None, None, :])
    st = at.any(axis=1) | (RECIPE_STATION[None, :] < 0)               # [P, N_RECIPES]
    return has_inputs(world) & st


# ---- phases ------------------------------------------------------------------------------------------
def gather(world, inputs):
    """Players with inputs["gather"] harvest one adjacent node; lowest slot wins each node, then a chance roll."""
    pl, cr = world["players"], world["craft"]
    nodes = cr["nodes"]
    idx, kind, ok = adjacent_node(world)
    skill = NODE_SKILL_A[kind]
    lvl = cr["level"][_PIDX, skill]
    req = NODE_LEVEL_REQ_A[kind]
    alive = pl["active"] & ~pl["dead"]
    claim = alive & inputs["gather"] & ok & has_room(pl["inv"]) & (lvl >= req)

    winner = jnp.full((MAX_NODES,), MAX_PLAYERS, jnp.int32).at[jnp.where(claim, idx, MAX_NODES)].min(_PIDX, mode="drop")
    got = claim & (winner[jnp.where(claim, idx, 0)] == _PIDX)

    sub, world = next_key(world)
    u = jax.random.uniform(sub, (MAX_PLAYERS,))
    chance = jnp.minimum(NODE_CHANCE_BASE_A[kind] + (lvl - req) * NODE_CHANCE_PER_LEVEL, NODE_CHANCE_MAX)
    success = got & (u * 1000.0 < chance)

    nidx = jnp.where(success, idx, MAX_NODES)
    new_nodes = {
        **nodes,
        "active": nodes["active"].at[nidx].set(False, mode="drop"),
        "respawn_in": nodes["respawn_in"].at[nidx].set(NODE_RESPAWN_A[kind], mode="drop"),
    }
    inv = add_item(pl["inv"], jnp.where(success, NODE_PRODUCT_A[kind], ITEM_NONE))
    exp = cr["exp"].at[jnp.where(success, _PIDX, MAX_PLAYERS), skill].add(NODE_EXP_A[kind], mode="drop")
    new_cr = {**cr, "nodes": new_nodes, "exp": exp, "level": level_from_exp(exp),
              "gathers": cr["gathers"] + success.astype(jnp.int32)}
    return {**world, "players": {**pl, "inv": inv}, "craft": new_cr}


def respawn(world):
    """Placed inactive nodes count respawn_in down and reactivate at 0."""
    cr = world["craft"]
    nodes = cr["nodes"]
    waiting = nodes["placed"] & ~nodes["active"]
    r = jnp.where(waiting, jnp.maximum(nodes["respawn_in"] - 1, 0), nodes["respawn_in"])
    new_nodes = {**nodes, "respawn_in": r, "active": nodes["active"] | (waiting & (r <= 0))}
    return {**world, "craft": {**cr, "nodes": new_nodes}}


def craft(world, inputs):
    """Players whose recipe is craftable consume its inputs and receive the output."""
    pl, cr = world["players"], world["craft"]
    r = jnp.clip(inputs["craft"], 0, N_RECIPES - 1)
    do = can_craft(world)[_PIDX, r] & (r > 0)
    inv = pl["inv"]
    for j in range(RECIPE_INPUTS_MAX):
        inv = remove_item(inv, jnp.where(do, RECIPE_INPUTS[r, j], ITEM_NONE))
    do = do & has_room(inv)
    inv = jnp.where(do[:, None], add_item(inv, jnp.where(do, RECIPE_OUTPUT[r], ITEM_NONE)), pl["inv"])
    skill = RECIPE_SKILL[r]
    exp = cr["exp"].at[jnp.where(do, _PIDX, MAX_PLAYERS), skill].add(RECIPE_EXP[r], mode="drop")
    new_cr = {**cr, "exp": exp, "level": level_from_exp(exp), "crafts": cr["crafts"] + do.astype(jnp.int32)}
    return {**world, "players": {**pl, "inv": inv}, "craft": new_cr}


def phase(world, inputs):
    """Respawn nodes, then gather, then craft."""
    return craft(gather(respawn(world), inputs), inputs)
