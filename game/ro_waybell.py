"""Waybell services in the JAX tick: save point and zone-aware teleport.

- `inputs["save"]` (bool): a living player inside a town gets that town's temple centre as respawn.
- `inputs["teleport"]` (int32, -1 = none, else a TOWNS index): a living player standing within WAYBELL_TALK_RANGE
  (Chebyshev) of the waybell tile of the town they are in pays `world_layout.WAYBELL_TELEPORT`'s fare from that town
  to the destination (ryn in world["items"]["ryn"]) and lands on the destination's town station tile
  (`world_layout.waybell_teleport_dest`). No route, not enough ryn, not at a waybell, or an occupied station (a living
  player or an active monster; the lowest slot wins a station two players want) means nothing happens and nothing
  is charged. Pure JAX, fixed shapes; the jump uses game/ro_warps.claim_moves (sorted tile keys, no MAP x MAP).
"""
import jax.numpy as jnp

from game.ro_warps import claim_moves
from game.world_layout import TOWNS, waybell_teleport_cost, waybell_teleport_dest, waybell_tiles, town_boxes, town_centers

WAYBELL_TALK_RANGE = 2  # tiles (Chebyshev) from the waybell tile within which the teleport service answers
_NT = len(TOWNS)


def make_state():
    return {}


def town_of_tile(pos):
    """int32 [N, 2] -> int32 [N]: index of the town whose box holds the tile, -1 outside."""
    boxes = town_boxes()
    out = jnp.full(pos.shape[0], -1, jnp.int32)
    for t in range(boxes.shape[0]):  # loop over a constant (5 towns)
        x0, y0, x1, y1 = (int(v) for v in boxes[t])
        inside = (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
        out = jnp.where(inside & (out < 0), t, out)
    return out


def at_waybell(pos, town):
    """bool [N]: pos is within WAYBELL_TALK_RANGE of the waybell tile of `town` (int32 [N], -1 = none -> False)."""
    k = jnp.asarray(waybell_tiles(), jnp.int32)[jnp.maximum(town, 0)]
    return (town >= 0) & (jnp.max(jnp.abs(pos - k), axis=1) <= WAYBELL_TALK_RANGE)


def teleport(world, dest, town, live):
    """Apply the waybell teleport: dest int32 [P] town index or -1, town [P] = town_of_tile(pos). Returns the world."""
    p, items = world["players"], world["items"]
    want = live & at_waybell(p["pos"], town) & (dest >= 0) & (dest < _NT)
    d = jnp.clip(dest, 0, _NT - 1)
    cost = jnp.asarray(waybell_teleport_cost(), jnp.int32)[jnp.maximum(town, 0), d]
    can = want & (cost >= 0) & (items["ryn"] >= cost)
    target = jnp.asarray(waybell_teleport_dest(), jnp.int32)[d]
    won = claim_moves(world, target, can)
    pos = jnp.where(won[:, None], target, p["pos"])
    ryn = items["ryn"] - jnp.where(won, cost, 0)
    return {**world, "players": {**p, "pos": pos}, "items": {**items, "ryn": ryn}}


def phase(world, inputs):
    p = world["players"]
    town = town_of_tile(p["pos"])
    live = p["active"] & ~p["dead"]
    ok = inputs["save"] & live & (town >= 0)
    centre = jnp.asarray(town_centers(), jnp.int32)[jnp.maximum(town, 0)]
    spawn = jnp.where(ok[:, None], centre, p["spawn"])
    world = {**world, "players": {**p, "spawn": spawn}}
    tp = inputs.get("teleport")  # absent until the lead adds it to state.empty_inputs (structure, not a traced value)
    if tp is not None:
        world = teleport(world, jnp.asarray(tp, jnp.int32), town, live)
    return world
