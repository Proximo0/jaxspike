"""Bot input policies. Read-only on the world."""
import jax
import jax.numpy as jnp

from game.config import (DIR_DELTAS, HUNT_RETREAT_HP, ITEM_MEAT, MAP, MAX_PLAYERS,
                         PLAYER_MAX_HP)
from game.map_gen import DIST_UNREACHABLE
from game.spatial import build_grid, nearest_in_block
from game.world_layout import RAT_RING, TOWNS

BOT_CHASE_RANGE = 6  # manhattan; must stay <= CELL since the search is 3x3 cells only
HUNT_RADIUS = int(RAT_RING * MAP)  # bots hunt between half and twice this walking distance from their gate
HEADING_TICKS = 50  # ticks a hunting bot keeps one wander heading

_DIRS = jnp.array(DIR_DELTAS[1:], jnp.int32)  # moves 1..4
_HOME = jnp.arange(MAX_PLAYERS, dtype=jnp.int32) % len(TOWNS)  # slot i lives in town i % NT


def _neighbour_dist(world, pos):
    """Return (own, nbr): home-gate distance at pos [N] and at its four neighbours [N, 4]; blocked ones are unreachable."""
    field = world["dist_town"]  # [NT, MAP, MAP]; index per player, never gather per player
    own = field[_HOME, pos[:, 0], pos[:, 1]].astype(jnp.int32)
    tgt = jnp.clip(pos[:, None, :] + _DIRS[None], 0, MAP - 1)  # [N, 4, 2]
    nbr = field[_HOME[:, None], tgt[..., 0], tgt[..., 1]].astype(jnp.int32)
    blocked = world["walls"][tgt[..., 0], tgt[..., 1]] != 0
    town = world["pz"][tgt[..., 0], tgt[..., 1]]
    return own, jnp.where(blocked, DIST_UNREACHABLE, nbr), town


def _gate_moves(world):
    """Return (toward, away, can_toward, can_away, own): moves 1..4 that lower / raise the home-gate distance."""
    own, nbr, town = _neighbour_dist(world, world["players"]["pos"])
    toward = jnp.argmin(nbr, axis=1) + 1
    can_toward = nbr.min(axis=1) < own
    # Never step back into town when walking away: the gate ties with the road and would jam the gate.
    far = jnp.where((nbr == DIST_UNREACHABLE) | town, -1, nbr)
    away = jnp.argmax(far, axis=1) + 1
    can_away = far.max(axis=1) > own
    return toward, away, can_toward, can_away, own


def bot_inputs(world, key):
    """Return {"move": int32, "attack": bool, "eat": bool}, each [MAX_PLAYERS], for every player slot.

    A bot hunts near its home town: it walks out of the gate, away along the distance field until
    HUNT_RADIUS, chases and fights monsters it meets, and heads home below HUNT_RETREAT_HP. In town
    it rests until full HP, then goes out again.
    """
    pl, mon = world["players"], world["monsters"]
    live = pl["active"] & ~pl["dead"]
    on_pz = world["pz"][pl["pos"][:, 0], pl["pos"][:, 1]]
    hurt = pl["hp"] < HUNT_RETREAT_HP
    resting = on_pz & (pl["hp"] < PLAYER_MAX_HP)

    # nearest_in_block only sees the 3x3 cell block around the player, so this
    # is a bounded search: monsters beyond that block are never found, which is
    # fine because BOT_CHASE_RANGE (6) is below CELL (8).
    table, _ = build_grid(mon["pos"], mon["active"])
    idx, dist = nearest_in_block(table, pl["pos"], mon["pos"])
    chase = live & ~hurt & ~on_pz & (dist <= BOT_CHASE_RANGE) & (dist > 0)

    # Step along the longer axis (x on ties). 1 N(y-1) 2 S(y+1) 3 W(x-1) 4 E(x+1).
    d = mon["pos"][idx] - pl["pos"]
    dx, dy = d[:, 0], d[:, 1]
    x_move = jnp.where(dx > 0, 4, 3)
    y_move = jnp.where(dy > 0, 2, 1)
    toward_mon = jnp.where(jnp.abs(dx) >= jnp.abs(dy), x_move, y_move)

    k_go, k_dir = jax.random.split(key)
    rand_dir = jax.random.randint(k_dir, (MAX_PLAYERS,), 0, 5)
    wander = jnp.where(jax.random.bernoulli(k_go, 0.5, (MAX_PLAYERS,)), rand_dir, 0)

    # Outside the hunt radius a bot keeps a heading for HEADING_TICKS so it covers ground
    # instead of diffusing; the heading is a pure function of (slot, tick), so no state.
    k_head = jax.random.fold_in(jax.random.PRNGKey(0), world["tick"] // HEADING_TICKS)
    heading = jax.random.randint(k_head, (MAX_PLAYERS,), 1, 5)
    ahead = jnp.clip(pl["pos"] + _DIRS[heading - 1], 0, MAP - 1)
    blocked_ahead = world["walls"][ahead[:, 0], ahead[:, 1]] != 0
    heading = jnp.where(blocked_ahead, jnp.where(wander == heading, 0, wander), heading)

    to_gate, away, can_to, can_away, own = _gate_moves(world)
    go_home = (hurt | on_pz | (own > 2 * HUNT_RADIUS)) & can_to  # healthy in town: leads out through the gate
    go_out = ~hurt & ~on_pz & (own < HUNT_RADIUS // 2) & can_away
    move = jnp.where(on_pz, wander, heading)
    move = jnp.where(go_out, away, move)
    move = jnp.where(go_home, to_gate, move)
    move = jnp.where(resting & ~hurt, 0, move)
    move = jnp.where(chase, toward_mon, move)
    move = jnp.where(live, move, 0).astype(jnp.int32)
    attack = chase & (dist == 1)
    has_meat = jnp.any(pl["inv"] == ITEM_MEAT, axis=1)
    eat = live & (pl["hp"] < PLAYER_MAX_HP // 2) & has_meat
    return {"move": move, "attack": attack, "eat": eat}
