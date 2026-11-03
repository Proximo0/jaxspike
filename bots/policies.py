"""Bot input policies. Read-only on the world."""
import jax
import jax.numpy as jnp

from game.config import HUNT_RETREAT_HP, ITEM_MEAT, MAP, MAX_PLAYERS, PLAYER_MAX_HP
from game.map_gen import DIST_UNREACHABLE
from game.ro_move import DIR_DELTAS9, dir_toward
from game.spatial import build_grid, nearest_in_block
from game.world_layout import RAT_RING, TOWNS

BOT_CHASE_RANGE = 6  # manhattan; must stay <= CELL since the search is 3x3 cells only
HUNT_RADIUS = int(RAT_RING * MAP)  # bots hunt between half and twice this walking distance from their gate
HEADING_TICKS = 50  # ticks a hunting bot keeps one wander heading

_DIRS = jnp.array(DIR_DELTAS9[1:], jnp.int32)  # moves 1..8: S, SW, W, NW, N, NE, E, SE
_DIRS9 = jnp.array(DIR_DELTAS9, jnp.int32)  # index by move code 0..8
_IS_DIAG = (_DIRS[:, 0] != 0) & (_DIRS[:, 1] != 0)  # [8] int-castable tie-break weights
_IS_DIAG = _IS_DIAG.astype(jnp.int32)[None]
_HOME = jnp.arange(MAX_PLAYERS, dtype=jnp.int32) % len(TOWNS)  # slot i lives in town i % NT


def _blocked(world, pos, deltas):
    """[N, K] True where stepping pos [N, 2] by deltas [N, K, 2] hits a wall; a diagonal also may not cut a wall corner."""
    walls = world["walls"]
    tgt = jnp.clip(pos[:, None, :] + deltas, 0, MAP - 1)
    side_x = walls[tgt[..., 0], pos[:, None, 1]] != 0  # orthogonal neighbours of a diagonal step
    side_y = walls[pos[:, None, 0], tgt[..., 1]] != 0
    diag = (deltas[..., 0] != 0) & (deltas[..., 1] != 0)
    return (walls[tgt[..., 0], tgt[..., 1]] != 0) | (diag & (side_x | side_y))


def _neighbour_dist(world, pos):
    """Return (own, nbr, town): home-gate distance at pos [N], at its eight neighbours [N, 8] (blocked or
    corner-cutting ones unreachable), and whether each neighbour is a protection zone [N, 8]."""
    field = world["dist_town"]  # [NT, MAP, MAP]; index per player, never gather per player
    own = field[_HOME, pos[:, 0], pos[:, 1]].astype(jnp.int32)
    deltas = jnp.broadcast_to(_DIRS[None], (pos.shape[0], 8, 2))
    tgt = jnp.clip(pos[:, None, :] + deltas, 0, MAP - 1)  # [N, 8, 2]
    nbr = field[_HOME[:, None], tgt[..., 0], tgt[..., 1]].astype(jnp.int32)
    town = world["pz"][tgt[..., 0], tgt[..., 1]]
    return own, jnp.where(_blocked(world, pos, deltas), DIST_UNREACHABLE, nbr), town


def _gate_moves(world):
    """Return (toward, away, can_toward, can_away, own): moves 1..8 that lower / raise the home-gate distance."""
    own, nbr, town = _neighbour_dist(world, world["players"]["pos"])
    # Ties go to an orthogonal step: scale distances by 2 and charge a diagonal 1 extra (away: 1 less).
    toward = jnp.argmin(nbr * 2 + _IS_DIAG, axis=1) + 1
    can_toward = nbr.min(axis=1) < own
    # Never step back into town when walking away: the gate ties with the road and would jam the gate.
    far = jnp.where((nbr == DIST_UNREACHABLE) | town, -1, nbr)
    away = jnp.argmax(jnp.where(far < 0, -1, far * 2 - _IS_DIAG), axis=1) + 1
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

    # Step toward the monster, diagonally when both axes differ. A diagonal neighbour (|dx| == |dy| == 1)
    # cannot be hit (range is orthogonal), so step along x to end up orthogonally adjacent.
    d = mon["pos"][idx] - pl["pos"]
    diag_nbr = (jnp.abs(d[:, 0]) == 1) & (jnp.abs(d[:, 1]) == 1)
    d = jnp.where(diag_nbr[:, None], d * jnp.array([1, 0], jnp.int32), d)
    toward_mon = dir_toward(d) + 1  # path_dir -1 (no step) becomes move 0

    k_go, k_dir = jax.random.split(key)
    rand_dir = jax.random.randint(k_dir, (MAX_PLAYERS,), 0, 9)
    wander = jnp.where(jax.random.bernoulli(k_go, 0.5, (MAX_PLAYERS,)), rand_dir, 0)

    # Outside the hunt radius a bot keeps a heading for HEADING_TICKS so it covers ground
    # instead of diffusing; the heading is a pure function of (slot, tick), so no state.
    k_head = jax.random.fold_in(jax.random.PRNGKey(0), world["tick"] // HEADING_TICKS)
    heading = jax.random.randint(k_head, (MAX_PLAYERS,), 1, 9)
    blocked_ahead = _blocked(world, pl["pos"], _DIRS[heading - 1][:, None, :])[:, 0]
    wander_blocked = _blocked(world, pl["pos"], _DIRS9[wander][:, None, :])[:, 0]
    wander = jnp.where(wander_blocked, 0, wander)
    heading = jnp.where(blocked_ahead, wander, heading)  # a blocked heading falls back to the (unblocked) wander move

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
