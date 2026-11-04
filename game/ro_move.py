"""RO movement: 8 directions with per-entity speed accumulators (docs/round2_contract.md section 1.1).

Entities are players then monsters: entity e < MAX_PLAYERS is player e, entity MAX_PLAYERS + i is monster i.
Every tick an entity that wants to move (`path_dir` 0..7) adds TICK_MS to `acc_ms`; it steps one cell when
`acc_ms >= cost` (cost = speed_ms straight, speed_ms * 14 // 10 diagonal) and keeps the remainder. A blocked step
keeps the accumulator (capped at the cost) and the intent, so it retries next tick. An entity with no intent
resets its accumulator, so a walk from rest takes one full step time before the first step.

Directions: move codes 0 stay, 1..8 = S, SW, W, NW, N, NE, E, SE (x right, y down). Facing and path_dir use the
RO convention 0..7 = code - 1 (0 = south, clockwise). A diagonal needs both orthogonal neighbours walkable (no
corner cutting through walls), as RO.

Occupancy and contested tiles are resolved by sorting tile keys (no MAP x MAP scatters under vmap): a step is
blocked by any living entity's start-of-tick tile, and the lowest entity index wins a tile two entities want.
"""
import jax
import jax.numpy as jnp

from game.config import MAP, MAX_MONSTERS, MAX_PLAYERS, TICK_MS
from game.data import monsters as _mon_data

# Move codes 0..8; the lead mirrors this into config.DIR_DELTAS.
DIR_DELTAS9 = ((0, 0), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))
LEGACY_TO_RO = (0, 5, 1, 3, 7)  # old 4-way codes 1 N, 2 S, 3 W, 4 E -> RO codes
N_ENT = MAX_PLAYERS + MAX_MONSTERS
DEFAULT_SPEED_MS = 150
DIAG_NUM, DIAG_DEN = 14, 10
MAX_SPEED_MS = 2000  # clamp so a diagonal cost stays inside int16 accumulators

_D9 = jnp.asarray(DIR_DELTAS9, jnp.int32)
_IS_PLAYER = jnp.arange(N_ENT) < MAX_PLAYERS
_IDX = jnp.arange(N_ENT, dtype=jnp.int32)
_NO_TILE = MAP * MAP  # sentinel key, sorts after every real tile
# Claim keys are plain tile indices (< MAP * MAP) and _occupied sorts 2 * key + 1: both fit int32 up to MAP 32767.
# Never multiply a tile key by MAX_PLAYERS / N_ENT (that overflowed at MAP 2048); the stable sort in
# _first_claims gives the lowest index the win. game/ro_warps.py and game/ro_waybell.py reuse these helpers.
assert 2 * MAP * MAP + 1 < 2 ** 31, "tile keys no longer fit int32: switch _occupied/_first_claims to int64"
# Monster ms/cell per kind: game/data/monsters.SPEED_MS once ro-monsters adds it, else MOVE_TICKS steps of 150 ms.
_MON_SPEED = jnp.asarray(getattr(_mon_data, "SPEED_MS", _mon_data.MOVE_TICKS * DEFAULT_SPEED_MS), jnp.int32)
_LEGACY_ACC = 30000  # legacy path: an accumulator this full steps on any due tick


def make_state():
    """The world["move"] block, all entities idle facing south at the default speed."""
    return {
        "speed_ms": jnp.full((N_ENT,), DEFAULT_SPEED_MS, jnp.int16),
        "acc_ms": jnp.full((N_ENT,), MAX_SPEED_MS, jnp.int16),  # rested: step_entities caps it at one step cost
        "dir": jnp.zeros((N_ENT,), jnp.int8),
        "path_dir": jnp.full((N_ENT,), -1, jnp.int8),
        "moving": jnp.zeros((N_ENT,), jnp.bool_),
    }


def code_to_dir(code):
    """Move code 0..8 -> path_dir 0..7, or -1 for 0 (stay) and anything out of range."""
    code = jnp.asarray(code, jnp.int32)
    return jnp.where((code >= 1) & (code <= 8), code - 1, -1)


def dir_toward(delta):
    """path_dir 0..7 for one step along sign(delta) [.., 2] (diagonal when both axes differ), -1 when delta is 0."""
    s = jnp.sign(delta).astype(jnp.int32)
    # code by (sx + 1) * 3 + (sy + 1): sx, sy in -1..1
    lut = jnp.array([4, 3, 2, 5, 0, 1, 6, 7, 8], jnp.int32)  # NW, W, SW, N, stay, S, NE, E, SE
    return code_to_dir(lut[(s[..., 0] + 1) * 3 + (s[..., 1] + 1)])


def attach(world):
    """(world, legacy). Until state.make_empty_world carries the block, phases run on a temporary one.

    Legacy keeps today's pace: players step on every tick they ask, monsters on ticks where
    tick % MOVE_TICKS[kind] == 0. Remove with detach() once the lead wires `"move": make_state()`.
    """
    if "move" in world:
        return world, False
    due = world["tick"] % _mon_data.MOVE_TICKS[world["monsters"]["kind"]] == 0
    acc = jnp.concatenate([jnp.full((MAX_PLAYERS,), _LEGACY_ACC, jnp.int32),
                           jnp.where(due, _LEGACY_ACC, -_LEGACY_ACC)])
    st = {**make_state(), "acc_ms": acc}
    return {**world, "move": st}, True


def detach(world, legacy):
    """Drop the temporary block attach() added."""
    if not legacy:
        return world
    return {k: v for k, v in world.items() if k != "move"}


def want(world, path_dir, mask=None):
    """Set move.path_dir [N] (0..7 or -1) where mask; default mask = path_dir >= 0 (-1 leaves the old intent)."""
    path_dir = jnp.asarray(path_dir, jnp.int32)
    mask = path_dir >= 0 if mask is None else mask
    mv = world["move"]
    pd = jnp.where(mask, path_dir, mv["path_dir"].astype(jnp.int32)).astype(jnp.int8)
    return {**world, "move": {**mv, "path_dir": pd}}


def want_players(world, codes):
    """Players' move codes 0..8 [MAX_PLAYERS]: a code sets the intent, 0 keeps any pending one."""
    pd = jnp.concatenate([code_to_dir(codes), jnp.full((MAX_MONSTERS,), -1, jnp.int32)])
    return want(world, pd)


def want_monsters(world, path_dir):
    """Monsters' intent [MAX_MONSTERS] (0..7 or -1), overwriting every monster's previous one."""
    pd = jnp.concatenate([jnp.full((MAX_PLAYERS,), -1, jnp.int32), jnp.asarray(path_dir, jnp.int32)])
    return want(world, pd, ~_IS_PLAYER)


def _occupied(target_flat, occupant_flat):
    """True where a target tile key matches an occupant key (sentinels never match a real target).

    One sort over occupants and targets together, then a running max of occupant keys. Occupants sort
    before targets with the same key, so a target is occupied exactly when the latest occupant key equals its own.
    """
    n_occ = occupant_flat.shape[0]
    keys = jnp.concatenate([occupant_flat, target_flat])
    is_occ = jnp.concatenate([jnp.ones((n_occ,), jnp.bool_), jnp.zeros((target_flat.shape[0],), jnp.bool_)])
    order = jnp.argsort(keys * 2 + (~is_occ).astype(jnp.int32))
    k_sorted = keys[order]
    occ_sorted = is_occ[order]
    latest_occ = jax.lax.cummax(jnp.where(occ_sorted, k_sorted, -1), axis=0)
    hit_sorted = ~occ_sorted & (latest_occ == k_sorted)
    return hit_sorted[jnp.argsort(order)][n_occ:]  # argsort inverts the permutation without a scatter


def _first_claims(keys):
    """True for the lowest index among equal keys (keys == _NO_TILE never win). Stable sort keeps index order."""
    order = jnp.argsort(keys, stable=True)
    k = keys[order]
    first = jnp.concatenate([jnp.array([True]), k[1:] != k[:-1]]) & (k < _NO_TILE)
    return first[jnp.argsort(order)]


def step_entities(world, group=None):
    """Advance accumulators and take at most one step per entity of `group` ("players", "monsters" or None = all)."""
    mv, p, m = world["move"], world["players"], world["monsters"]
    grp = _IS_PLAYER if group == "players" else ~_IS_PLAYER if group == "monsters" else jnp.ones((N_ENT,), bool)
    pos = jnp.concatenate([p["pos"], m["pos"]])
    alive = jnp.concatenate([p["active"] & ~p["dead"], m["active"]])
    speed = jnp.where(_IS_PLAYER, mv["speed_ms"].astype(jnp.int32),
                      jnp.concatenate([jnp.zeros((MAX_PLAYERS,), jnp.int32), _MON_SPEED[m["kind"]]]))
    speed = jnp.clip(speed, 1, MAX_SPEED_MS)
    pd = mv["path_dir"].astype(jnp.int32)
    pd = jnp.where(grp & ~alive, -1, pd)  # the dead and logged-out forget their intent
    wants = grp & alive & (pd >= 0)

    delta = _D9[jnp.where(wants, pd + 1, 0)]
    diag = jnp.all(delta != 0, axis=1)
    cost = jnp.where(diag, speed * DIAG_NUM // DIAG_DEN, speed)
    acc = mv["acc_ms"].astype(jnp.int32)
    # acc is "ms since the last step", capped at one step cost: a rested entity steps on its first intent (RO
    # starts walking at once) and nobody banks more than one step by standing still.
    acc = jnp.where(grp, jnp.minimum(acc, cost), acc)
    acc = jnp.where(wants, acc + TICK_MS, jnp.where(grp, jnp.minimum(acc + TICK_MS, cost), acc))
    ready = wants & (acc >= cost)

    target = pos + delta
    in_bounds = jnp.all((target >= 0) & (target < MAP), axis=1)
    tc = jnp.clip(target, 0, MAP - 1)
    walls = world["walls"]
    side_x = walls[tc[:, 0], pos[:, 1]] == 0  # orthogonal neighbours of a diagonal step
    side_y = walls[pos[:, 0], tc[:, 1]] == 0
    free = (walls[tc[:, 0], tc[:, 1]] == 0) & (~diag | (side_x & side_y))
    free = free & (_IS_PLAYER | ~world["pz"][tc[:, 0], tc[:, 1]])  # monsters never enter protection zones
    tflat = tc[:, 0] * MAP + tc[:, 1]
    occupied = _occupied(tflat, jnp.where(alive, pos[:, 0] * MAP + pos[:, 1], _NO_TILE))
    attempt = ready & in_bounds & free & ~occupied
    won = _first_claims(jnp.where(attempt, tflat, _NO_TILE))

    new_pos = jnp.where(won[:, None], target, pos)
    acc = jnp.where(won, acc - cost, jnp.minimum(acc, cost))
    acc = jnp.clip(acc, -_LEGACY_ACC, _LEGACY_ACC)
    pd_left = jnp.where(won, -1, pd)
    move = {
        "speed_ms": jnp.where(grp, speed, mv["speed_ms"].astype(jnp.int32)).astype(jnp.int16),
        "acc_ms": acc.astype(jnp.int16),
        "dir": jnp.where(wants, pd, mv["dir"].astype(jnp.int32)).astype(jnp.int8),
        "path_dir": pd_left.astype(jnp.int8),
        "moving": jnp.where(grp, won | (alive & (pd_left >= 0)), mv["moving"]),
    }
    players = {**p, "pos": new_pos[:MAX_PLAYERS]}
    monsters = {**m, "pos": new_pos[MAX_PLAYERS:]}
    return {**world, "players": players, "monsters": monsters, "move": move}
