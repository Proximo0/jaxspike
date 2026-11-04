"""RO monster modes: the decision half of the monster phase (docs/round2_contract.md section 0, team ro-monsters).

Every tick each living monster gets a state and one wanted step, `path_dir` 0..7 (RO convention, -1 = stand),
which game/monsters.py hands to game/ro_move.py; nothing here writes monsters.pos.

States (`world["monai"]["state"]`): 0 idle, 1 chase, 2 return home.
- Idle: random walk. Every WANDER_MIN_MS..WANDER_MAX_MS a goal 1..5 cells away in one of 8 directions, clipped to a
  WANDER_BOX around the spawn tile, never a wall or protection-zone tile. A looter (Blobbin) instead walks to the
  nearest active drop within sight and picks it into `held_item` when within one cell (lowest monster index wins a
  drop: scatter-min); it holds one item and drops it where it dies.
- Chase: entered from idle when the monster is hostile (aggressive kind, hurt, or an assist kind pulled in by a
  kind-mate within sight that is fighting) and a targetable player is within SIGHT[kind] while the monster is within
  CHASE[kind] cells (chebyshev) of its spawn. It follows the nearest player within one grid CELL and stops at
  ATTACK_RANGE (a diagonal neighbour is approached along x, since attacks need an orthogonal neighbour).
- Return: entered when a chase leaves the chase radius or an idle monster stands outside the wander box; walks to the
  spawn ignoring players and turns idle within one cell of it.
- Immovable kinds never move. Cast sensor is a placeholder flag (no cast event to react to yet).

Respawn: when the block exists, a monster that died since the last tick gets RESPAWN_MS[kind] // TICK_MS ticks
(MVP kinds keep game/ro_mvp.py's timer). Without the block (state.make_empty_world has not wired it yet) the same
rules run on a block rebuilt each tick: chase and return-from-outside-the-box still work, wanders shrink to single
steps at the same average rate, looting and per-kind respawn are off (combat's MONSTER_RESPAWN_TICKS applies).
"""
import jax
import jax.numpy as jnp

from game.config import CELL, DROP_TTL, G, ITEM_NONE, K_CELL, MAP, MAX_DROPS, MAX_MONSTERS, MAX_PLAYERS, NCELLS, TICK_MS
from game.data import monsters as mdata
from game.ro_move import DIR_DELTAS9, dir_toward
from game.spatial import BIG, block_candidates
from game.state import next_key

IDLE, CHASE, RETURN = 0, 1, 2
MAX_SIGHT = CELL  # the 3x3 block of the spatial grid covers at least CELL cells around a query
assert int(mdata.SIGHT.max()) <= MAX_SIGHT
WANDER_MIN_TICKS = max(1, mdata.WANDER_MIN_MS // TICK_MS)
WANDER_MAX_TICKS = max(WANDER_MIN_TICKS, mdata.WANDER_MAX_MS // TICK_MS)
RESPAWN_TICKS = jnp.maximum(1, mdata.RESPAWN_MS // TICK_MS)

_STEPS8 = jnp.asarray(DIR_DELTAS9[1:], jnp.int32)  # path_dir 0..7 -> (dx, dy)
_IDX = jnp.arange(MAX_MONSTERS, dtype=jnp.int32)
_MM = MAP * MAP  # packed tile keys x * MAP + y live below this
_WANDER_LEN_N = mdata.WANDER_MAX_CELLS - mdata.WANDER_MIN_CELLS + 1
_WANDER_TICK_N = WANDER_MAX_TICKS - WANDER_MIN_TICKS + 1


# ---- spatial lookups (ro3-perf) ----------------------------------------------------------------------------------
# The same CELL grid as game/spatial.py, but the tables hold packed payloads (tile key, index, kind) instead of only
# indices, so a query gathers 9 contiguous rows of K_CELL per table and never a random [Q, 9 * K_CELL, 2] slice of
# the positions. One single-operand sort of cell * n + index replaces argsort + searchsorted; cell order, index
# order within a cell and the K_CELL overflow rule stay those of spatial.build_grid, so nearest ties break the same.

def _grid_tables(pos, active, payloads):
    """Bucket entities into [NCELLS, K_CELL] tables, one per payload (int32 [n] each; -1 marks an empty entry)."""
    n = pos.shape[0]
    cxy = pos // CELL
    cell = jnp.where(active, cxy[:, 0] * G + cxy[:, 1], NCELLS)
    key = jnp.sort(cell * n + jnp.arange(n, dtype=jnp.int32))
    order = key % n
    sc = key // n
    ar = jnp.arange(n, dtype=jnp.int32)
    is_first = jnp.concatenate([jnp.array([True]), sc[1:] != sc[:-1]])
    rank = ar - jax.lax.cummax(jnp.where(is_first, ar, 0), axis=0)
    keep = (sc < NCELLS) & (rank < K_CELL)
    row = jnp.where(keep, sc, NCELLS)  # dropped entries write out of bounds
    col = jnp.where(keep, rank, 0)
    return [jnp.full((NCELLS, K_CELL), -1, jnp.int32).at[row, col].set(p[order], mode="drop") for p in payloads]


def _packed(pos):
    return pos[:, 0] * MAP + pos[:, 1]


def _manhattan(packed, qpos):
    """|dx| + |dy| from packed tile keys [Q, C] to query positions [Q, 2] (packed must be >= 0)."""
    return jnp.abs(packed // MAP - qpos[:, 0:1]) + jnp.abs(packed % MAP - qpos[:, 1:2])


def _nearest(pos, active, qpos):
    """Nearest manhattan entity of (pos, active) in each query's 3x3 block: (idx int32 [Q], dist int32 [Q]).

    Like spatial.nearest_in_block: idx is a valid index even when dist == BIG (no entity in range)."""
    n = pos.shape[0]
    tidx, tpos = _grid_tables(pos, active, (jnp.arange(n, dtype=jnp.int32), _packed(pos)))
    cidx, cpos = block_candidates(tidx, qpos), block_candidates(tpos, qpos)
    d = jnp.where(cidx >= 0, _manhattan(jnp.maximum(cpos, 0), qpos), BIG)
    col = jnp.argmin(d, axis=1)
    idx = jnp.take_along_axis(jnp.maximum(cidx, 0), col[:, None], axis=1)[:, 0]
    return idx, jnp.min(d, axis=1)


def _fighting_mate_near(pos, kind, fighting, sight):
    """bool [MAX_MONSTERS]: a fighting monster of the same kind, other than itself, within `sight` (manhattan)."""
    (table,) = _grid_tables(pos, fighting, (kind * _MM + _packed(pos),))
    cand = block_candidates(table, pos)
    valid = cand >= 0
    safe = jnp.maximum(cand, 0)
    cpos = safe % _MM
    mate = (valid & (safe // _MM == kind[:, None]) & (cpos != _packed(pos)[:, None])  # no two monsters share a tile
            & (_manhattan(cpos, pos) <= sight[:, None]))
    return mate.any(axis=1)

TABLES = {
    "aggro": mdata.AGGRO, "assist": mdata.ASSIST, "looter": mdata.LOOTER, "immovable": mdata.IMMOVABLE,
    "sight": mdata.SIGHT, "chase": mdata.CHASE, "atk_range": mdata.ATTACK_RANGE, "hp": mdata.HP,
    "mvp": mdata.MVP, "respawn_ticks": RESPAWN_TICKS,
}


def make_state():
    """The world["monai"] block: every monster idle, no item held, wander due at once."""
    return {
        "state": jnp.zeros((MAX_MONSTERS,), jnp.int8),
        "goal": jnp.full((MAX_MONSTERS, 2), -1, jnp.int32),  # -1 = no goal yet
        "wander_in": jnp.zeros((MAX_MONSTERS,), jnp.int32),
        "held_item": jnp.full((MAX_MONSTERS,), ITEM_NONE, jnp.int32),
        "was_active": jnp.zeros((MAX_MONSTERS,), jnp.bool_),
    }


def _cheb(a, b):
    return jnp.abs(a - b).max(axis=-1)


def _place(drops, want, pos, item):
    """Put item[i] at pos[i] for every i with want, in free drop slots (cumsum ranks, like loot._spawn_drops)."""
    rank = jnp.cumsum(want) - 1
    free = ~drops["active"]
    f_rank = jnp.cumsum(free) - 1
    slot_of_rank = jnp.full((MAX_DROPS,), MAX_DROPS, jnp.int32).at[jnp.where(free, f_rank, MAX_DROPS)].set(
        jnp.arange(MAX_DROPS, dtype=jnp.int32), mode="drop")
    slot = slot_of_rank.at[jnp.where(want, rank, MAX_DROPS)].get(mode="fill", fill_value=MAX_DROPS)
    new = {
        **drops,
        "pos": drops["pos"].at[slot].set(pos, mode="drop"),
        "item": drops["item"].at[slot].set(item, mode="drop"),
        "active": drops["active"].at[slot].set(True, mode="drop"),
        "ttl": drops["ttl"].at[slot].set(DROP_TTL, mode="drop"),
    }
    if "count" in drops:
        new["count"] = drops["count"].at[slot].set(1, mode="drop")
    for k in ("refine", "cards"):  # ro3-items3: a spilled item must not inherit a decayed drop's refine/cards
        if k in drops:
            new[k] = drops[k].at[slot].set(0, mode="drop")
    return new


def on_deaths(world, tables=TABLES):
    """Before the respawn countdown: per-kind timers for monsters that died since last tick, held items dropped.

    A no-op without the block."""
    if "monai" not in world:
        return world
    ai, m = world["monai"], world["monsters"]
    kind = m["kind"]
    newly_dead = ai["was_active"] & ~m["active"] & (m["respawn_in"] > 0) & ~tables["mvp"][kind]
    respawn_in = jnp.where(newly_dead, tables["respawn_ticks"][kind], m["respawn_in"])
    spill = ~m["active"] & (ai["held_item"] != ITEM_NONE)
    drops = _place(world["drops"], spill, m["pos"], ai["held_item"])
    held = jnp.where(spill, ITEM_NONE, ai["held_item"])
    return {**world, "monsters": {**m, "respawn_in": respawn_in}, "drops": drops,
            "monai": {**ai, "held_item": held}}


def _wander_goal(bits, pos, spawn, walls, pz):
    """A goal 1..WANDER_MAX_CELLS cells away in one of 8 directions, clipped to the wander box; pos if it is blocked.

    `bits` uint32 [MAX_MONSTERS]: bits 0-2 pick the direction, bits 3-15 the length (one random draw per tick
    serves the goal and the timer; the modulo bias on 13 bits is below 1e-3)."""
    d = (bits & 7).astype(jnp.int32)
    n = mdata.WANDER_MIN_CELLS + ((bits >> 3) & 0x1FFF).astype(jnp.int32) % _WANDER_LEN_N
    goal = pos + _STEPS8[d] * n[:, None]
    goal = jnp.clip(goal, spawn - mdata.WANDER_BOX, spawn + mdata.WANDER_BOX)
    goal = jnp.clip(goal, 0, MAP - 1)
    bad = (walls[goal[:, 0], goal[:, 1]] != 0) | pz[goal[:, 0], goal[:, 1]]
    return jnp.where(bad[:, None], pos, goal)


def _step_dir(world, occ, pos, goal, approach):
    """path_dir toward goal: the 8-direction step, else its x part, else its y part (first one that is free).

    `approach`: chasing monsters reach a diagonal neighbour along x (combat needs an orthogonal neighbour)."""
    walls = world["walls"]
    sign = jnp.sign(goal - pos)
    step_x = sign * jnp.array([1, 0], jnp.int32)
    step_y = sign * jnp.array([0, 1], jnp.int32)
    diag_nbr = approach & (_cheb(goal, pos) == 1) & jnp.all(sign != 0, axis=1)
    first = jnp.where(diag_nbr[:, None], step_x, sign)

    def free(step):
        tgt = jnp.clip(pos + step, 0, MAP - 1)
        corner = (walls[tgt[:, 0], pos[:, 1]] == 0) & (walls[pos[:, 0], tgt[:, 1]] == 0)
        return (step != 0).any(axis=1) & ~occ[tgt[:, 0], tgt[:, 1]] & corner

    ok1, okx, oky = free(first), free(step_x), free(step_y)
    step = jnp.where(ok1[:, None], first, jnp.where(okx[:, None], step_x, step_y))
    return jnp.where(ok1 | okx | oky, dir_toward(step), -1)


def decide(world, movers, occ, tables=TABLES):
    """(world, path_dir int32 [MAX_MONSTERS]). movers: monsters alive at the start of the phase (revived ones stand).

    occ: bool [MAP, MAP] blocked tiles (walls, pz, living entities) for the fallback step choice."""
    has_block = "monai" in world
    key, world = next_key(world)
    # One draw of random bits per monster covers the wander goal (bits 0-15) and the timer (bits 16-31).
    bits = jax.random.bits(key, (MAX_MONSTERS,), jnp.uint32)
    p, m, dr = world["players"], world["monsters"], world["drops"]
    if has_block:
        ai = world["monai"]
    else:  # stateless fallback: each tick a monster starts a wander with probability ~1 / WANDER_MAX_TICKS
        ai = {**make_state(), "goal": m["pos"],
              "wander_in": 1 + ((bits >> 16) & 0xFF).astype(jnp.int32) % WANDER_MAX_TICKS}
    pos, spawn, kind = m["pos"], m["spawn"], m["kind"]
    sight = tables["sight"][kind]
    prev = ai["state"].astype(jnp.int32)

    # Nearest targetable player (living, outside the protection zone).
    targetable = p["active"] & ~p["dead"] & ~world["pz"][p["pos"][:, 0], p["pos"][:, 1]]
    tidx, pdist = _nearest(p["pos"], targetable, pos)
    tpos = p["pos"][tidx]

    # Assist: a kind-mate within sight that is chasing or hurt pulls this monster into the fight.
    hurt = m["active"] & (m["hp"] < tables["hp"][kind])
    fighting = m["active"] & ((prev == CHASE) | hurt)
    pulled = tables["assist"][kind] & _fighting_mate_near(pos, kind, fighting, sight)
    hostile = tables["aggro"][kind] | hurt | pulled

    home_d = _cheb(pos, spawn)
    in_chase = home_d <= tables["chase"][kind]
    out_box = home_d > mdata.WANDER_BOX
    sees = hostile & (pdist <= sight) & in_chase
    keeps = hostile & (pdist <= MAX_SIGHT) & in_chase
    state = jnp.where(
        prev == CHASE, jnp.where(keeps, CHASE, jnp.where(~in_chase | out_box, RETURN, IDLE)),
        jnp.where(prev == RETURN, jnp.where(home_d <= 1, IDLE, RETURN),
                  jnp.where(sees, CHASE, jnp.where(out_box, RETURN, IDLE))))
    state = jnp.where(movers, state, IDLE)
    idle = state == IDLE

    # Looters walk to the nearest drop within sight and pick it up from one cell away.
    if has_block:
        didx, ddist = _nearest(dr["pos"], dr["active"], pos)
        looting = idle & movers & tables["looter"][kind] & (ai["held_item"] == ITEM_NONE) & (ddist <= sight)
        reach = looting & (_cheb(dr["pos"][didx], pos) <= 1) & dr["active"][didx]
        claim = jnp.full((MAX_DROPS,), MAX_MONSTERS, jnp.int32).at[jnp.where(reach, didx, MAX_DROPS)].min(
            _IDX, mode="drop")
        got = reach & (claim[didx] == _IDX)
        taken = claim < MAX_MONSTERS
        held = jnp.where(got, dr["item"][didx], ai["held_item"])
        dr = {**dr, "active": dr["active"] & ~taken, "item": jnp.where(taken, ITEM_NONE, dr["item"])}
        looting = looting & ~got
    else:
        didx = jnp.zeros((MAX_MONSTERS,), jnp.int32)
        looting = jnp.zeros((MAX_MONSTERS,), jnp.bool_)
        held = ai["held_item"]

    # Idle random walk: count down, then roll a new goal and timer.
    strolling = idle & ~looting
    new_timer = WANDER_MIN_TICKS + (bits >> 16).astype(jnp.int32) % _WANDER_TICK_N
    timer = jnp.where(strolling, ai["wander_in"] - 1, new_timer)  # a busy monster waits a full period after
    due = strolling & (timer <= 0)
    keep_goal = strolling & (prev == IDLE) & (ai["goal"][:, 0] >= 0)  # turning idle: start from where it stands
    goal = jnp.where(due[:, None], _wander_goal(bits, pos, spawn, world["walls"], world["pz"]),
                     jnp.where(keep_goal[:, None], ai["goal"], pos))
    timer = jnp.where(due, new_timer, timer)
    goal = jnp.where(movers[:, None], goal, spawn)

    # One wanted step per monster toward its state's target.
    chasing = state == CHASE
    to = jnp.where(chasing[:, None], tpos, jnp.where((state == RETURN)[:, None], spawn,
                                                     jnp.where(looting[:, None], dr["pos"][didx], goal)))
    far = jnp.where(chasing, jnp.abs(tpos - pos).sum(-1) > tables["atk_range"][kind], jnp.any(to != pos, axis=1))
    go = movers & ~tables["immovable"][kind] & far
    path_dir = jnp.where(go, _step_dir(world, occ, pos, to, chasing), -1)

    if has_block:
        ai = {"state": state.astype(jnp.int8), "goal": goal, "wander_in": timer, "held_item": held,
              "was_active": m["active"]}
        world = {**world, "monai": ai, "drops": dr}
    return world, path_dir.astype(jnp.int32)


def counts(world):
    """Diagnostics for soaks: living monsters idle-standing, wandering (idle with a step), chasing, returning, holding."""
    ai, m = world["monai"], world["monsters"]
    live = m["active"]
    st = ai["state"].astype(jnp.int32)
    walking = (ai["goal"][:, 0] >= 0) & jnp.any(ai["goal"] != m["pos"], axis=-1)
    return {
        "stand": jnp.sum(live & (st == IDLE) & ~walking), "wander": jnp.sum(live & (st == IDLE) & walking),
        "chase": jnp.sum(live & (st == CHASE)), "return": jnp.sum(live & (st == RETURN)),
        "held": jnp.sum(ai["held_item"] != ITEM_NONE),
    }
