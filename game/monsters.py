"""Monster AI phase: respawn countdown and one-step chase toward the nearest player."""
import jax.numpy as jnp

from game.config import MAP, MAX_MONSTERS, MONSTER_AGGRO_RANGE, MONSTER_HP
from game.spatial import build_grid, nearest_in_block

_HP_TABLE = jnp.array(MONSTER_HP, dtype=jnp.int32)


def _respawn(m, occ):
    """Count down dead monsters and reset the ones that reach 0 onto a free spawn tile.

    A monster whose spawn tile is occupied waits there (countdown held at 1) and retries next tick.
    """
    dead = ~m["active"] & (m["respawn_in"] > 0)
    left = jnp.where(dead, m["respawn_in"] - 1, m["respawn_in"])
    spawn_blocked = occ[m["spawn"][:, 0], m["spawn"][:, 1]]
    left = jnp.where(dead & (left == 0) & spawn_blocked, 1, left)
    revive = dead & (left == 0) & ~spawn_blocked
    return {
        **m,
        "respawn_in": left,
        "active": m["active"] | revive,
        "hp": jnp.where(revive, _HP_TABLE[m["kind"]], m["hp"]),
        "pos": jnp.where(revive[:, None], m["spawn"], m["pos"]),
        "attack_cd": jnp.where(revive, 0, m["attack_cd"]),
    }


def _occupancy(world):
    """Bool [MAP, MAP]: tiles holding a wall, a living player or an active monster. Dead players are not physical."""
    p, m = world["players"], world["monsters"]
    occ = world["walls"] != 0
    occ = occ.at[p["pos"][:, 0], p["pos"][:, 1]].max(p["active"] & ~p["dead"])
    return occ.at[m["pos"][:, 0], m["pos"][:, 1]].max(m["active"])


def _chase(world, movers):
    """Move each mover one tile toward its nearest living player; lowest index wins ties."""
    p, m = world["players"], world["monsters"]
    pos = m["pos"]
    table, _ = build_grid(p["pos"], p["active"] & ~p["dead"])
    tidx, dist = nearest_in_block(table, pos, p["pos"])
    chase = movers & (dist <= MONSTER_AGGRO_RANGE)

    delta = p["pos"][tidx] - pos
    sign = jnp.sign(delta)
    x_first = jnp.abs(delta[:, 0]) >= jnp.abs(delta[:, 1])
    step_x = jnp.stack([sign[:, 0], jnp.zeros_like(sign[:, 0])], axis=1)
    step_y = jnp.stack([jnp.zeros_like(sign[:, 1]), sign[:, 1]], axis=1)
    first = jnp.where(x_first[:, None], step_x, step_y)
    second = jnp.where(x_first[:, None], step_y, step_x)

    occ = _occupancy(world)

    def free(step):
        tgt = jnp.clip(pos + step, 0, MAP - 1)
        return (step != 0).any(axis=1) & ~occ[tgt[:, 0], tgt[:, 1]], tgt

    ok1, t1 = free(first)
    ok2, t2 = free(second)
    target = jnp.where(ok1[:, None], t1, t2)
    go = chase & (ok1 | ok2)

    # Scatter-min of monster index per target tile: the lowest index claims it.
    flat = jnp.where(go, target[:, 0] * MAP + target[:, 1], MAP * MAP)
    idx = jnp.arange(MAX_MONSTERS, dtype=jnp.int32)
    claim = jnp.full((MAP * MAP,), MAX_MONSTERS, jnp.int32).at[flat].min(idx, mode="drop")
    won = go & (claim[jnp.minimum(flat, MAP * MAP - 1)] == idx)
    return jnp.where(won[:, None], target, pos)


def monster_ai(world):
    """Run respawn countdown, then move aggroed monsters one step toward players."""
    movers = world["monsters"]["active"]  # monsters respawning this tick do not move yet
    m = _respawn(world["monsters"], _occupancy(world))
    new_pos = _chase(world, movers)
    m = {**m, "pos": jnp.where(movers[:, None], new_pos, m["pos"])}
    return {**world, "monsters": m}
