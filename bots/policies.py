"""Bot input policies. Read-only on the world."""
import jax
import jax.numpy as jnp

from game.config import MAX_PLAYERS
from game.spatial import build_grid, nearest_in_block

BOT_CHASE_RANGE = 6  # manhattan; must stay <= CELL since the search is 3x3 cells only


def bot_inputs(world, key):
    """Return {"move": int32 [MAX_PLAYERS], "attack": bool [MAX_PLAYERS]} for every player slot."""
    pl, mon = world["players"], world["monsters"]
    live = pl["active"] & ~pl["dead"]

    # nearest_in_block only sees the 3x3 cell block around the player, so this
    # is a bounded search: monsters beyond that block are never found, which is
    # fine because BOT_CHASE_RANGE (6) is below CELL (8).
    table, _ = build_grid(mon["pos"], mon["active"])
    idx, dist = nearest_in_block(table, pl["pos"], mon["pos"])
    chase = live & (dist <= BOT_CHASE_RANGE) & (dist > 0)

    # Step along the longer axis (x on ties). 1 N(y-1) 2 S(y+1) 3 W(x-1) 4 E(x+1).
    d = mon["pos"][idx] - pl["pos"]
    dx, dy = d[:, 0], d[:, 1]
    x_move = jnp.where(dx > 0, 4, 3)
    y_move = jnp.where(dy > 0, 2, 1)
    toward = jnp.where(jnp.abs(dx) >= jnp.abs(dy), x_move, y_move)

    k_go, k_dir = jax.random.split(key)
    rand_dir = jax.random.randint(k_dir, (MAX_PLAYERS,), 0, 5)
    wander = jnp.where(jax.random.bernoulli(k_go, 0.5, (MAX_PLAYERS,)), rand_dir, 0)

    move = jnp.where(chase, toward, wander)
    move = jnp.where(live, move, 0).astype(jnp.int32)
    attack = chase & (dist == 1)
    return {"move": move, "attack": attack}
