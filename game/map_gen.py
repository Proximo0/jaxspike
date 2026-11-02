"""Map generation: random walls with a border ring and a clear spawn square."""
import jax
import jax.numpy as jnp

from game.config import MAP

WALL_DENSITY = 0.15
SPAWN_LO = 16   # clear square covers tiles SPAWN_LO..SPAWN_HI inclusive
SPAWN_HI = 23


def make_walls(key):
    """Return int8 [MAP, MAP] walls (1 = blocked) with border ring and clear spawn square."""
    walls = jax.random.uniform(key, (MAP, MAP)) < WALL_DENSITY
    idx = jnp.arange(MAP)
    border = (idx == 0) | (idx == MAP - 1)
    walls = walls | border[:, None] | border[None, :]
    spawn = (idx >= SPAWN_LO) & (idx <= SPAWN_HI)
    walls = walls & ~(spawn[:, None] & spawn[None, :])
    return walls.astype(jnp.int8)


def walkable(walls, pos):
    """Return bool [N]: pos (int [N,2]) is in bounds and not a wall."""
    x, y = pos[:, 0], pos[:, 1]
    inb = (x >= 0) & (x < MAP) & (y >= 0) & (y < MAP)
    cx = jnp.clip(x, 0, MAP - 1)
    cy = jnp.clip(y, 0, MAP - 1)
    return inb & (walls[cx, cy] == 0)
