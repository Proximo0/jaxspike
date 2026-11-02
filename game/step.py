"""Full tick. Owned by the lead. Phase order is fixed here and documented in docs/brief.md.

Phases: movement, monsters, combat, loot, survival (regen + eating), then the tick increment.
Each phase is a pure function world -> world (or world, inputs -> world).
"""
import jax
import jax.numpy as jnp

from game import combat, loot, monsters, movement, survival


def step(world, inputs):
    # `died` only means "died during this tick", so clear it before the phases run.
    world = {**world, "monsters": {**world["monsters"], "died": jnp.zeros_like(world["monsters"]["died"])}}
    world = movement.move_players(world, inputs)
    world = monsters.monster_ai(world)
    world = combat.combat(world, inputs)
    world = loot.loot(world)
    world = survival.survival(world, inputs)
    return {**world, "tick": world["tick"] + 1}


step_jit = jax.jit(step)
