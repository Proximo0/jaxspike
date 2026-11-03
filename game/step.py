"""Full tick. Owned by the lead. Phase order is fixed here and documented in docs/brief.md and docs/ro_contract.md.

Phases: movement, warps, monsters, skills (casts resolve, buffs tick), combat, loot, stats (exp, levels),
items, shops, survival (regen + eating), craft, party, mvp, kafra, anim, then the tick increment.
Each phase is a pure function world -> world (or world, inputs -> world).
"""
import jax
import jax.numpy as jnp

from game import (combat, loot, monsters, movement, ro_anim, ro_craft, ro_items, ro_kafra, ro_mvp, ro_party,
                  ro_shop, ro_skills, ro_stats, ro_warps, survival)


def step(world, inputs):
    # `died` only means "died during this tick", so clear it before the phases run.
    world = {**world, "monsters": {**world["monsters"], "died": jnp.zeros_like(world["monsters"]["died"])}}
    world = movement.move_players(world, inputs)
    world = ro_warps.apply_warps(world)
    world = monsters.monster_ai(world)
    world = ro_skills.phase(world, inputs)
    world = combat.combat(world, inputs)
    world = loot.loot(world)
    world = ro_stats.phase(world, inputs)
    world = ro_items.phase(world, inputs)
    world = ro_shop.phase(world, inputs)
    world = survival.survival(world, inputs)
    world = ro_craft.phase(world, inputs)
    world = ro_party.phase(world, inputs)
    world = ro_mvp.phase(world)
    world = ro_kafra.phase(world, inputs)
    world = ro_anim.phase(world)  # last: derives walk/standby/idle from what the phases above did
    return {**world, "tick": world["tick"] + 1}


step_jit = jax.jit(step)
