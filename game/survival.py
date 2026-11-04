"""Survival phase: regen (doubled while sitting), sit toggle, eating."""
import jax.numpy as jnp

from game.config import ITEM_MEAT, ITEM_NONE, MAX_MONSTERS, MAX_PLAYERS, MEAT_HEAL
from game import ro_anim, ro_stats

# HP regen is ro_stats.hp_regen (RO: 1 + max_hp // 200 + VIT // 5 every HP_REGEN_TICKS, twice as often while sitting;
# game/data/stats.py). The old config REGEN_HP / REGEN_TICKS are no longer read here.


def survival(world, inputs):
    """Living players regen on schedule and may eat their lowest-index meat; both heals sum before the cap.

    The one-shot input `sit` (bool [MAX_PLAYERS], optional) toggles sitting for living players; the sitting state is
    the anim action SIT (no anim block: sit is ignored). Regen comes from ro_stats.hp_regen with the sitting state
    before this tick's toggle: the tick you sit does not regen faster yet, the tick you stand still does."""
    pl = world["players"]
    alive = pl["active"] & ~pl["dead"]

    if "anim" in world:
        sitting = world["anim"]["action"][:MAX_PLAYERS] == ro_anim.SIT
        toggle = alive & inputs.get("sit", jnp.zeros((MAX_PLAYERS,), jnp.bool_))
        pad = jnp.zeros((MAX_MONSTERS,), jnp.bool_)
        world = ro_anim.set_action(world, jnp.concatenate([toggle & sitting, pad]), ro_anim.IDLE, 0)
        world = ro_anim.set_action(world, jnp.concatenate([toggle & ~sitting, pad]), ro_anim.SIT, 0)
    else:
        sitting = jnp.zeros((MAX_PLAYERS,), jnp.bool_)

    is_meat = pl["inv"] == ITEM_MEAT
    eats = alive & inputs["eat"] & jnp.any(is_meat, axis=1)
    slot = jnp.argmax(is_meat, axis=1)
    cleared = eats[:, None] & (jnp.arange(pl["inv"].shape[1])[None, :] == slot[:, None])
    inv = jnp.where(cleared, ITEM_NONE, pl["inv"])

    hp = jnp.minimum(pl["hp"] + ro_stats.hp_regen(world, sitting) + eats * MEAT_HEAL,
                    jnp.maximum(ro_stats.derived(world)["max_hp"], pl["hp"])).astype(pl["hp"].dtype)
    return {**world, "players": {**pl, "hp": hp, "inv": inv}}
