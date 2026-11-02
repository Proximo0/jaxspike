"""Survival phase: periodic hp regen and eating meat."""
import jax.numpy as jnp

from game.config import ITEM_MEAT, ITEM_NONE, MEAT_HEAL, REGEN_HP, REGEN_TICKS
from game import ro_stats


def survival(world, inputs):
    """Living players regen on schedule and may eat their lowest-index meat; both heals sum before the cap."""
    pl = world["players"]
    alive = pl["active"] & ~pl["dead"]
    regen = alive & (world["tick"] % REGEN_TICKS == 0)

    is_meat = pl["inv"] == ITEM_MEAT
    eats = alive & inputs["eat"] & jnp.any(is_meat, axis=1)
    slot = jnp.argmax(is_meat, axis=1)
    cleared = eats[:, None] & (jnp.arange(pl["inv"].shape[1])[None, :] == slot[:, None])
    inv = jnp.where(cleared, ITEM_NONE, pl["inv"])

    hp = jnp.minimum(pl["hp"] + regen * REGEN_HP + eats * MEAT_HEAL,
                    jnp.maximum(ro_stats.derived(world)["max_hp"], pl["hp"])).astype(pl["hp"].dtype)
    return {**world, "players": {**pl, "hp": hp, "inv": inv}}
