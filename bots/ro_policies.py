"""Combined bot policy for the RO world: hunting + gathering moves, skills, items, stat allotment, parties.

Owned by the lead. Each team's policy is pure and read-only on the world; this composes them into the full
inputs dict that game.step.step expects (see game.state.empty_inputs). `target` is shared by skills (entity id)
and party invites (player slot): the party value wins on a tick with an invite.
"""
import jax
import jax.numpy as jnp

from bots.ro_craft_policy import craft_bot_inputs
from bots.ro_items_policy import item_inputs
from bots.ro_party_policy import party_inputs
from bots.ro_skills_policy import bot_inputs as skills_bot_inputs
from bots.ro_stats_policy import bot_allot, bot_job


def ro_bot_inputs(world, key):
    """Return the full inputs dict for every slot (bots fill all; the server overrides human slots)."""
    k_craft, k_skill, k_item, k_stat, k_party = jax.random.split(key, 5)
    inp = craft_bot_inputs(world, k_craft)          # base hunting policy + gather/craft, move overridden for gatherers
    sk = skills_bot_inputs(world, k_skill)
    pp = party_inputs(world, k_party)
    inp = {
        **inp,
        "learn": sk["learn"],
        "skill": sk["skill"],
        "target": jnp.where(pp["party"] == 3, pp["target"], sk["target"]).astype(jnp.int32),
        **item_inputs(world, k_item),
        "allot": bot_allot(world, k_stat),
        "job": bot_job(world, k_stat),
        "party": pp["party"],
        "save": pp["save"],
    }
    return inp
