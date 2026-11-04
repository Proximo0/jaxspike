"""Combined bot policy for the RO world: hunting + gathering moves, navigation modes, skills, items, stats, parties.

Each team's policy is pure and read-only on the world; this composes them into the full inputs dict that
game.step.step expects (see game.state.empty_inputs). `target` is shared by skills (entity id) and party invites
(player slot): the party value wins on a tick with an invite. Moves: the craft policy (base hunting + gatherers),
then bots.ro_nav_policy's moves for every slot: hunting (wall- and crowd-aware chase, attack), retreat/rest by
max-HP fraction, zone-aware roaming, dungeon diving, party following. A gatherer keeps the craft policy's move,
gather and craft only while that policy reports it busy (craft_busy) and no monster is in reach: it hunts between gathers.
Optional engine inputs ("sit", "buy", "sell") are emitted only when game.state.empty_inputs() has the key.
"""
import jax
import jax.numpy as jnp

from bots.ro_craft_policy import craft_bot_inputs
from bots.ro_items_policy import engine_inputs, item_inputs, wants_potions
from bots.ro_nav_policy import nav_moves
from bots.ro_party_policy import party_inputs
from bots.ro_skills_policy import cast_inputs, learn_input
from bots.ro_stats_policy import bot_allot, bot_job


def ro_bot_inputs(world, key):
    """Return the full inputs dict for every slot (bots fill all; the server overrides human slots)."""
    k_craft, k_nav, k_item, k_stat, k_party = jax.random.split(key, 5)
    inp = craft_bot_inputs(world, k_craft)          # base hunting policy + gather/craft, move overridden for gatherers
    shop = wants_potions(world) if "buy" in engine_inputs() else None
    nav, sit, attack, fight = nav_moves(world, k_nav, inp["move"], shop)
    busy = inp.pop("craft_busy") & ~fight     # a gatherer keeps its craft move unless a monster is in reach (bots3)
    move = jnp.where(busy, inp["move"], nav).astype(jnp.int32)
    inp = {**inp, "gather": inp["gather"] & busy, "craft": jnp.where(busy, inp["craft"], 0).astype(jnp.int32)}
    skill, target = cast_inputs(world)
    pp = party_inputs(world, k_party)
    inp = {
        **inp,
        "move": move,
        "attack": attack,
        "learn": learn_input(world),
        "skill": skill,
        "target": jnp.where(pp["party"] == 3, pp["target"], target).astype(jnp.int32),
        **item_inputs(world, k_item),
        "allot": bot_allot(world, k_stat),
        "job": bot_job(world, k_stat),
        "party": pp["party"],
        "save": pp["save"],
    }
    if "sit" in engine_inputs():
        inp["sit"] = sit
    if "teleport" in engine_inputs():  # kafra teleport: bots never buy one (yet); -1 = none
        inp["teleport"] = jnp.full_like(inp["allot"], -1)
    return inp
