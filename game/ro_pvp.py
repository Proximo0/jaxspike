"""PvP (ro4-pvp): per-player PK flag, PvP kill/death counters, the toggle phase and the area rules. Pure JAX.

State block world["pvp"] (make_state), int32/bool [MAX_PLAYERS]:
  pk          the player's PK flag (toggled by the one-shot `pvp` input; cleared on an inactive slot)
  pvp_kills   players this slot killed in PvP
  pvp_deaths  times this slot was killed in PvP (these deaths are NOT counted in players.deaths, so the
              ro_stats death penalty never fires for them: PvP deaths cost no exp; respawn is the usual one)

Where PvP is allowed (world_layout): the arena floor (`in_arena`) always, for everyone in it; elsewhere only in a
zone whose `pvp_zone` flag is on AND with both players PK-enabled (towns and dungeons never; fields off in v1).
`pair_ok` is the rule for one attacker/target pair: both in the arena, or both outside it in PvP zones with PK on.

The damage itself is game/combat.py's PvP path (same RO formulas as players vs monsters). It needs at most
PVP_SLOTS PvP-capable players per world at once: combat compacts them into a fixed [PVP_SLOTS] list and matches
them pairwise ([PVP_SLOTS, PVP_SLOTS]); players beyond PVP_SLOTS (lowest slots first) skip PvP that tick.

phase(world, inputs) runs after combat: the `pvp` input (bool [MAX_PLAYERS], one-shot) flips the PK flag; turning it
ON is refused while the player stands in a town (outside the arena) or is dead. Without the input key the phase only
clears the flag of inactive slots.
"""
import jax.numpy as jnp

from game import world_layout as wl
from game.config import MAX_PLAYERS

PVP_SLOTS = 64  # fixed PvP candidate list per world (see the module docstring)
_PVP_ZONE = jnp.asarray(wl.pvp_zone())                                         # bool [N_ZONES]
_TOWN_ZONE = jnp.asarray([z[1] == wl.ZONE_TOWN for z in wl.ZONES], jnp.bool_)  # bool [N_ZONES]


def make_state():
    return {
        "pk": jnp.zeros((MAX_PLAYERS,), jnp.bool_),
        "pvp_kills": jnp.zeros((MAX_PLAYERS,), jnp.int32),
        "pvp_deaths": jnp.zeros((MAX_PLAYERS,), jnp.int32),
    }


def areas(world):
    """(arena bool [MAX_PLAYERS], field bool [MAX_PLAYERS]): on the arena floor; in a PvP zone with PK on (outside
    the arena). Only living, active players count."""
    p = world["players"]
    alive = p["active"] & ~p["dead"]
    arena = wl.in_arena(p["pos"]) & alive
    pk = world["pvp"]["pk"] if "pvp" in world else jnp.zeros_like(alive)
    field = _PVP_ZONE[wl.zone_of(p["pos"])] & pk & alive & ~arena
    return arena, field


def pair_ok(arena_a, field_a, arena_b, field_b):
    """bool, broadcasting: attacker a may hit target b (both in the arena, or both in PvP zones with PK on)."""
    return (arena_a & arena_b) | (field_a & field_b)


def phase(world, inputs):
    """Apply the PK toggle (one-shot `pvp` input) and clear the flag of inactive slots. No-op without the block."""
    if "pvp" not in world:
        return world
    p, v = world["players"], world["pvp"]
    pk = v["pk"]
    toggle = inputs.get("pvp", None)
    if toggle is not None:
        in_town = _TOWN_ZONE[wl.zone_of(p["pos"])] & ~wl.in_arena(p["pos"])
        may_on = p["active"] & ~p["dead"] & ~in_town
        flip = toggle.astype(jnp.bool_) & p["active"] & (pk | may_on)  # turning off always works, on only outside towns
        pk = pk ^ flip
    return {**world, "pvp": {**v, "pk": pk & p["active"]}}
