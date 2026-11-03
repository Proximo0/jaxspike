"""Player movement phase: 8 directions with speed accumulators (game/ro_move.py).

inputs["move"] is a move code 0..8 (0 = no new intent, 1..8 = S, SW, W, NW, N, NE, E, SE). A code sets the
player's path_dir; 0 keeps a pending one, so a one-tick keyboard tap still makes its step when the accumulator
allows. Occupancy and tile claims are resolved by sorting tile keys, not by scattering into a MAP x MAP grid
(batched scatters into full-map arrays blew up GPU memory under vmap).
"""
import jax.numpy as jnp

from game import ro_move
from game.ro_move import _occupied  # noqa: F401  re-exported: game/ro_warps.py imports it from here


def _inverse_permutation(order):
    """Inverse of a permutation, computed with a sort instead of a scatter (scatters with batched indices expand badly under vmap)."""
    return jnp.argsort(order)


def move_players(world, inputs):
    """Set players' intents from inputs["move"] and step the players whose accumulator allows; lowest slot wins ties."""
    world, legacy = ro_move.attach(world)  # no-op on state.make_empty_world worlds, which carry the block
    world = ro_move.want_players(world, inputs["move"])
    world = ro_move.step_entities(world, "players")
    return ro_move.detach(world, legacy)
