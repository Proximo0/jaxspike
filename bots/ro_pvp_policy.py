"""Duelling bots (ro4-pvp): a few Solmera bots walk into the PvP arena and fight each other there. Pure, read-only.

Duellists are the slots with slot % DUEL_EVERY == 0 (home town slot % 5 == 0 = Solmera, never a gatherer: gatherers
are slot % 4 == 1). Every tick a living duellist:
- outside the arena walks greedily (8 directions, axis steps when the diagonal is walled) to the staging tile just
  north of the arena gate, then south through the gate;
- inside the arena steps toward the nearest other living player on the arena floor and holds the attack input (the
  engine's PvP path swings when that player is in weapon range); with nobody else inside it waits in the middle.
PvP deaths cost no exp; a dead duellist respawns on its temple and walks back. Other slots are left untouched.

`pvp_overrides(world, inputs)` returns the inputs with move, attack (and sit, when present) replaced on duellist slots;
compose it last in bots.ro_policies.ro_bot_inputs.
"""
import jax.numpy as jnp

from bots.ro_nav_policy import DELTAS, blocked
from game.config import MAX_PLAYERS
from game.world_layout import arena_box, arena_gate, in_arena

DUEL_EVERY = 10  # slot % DUEL_EVERY == 0 duels (13 of the 128 soak bots per world)
_SLOT = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
IS_DUELLIST = _SLOT % DUEL_EVERY == 0
_GATE = jnp.asarray(arena_gate()[0], jnp.int32)          # west gate tile
_STAGE = _GATE + jnp.asarray([0, -1], jnp.int32)         # floor tile just outside (north of) the gate
_BOX = arena_box()
_CENTRE = jnp.asarray([(_BOX[0] + _BOX[2]) // 2, (_BOX[1] + _BOX[3]) // 2], jnp.int32)


def _code_of(delta):
    """int32 [N]: the move code (DELTAS index) of each unit delta [N, 2] (0 = stay)."""
    return jnp.argmax(jnp.all(DELTAS[None, :, :] == delta[:, None, :], axis=-1), axis=1).astype(jnp.int32)


def _step_toward(world, pos, goal, axis_first):
    """int32 [N] move codes: diagonal toward goal, else the x or y axis step, whichever is not walled (else stay).
    Where axis_first, the axis steps come first (closing in on a foe: two bots a diagonal apart would otherwise
    both try each other's tile forever)."""
    s = jnp.sign(goal - pos).astype(jnp.int32)
    sx, sy = s * jnp.asarray([1, 0]), s * jnp.asarray([0, 1])
    cand = jnp.where(axis_first[:, None, None], jnp.stack([sx, sy, s], axis=1), jnp.stack([s, sx, sy], axis=1))
    ok = ~blocked(world, pos, cand) & jnp.any(cand != 0, axis=-1)
    pick = jnp.take_along_axis(cand, jnp.argmax(ok, axis=1)[:, None, None], axis=1)[:, 0]
    return jnp.where(jnp.any(ok, axis=1), _code_of(pick), 0)


def duel_moves(world):
    """(move int32 [P], attack bool [P], mask bool [P]): the duellists' inputs and which slots they replace."""
    p = world["players"]
    pos = p["pos"]
    alive = p["active"] & ~p["dead"]
    inside = in_arena(pos) & alive
    # nearest other living player on the arena floor (manhattan; [P, P] over the few arena occupants is masked)
    d = jnp.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    other = inside[None, :] & (_SLOT[:, None] != _SLOT[None, :])
    d = jnp.where(other, d, 1 << 30)
    near = jnp.argmin(d, axis=1)
    has = jnp.any(other, axis=1)
    foe = jnp.where(has[:, None], pos[near], _CENTRE[None, :])
    adjacent = has & (jnp.min(d, axis=1) <= 1)

    at_stage = jnp.all(pos == _STAGE[None, :], axis=1) | jnp.all(pos == _GATE[None, :], axis=1)
    goal = jnp.where(inside[:, None], foe, jnp.where(at_stage[:, None], _GATE[None, :] + jnp.asarray([0, 1]),
                                                     _STAGE[None, :]))
    move = jnp.where(inside & adjacent, 0, _step_toward(world, pos, goal, inside))
    mask = IS_DUELLIST & p["active"]
    return jnp.where(alive, move, 0).astype(jnp.int32), inside, mask


def pvp_overrides(world, inputs):
    """inputs with the duellists' move / attack (and sit = False) in place of the other policies' values."""
    move, attack, mask = duel_moves(world)
    out = {**inputs, "move": jnp.where(mask, move, inputs["move"]).astype(jnp.int32),
           "attack": jnp.where(mask, attack, inputs["attack"])}
    if "sit" in inputs:
        out["sit"] = jnp.where(mask, False, inputs["sit"])
    return out
