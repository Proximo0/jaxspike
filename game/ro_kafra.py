"""Kafra save point: a living player in a town who asks to save gets that town's temple centre as respawn."""
import jax.numpy as jnp

from game.world_layout import town_boxes, town_centers


def make_state():
    return {}


def town_of_tile(pos):
    """int32 [N, 2] -> int32 [N]: index of the town whose box holds the tile, -1 outside."""
    boxes = town_boxes()
    out = jnp.full(pos.shape[0], -1, jnp.int32)
    for t in range(boxes.shape[0]):  # loop over a constant (5 towns)
        x0, y0, x1, y1 = (int(v) for v in boxes[t])
        inside = (pos[:, 0] >= x0) & (pos[:, 0] <= x1) & (pos[:, 1] >= y0) & (pos[:, 1] <= y1)
        out = jnp.where(inside & (out < 0), t, out)
    return out


def phase(world, inputs):
    p = world["players"]
    town = town_of_tile(p["pos"])
    ok = inputs["save"] & p["active"] & ~p["dead"] & (town >= 0)
    centre = jnp.asarray(town_centers(), jnp.int32)[jnp.maximum(town, 0)]
    spawn = jnp.where(ok[:, None], centre, p["spawn"])
    return {**world, "players": {**p, "spawn": spawn}}
