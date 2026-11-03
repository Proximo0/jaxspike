"""Monster AI phase: respawn countdown, RO monster modes, then the step.

The decision (chase, give up and return home, random walk, assist, looting, immovable kinds) is
game/ro_monster_modes.py and only writes move.path_dir; game/ro_move.py steps it at the kind's speed (ms per cell)."""
import jax.numpy as jnp

from game import ro_monster_modes, ro_move
from game.config import MONSTER_HP

_HP_TABLE = jnp.array(MONSTER_HP, dtype=jnp.int32)


def _respawn(m, occ):
    """Count down dead monsters and reset the ones that reach 0 onto a free spawn tile.

    A monster whose spawn tile is occupied waits there (countdown held at 1) and retries next tick.
    """
    dead = ~m["active"] & (m["respawn_in"] > 0)
    left = jnp.where(dead, m["respawn_in"] - 1, m["respawn_in"])
    spawn_blocked = occ[m["spawn"][:, 0], m["spawn"][:, 1]]
    left = jnp.where(dead & (left == 0) & spawn_blocked, 1, left)
    revive = dead & (left == 0) & ~spawn_blocked
    return {
        **m,
        "respawn_in": left,
        "active": m["active"] | revive,
        "hp": jnp.where(revive, _HP_TABLE[m["kind"]], m["hp"]),
        "pos": jnp.where(revive[:, None], m["spawn"], m["pos"]),
        "attack_cd": jnp.where(revive, 0, m["attack_cd"]),
    }


def _occupancy(world):
    """Bool [MAP, MAP]: tiles holding a wall, a protection-zone tile, a living player or an active monster. Dead players are not physical."""
    p, m = world["players"], world["monsters"]
    occ = (world["walls"] != 0) | world["pz"]
    occ = occ.at[p["pos"][:, 0], p["pos"][:, 1]].max(p["active"] & ~p["dead"])
    return occ.at[m["pos"][:, 0], m["pos"][:, 1]].max(m["active"])


def monster_ai(world):
    """Run respawn countdown, decide each monster's step, then move them through the speed accumulators."""
    world, legacy = ro_move.attach(world)  # no-op on worlds that carry the block
    world = ro_monster_modes.on_deaths(world)  # per-kind respawn timers, looted items fall where a monster died
    movers = world["monsters"]["active"]  # monsters respawning this tick do not move yet
    m = _respawn(world["monsters"], _occupancy(world))
    world = {**world, "monsters": m}  # occupancy includes monsters revived this tick
    world, path_dir = ro_monster_modes.decide(world, movers, _occupancy(world))
    # ro-move owns the step application below: the decision above only writes move.path_dir.
    world = ro_move.want_monsters(world, path_dir)
    world = ro_move.step_entities(world, "monsters")
    return ro_move.detach(world, legacy)
