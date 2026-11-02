"""Bot party policy: one party per home town. Pure, read-only on the world."""
import jax.numpy as jnp

from game.config import MAX_PLAYERS
from game.ro_party import MAX_PARTIES, PARTY_MAX_MEMBERS
from game.world_layout import TOWNS

NT = len(TOWNS)
_PIDX = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
_TOWN = _PIDX % NT


def party_inputs(world, key):
    """Return {"party": int32, "target": int32, "save": bool}, each [MAX_PLAYERS]."""
    pl, st = world["players"], world["party"]
    pid, inv_from = st["party_id"], st["invite_from"]
    live = pl["active"] & ~pl["dead"]
    in_party = pid >= 0

    # The lowest living partyless uninvited slot of a town creates, only while the town has no party.
    town_has_party = jnp.zeros((NT,), jnp.bool_).at[_TOWN].max(pl["active"] & in_party)
    free = live & ~in_party & (inv_from == -1)
    low_free = jnp.full((NT,), MAX_PLAYERS, jnp.int32).at[_TOWN].min(jnp.where(free, _PIDX, MAX_PLAYERS))
    create = free & (_PIDX == low_free[_TOWN]) & ~town_has_party[_TOWN]

    # Leaders invite the lowest same-town active partyless slot with no pending invite.
    cand = pl["active"] & ~in_party & (inv_from == -1)
    low_cand = jnp.full((NT,), MAX_PLAYERS, jnp.int32).at[_TOWN].min(jnp.where(cand, _PIDX, MAX_PLAYERS))
    room = st["member_count"][jnp.clip(pid, 0, MAX_PARTIES - 1)] < PARTY_MAX_MEMBERS
    tgt = low_cand[_TOWN]
    invite = live & st["leader"] & in_party & room & (tgt < MAX_PLAYERS)

    accept = pl["active"] & (inv_from >= 0)
    party = jnp.where(accept, 4, jnp.where(invite, 3, jnp.where(create, 1, 0))).astype(jnp.int32)
    target = jnp.where(invite, tgt, -1).astype(jnp.int32)
    return {"party": party, "target": target, "save": jnp.zeros((MAX_PLAYERS,), jnp.bool_)}
