"""Bot party policy: parties per home town (a new one when the town's parties are full). Pure, read-only."""
import jax.numpy as jnp

from game.config import MAX_PLAYERS
from game.ro_party import MAX_PARTIES
from game.world_layout import TOWNS

NT = len(TOWNS)
_PIDX = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
_TOWN = _PIDX % NT
PARTY_SIZE = 4  # bots form parties of 2 to PARTY_SIZE
GATHER_EVERY = 4  # same rule as bots/ro_craft_policy.py (kept local: no import cycle)
_IS_GATHERER = _PIDX % GATHER_EVERY == 1  # gatherers stay solo: they never create, invite or accept


def leader_of(world):
    """Per slot: (lpos int32 [N, 2], has bool [N], lslot int32 [N]) of its party's leader.

    `has` is True only for a living active partied non-leader whose leader is active and alive. Leader per party is
    the lowest leader-flagged slot, else the lowest member slot. lpos/lslot are arbitrary where `has` is False.
    """
    pl, st = world["players"], world["party"]
    pid = st["party_id"]
    in_party = pid >= 0
    seg = jnp.where(in_party, pid, MAX_PARTIES)
    flagged = jnp.full((MAX_PARTIES,), MAX_PLAYERS, jnp.int32).at[
        jnp.where(st["leader"] & in_party, seg, MAX_PARTIES)].min(_PIDX, mode="drop")
    lowest = jnp.full((MAX_PARTIES,), MAX_PLAYERS, jnp.int32).at[seg].min(_PIDX, mode="drop")
    table = jnp.where(flagged < MAX_PLAYERS, flagged, lowest)
    lslot = jnp.clip(table[jnp.clip(pid, 0, MAX_PARTIES - 1)], 0, MAX_PLAYERS - 1).astype(jnp.int32)
    has = (pl["active"] & ~pl["dead"] & in_party & (lslot != _PIDX)
           & pl["active"][lslot] & ~pl["dead"][lslot] & (table[jnp.clip(pid, 0, MAX_PARTIES - 1)] < MAX_PLAYERS))
    return pl["pos"][lslot].astype(jnp.int32), has, lslot


def party_inputs(world, key):
    """Return {"party": int32, "target": int32, "save": bool}, each [MAX_PLAYERS]."""
    pl, st = world["players"], world["party"]
    pid, inv_from = st["party_id"], st["invite_from"]
    live = pl["active"] & ~pl["dead"]
    in_party = pid >= 0

    # The lowest living partyless uninvited slot of a town creates, only while no party of the town has room
    # (a town with more bots than PARTY_SIZE gets another party once the others are full).
    room_pl = in_party & (st["member_count"][jnp.clip(pid, 0, MAX_PARTIES - 1)] < PARTY_SIZE)
    town_has_party = jnp.zeros((NT,), jnp.bool_).at[_TOWN].max(pl["active"] & room_pl)
    free = live & ~in_party & (inv_from == -1) & ~_IS_GATHERER
    low_free = jnp.full((NT,), MAX_PLAYERS, jnp.int32).at[_TOWN].min(jnp.where(free, _PIDX, MAX_PLAYERS))
    create = free & (_PIDX == low_free[_TOWN]) & ~town_has_party[_TOWN]

    # Leaders invite the lowest same-town active partyless slot with no pending invite.
    cand = pl["active"] & ~in_party & (inv_from == -1) & ~_IS_GATHERER
    low_cand = jnp.full((NT,), MAX_PLAYERS, jnp.int32).at[_TOWN].min(jnp.where(cand, _PIDX, MAX_PLAYERS))
    tgt = low_cand[_TOWN]
    # Pending invites count against the room so a party never overshoots PARTY_SIZE when they are accepted together.
    inviter_pid = jnp.where(inv_from >= 0, pid[jnp.clip(inv_from, 0, MAX_PLAYERS - 1)], -1)
    n_pending = jnp.zeros((MAX_PARTIES,), jnp.int32).at[jnp.where(inviter_pid >= 0, inviter_pid, MAX_PARTIES)].add(
        1, mode="drop")
    pc = jnp.clip(pid, 0, MAX_PARTIES - 1)
    room = (st["member_count"][pc] + n_pending[pc]) < PARTY_SIZE
    invite = live & st["leader"] & in_party & room & (tgt < MAX_PLAYERS)

    accept = pl["active"] & (inv_from >= 0) & ~_IS_GATHERER
    party = jnp.where(accept, 4, jnp.where(invite, 3, jnp.where(create, 1, 0))).astype(jnp.int32)
    target = jnp.where(invite, tgt, -1).astype(jnp.int32)
    return {"party": party, "target": target, "save": jnp.zeros((MAX_PLAYERS,), jnp.bool_)}
