"""RO parties: create, invite, accept, leave, shared experience. Pure JAX, fixed shapes.

State block `world["party"]` (see make_state). No [N, N] or [PARTIES, N] tables anywhere: ranks use
cumsum/scatter-min/segment_sum, per-party accept ranks use a stable argsort over party ids.

Decisions: a killer's exp is pooled with every other killer's in the same party; the killer with the lowest
slot is the reference for the level gap and radius (it is always eligible and keeps the remainder). Other
killers that are out of range of the reference lose their own gain to the pool (sum is conserved).
A declined or failed accept (party full, inviter gone) clears the pending invite.
"""
import jax
import jax.numpy as jnp

from game.config import MAX_PLAYERS

MAX_PARTIES = 128
PARTY_MAX_MEMBERS = 12
SHARE_LEVEL_RANGE = 15
SHARE_RADIUS = 64
INVITE_TTL = 100

_PIDX = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)


def make_state():
    return {
        "party_id": jnp.full((MAX_PLAYERS,), -1, jnp.int32),
        "leader": jnp.zeros((MAX_PLAYERS,), jnp.bool_),
        "member_count": jnp.zeros((MAX_PARTIES,), jnp.int32),
        "invite_from": jnp.full((MAX_PLAYERS,), -1, jnp.int32),
        "invite_ttl": jnp.zeros((MAX_PLAYERS,), jnp.int32),
    }


def party_of(world):
    return world["party"]["party_id"]


def _base_level(world):
    return world.get("stats", {}).get("base_level", jnp.ones((MAX_PLAYERS,), jnp.int32))


def _seg(pid):
    """Segment ids that send non-members to an out-of-range bucket (dropped by segment ops)."""
    return jnp.where(pid >= 0, pid, MAX_PARTIES)


def _count(pid):
    return jax.ops.segment_sum((pid >= 0).astype(jnp.int32), _seg(pid), num_segments=MAX_PARTIES)


def phase(world, inputs):
    pl, st = world["players"], world["party"]
    active, dead = pl["active"], pl["dead"]
    op = inputs["party"]
    tgt = inputs["target"]
    pid, leader = st["party_id"], st["leader"]
    inv_from, inv_ttl = st["invite_from"], st["invite_ttl"]

    # 1. clean logged-out players
    pid = jnp.where(active, pid, -1)
    leader = leader & active & (pid >= 0)
    inv_from = jnp.where(active, inv_from, -1)

    # 2. invite ttl tick
    inv_ttl = jnp.where(inv_from >= 0, inv_ttl - 1, 0)
    inv_from = jnp.where(inv_ttl > 0, inv_from, -1)
    inv_ttl = jnp.where(inv_from >= 0, inv_ttl, 0)

    # 3. leave
    leaving = active & (op == 2) & (pid >= 0)
    pid = jnp.where(leaving, -1, pid)
    leader = leader & ~leaving
    # Hand leadership to the lowest remaining member of any party that has members but no leader.
    pc = jnp.clip(pid, 0, MAX_PARTIES - 1)
    has_leader = jax.ops.segment_sum(leader.astype(jnp.int32), _seg(pid), num_segments=MAX_PARTIES) > 0
    lowest = jnp.full((MAX_PARTIES,), MAX_PLAYERS, jnp.int32).at[_seg(pid)].min(_PIDX, mode="drop")
    leader = leader | ((pid >= 0) & ~has_leader[pc] & (lowest[pc] == _PIDX))
    count = _count(pid)

    # 4. create: the k-th creator takes the k-th free party id
    creator = active & ~dead & (op == 1) & (pid == -1)
    c_rank = jnp.cumsum(creator) - 1
    free = count == 0
    f_rank = jnp.cumsum(free) - 1
    id_of_rank = jnp.full((MAX_PARTIES,), MAX_PARTIES, jnp.int32).at[
        jnp.where(free, f_rank, MAX_PARTIES)
    ].set(jnp.arange(MAX_PARTIES, dtype=jnp.int32), mode="drop")
    new_id = id_of_rank.at[jnp.where(creator, c_rank, MAX_PARTIES)].get(mode="fill", fill_value=MAX_PARTIES)
    made = creator & (new_id < MAX_PARTIES)
    pid = jnp.where(made, new_id, pid)
    leader = leader | made
    inv_from = jnp.where(pid >= 0, -1, inv_from)
    inv_ttl = jnp.where(inv_from >= 0, inv_ttl, 0)
    count = _count(pid)

    # 5. invite: the lowest leader slot wins each target
    pc = jnp.clip(pid, 0, MAX_PARTIES - 1)
    t = jnp.clip(tgt, 0, MAX_PLAYERS - 1)
    can_invite = (
        active & ~dead & (op == 3) & leader & (pid >= 0) & (count[pc] < PARTY_MAX_MEMBERS)
        & (tgt >= 0) & (tgt < MAX_PLAYERS) & (tgt != _PIDX)
        & active[t] & (pid[t] == -1) & (inv_from[t] == -1)
    )
    winner = jnp.full((MAX_PLAYERS,), MAX_PLAYERS, jnp.int32).at[
        jnp.where(can_invite, t, MAX_PLAYERS)
    ].min(_PIDX, mode="drop")
    invited = winner < MAX_PLAYERS
    inv_from = jnp.where(invited, winner, inv_from)
    inv_ttl = jnp.where(invited, INVITE_TTL, inv_ttl)

    # 6. accept: admit in slot order per party up to the room left
    pending = active & (op == 4) & (inv_from >= 0) & (pid == -1)
    inviter = jnp.clip(inv_from, 0, MAX_PLAYERS - 1)
    ipid = pid[inviter]
    cand = pending & leader[inviter] & (ipid >= 0)
    key = jnp.where(cand, ipid, MAX_PARTIES)
    order = jnp.argsort(key, stable=True)
    key_s = key[order]
    csum = jnp.cumsum(cand[order])
    per_party = jax.ops.segment_sum(cand.astype(jnp.int32), key, num_segments=MAX_PARTIES + 1)
    offset = jnp.cumsum(per_party) - per_party
    rank = jnp.zeros((MAX_PLAYERS,), jnp.int32).at[order].set(csum - 1 - offset[key_s])
    room = PARTY_MAX_MEMBERS - count[jnp.clip(ipid, 0, MAX_PARTIES - 1)]
    joined = cand & (rank < room)
    pid = jnp.where(joined, ipid, pid)
    inv_from = jnp.where(pending, -1, inv_from)  # accepted, rejected or stale: the invite is consumed
    inv_ttl = jnp.where(pending, 0, inv_ttl)

    # 7. recount
    new_st = {"party_id": pid, "leader": leader & (pid >= 0), "member_count": _count(pid),
              "invite_from": inv_from, "invite_ttl": inv_ttl}
    return {**world, "party": new_st}


def share_exp(world, exp_gain):
    """Split each party's pooled kill exp among eligible members; solo players keep their gain."""
    pl = world["players"]
    pid = world["party"]["party_id"]
    gain = exp_gain.astype(jnp.int32)
    in_party = pid >= 0
    killer = in_party & (gain > 0)
    seg = _seg(pid)
    pc = jnp.clip(pid, 0, MAX_PARTIES - 1)

    ref = jnp.full((MAX_PARTIES,), MAX_PLAYERS, jnp.int32).at[
        jnp.where(killer, pid, MAX_PARTIES)
    ].min(_PIDX, mode="drop")
    pool = jax.ops.segment_sum(jnp.where(killer, gain, 0), seg, num_segments=MAX_PARTIES)

    ref_m = ref[pc]  # reference killer slot for each member's party
    has_ref = in_party & (ref_m < MAX_PLAYERS)
    rc = jnp.clip(ref_m, 0, MAX_PLAYERS - 1)
    lv = _base_level(world)
    dist = jnp.abs(pl["pos"][:, 0] - pl["pos"][rc, 0]) + jnp.abs(pl["pos"][:, 1] - pl["pos"][rc, 1])
    is_ref = has_ref & (_PIDX == ref_m)
    near = pl["active"] & ~pl["dead"] & (jnp.abs(lv - lv[rc]) <= SHARE_LEVEL_RANGE) & (dist <= SHARE_RADIUS)
    eligible = has_ref & (is_ref | near)
    n_elig = jax.ops.segment_sum(eligible.astype(jnp.int32), seg, num_segments=MAX_PARTIES)
    n_safe = jnp.maximum(n_elig, 1)
    share = pool // n_safe
    rem = pool - share * n_safe
    out = jnp.where(eligible, share[pc], 0) + jnp.where(is_ref, rem[pc], 0)
    return jnp.where(in_party, out, gain).astype(jnp.int32)
