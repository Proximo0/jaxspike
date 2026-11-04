"""Card-drop announcements (round 4, ro4-cards; docs/cards_wire.md).

`card_events` runs inside the server's jitted tick: a fixed-shape compare of the drops block before and after
`step` finds card piles that appeared this tick on the tile of a monster that died this tick (so a player's own
`drop` of a card is never announced), and names the finder. The host reads the few fixed-size rows in its one pull
and broadcasts one `card` message per row (`card_messages`).
"""
import jax.numpy as jnp

from game.config import MAX_DROPS, MAX_MONSTERS
from game.data import items as item_data
from net import protocol

MAX_CARD_EVENTS = 8   # card piles announced per tick at most (1/10000 drops: more than one per tick is rare)
DIED_SCAN = 64        # monsters that died this tick matched against the new piles (the rest are ignored)
FIND_RADIUS = 9       # fallback finder: the nearest living player within this many tiles (bow range)
_BIG = 1 << 20


def card_events(pre_drops, post_drops, monsters, pre_top, players):
    """New card drops of this tick as fixed-shape rows. Pure, works under jit and vmap.

    pre_drops / post_drops: world["drops"] before and after step. monsters: world["monsters"] after step (its `died`
    flags this tick's deaths and their positions are the drop tiles). pre_top: world["mvp"]["top_damager"] before
    step (the step's mvp phase clears it for the dead). players: world["players"] after step.

    Returns {"valid": bool [K], "item": int32 [K], "kind": int32 [K] monster kind, "finder": int32 [K] player slot
    or -1}, K = MAX_CARD_EVENTS. The finder is the dead monster's top damager before the killing tick when that
    player is still active, else the nearest living player within FIND_RADIUS (lowest slot on ties), else -1.
    """
    item = post_drops["item"]
    is_card = post_drops["active"] & (item_data.KIND[item] == item_data.KIND_CARD)
    fresh = is_card & (~pre_drops["active"] | (pre_drops["item"] != item) | (post_drops["ttl"] >= pre_drops["ttl"]))
    (idx,) = jnp.nonzero(fresh, size=MAX_CARD_EVENTS, fill_value=MAX_DROPS)
    have = idx < MAX_DROPS
    ci = jnp.clip(idx, 0, MAX_DROPS - 1)
    pos = post_drops["pos"][ci]                                                  # [K, 2]

    (didx,) = jnp.nonzero(monsters["died"], size=DIED_SCAN, fill_value=MAX_MONSTERS)
    dvalid = didx < MAX_MONSTERS
    dc = jnp.clip(didx, 0, MAX_MONSTERS - 1)
    dpos = monsters["pos"][dc]                                                   # [D, 2]
    match = jnp.all(pos[:, None, :] == dpos[None, :, :], axis=-1) & dvalid[None, :]  # [K, D]
    from_kill = match.any(axis=1)
    mon = dc[jnp.argmax(match, axis=1)]
    kind = monsters["kind"][mon]

    n_p = players["pos"].shape[0]
    top = pre_top[mon]
    top_ok = (top >= 0) & players["active"][jnp.clip(top, 0, n_p - 1)]
    alive = players["active"] & ~players["dead"]
    cheb = jnp.max(jnp.abs(pos[:, None, :] - players["pos"][None, :, :]), axis=-1)  # [K, P]
    dist = jnp.where(alive[None, :], cheb, _BIG)
    near = jnp.argmin(dist, axis=1).astype(jnp.int32)
    near_ok = jnp.min(dist, axis=1) <= FIND_RADIUS
    finder = jnp.where(top_ok, top, jnp.where(near_ok, near, -1)).astype(jnp.int32)

    valid = have & from_kill
    return {"valid": valid, "item": jnp.where(valid, item[ci], 0).astype(jnp.int32),
            "kind": jnp.where(valid, kind, -1).astype(jnp.int32), "finder": jnp.where(valid, finder, -1)}


def card_messages(ev: dict, names: dict, tick: int) -> list:
    """Host side: one `card` message per valid row of a pulled `card_events` result (numpy). `names` maps logged-in
    human slots to names; bots and unknown finders are announced as 'Someone' (finder null on the wire)."""
    out = []
    for k in range(len(ev["valid"])):
        if not bool(ev["valid"][k]):
            continue
        item, kind, slot = int(ev["item"][k]), int(ev["kind"][k]), int(ev["finder"][k])
        who = names.get(slot) if slot >= 0 else None
        out.append(protocol.card_message(item, kind, who, tick))
    return out
