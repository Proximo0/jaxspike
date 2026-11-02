"""MVP bosses: damage tracking, kill event ring, top-damager drops, in-place respawn.

Pure JAX, fixed shapes, int32. All state lives in world["mvp"] (see make_state). MVP identity comes from the
`MVP` flag of game/data/monsters.py (Golden Thief Bug 18, Eddga 19); state is indexed by monster kind
(`[N_KINDS]`), and the event ring stores the monster kind. MVPs are placed by `ro_spawns`/`spawn_population`
(each MVP kind once, in its dungeon), so the slot's own `spawn` is the respawn tile; there is no placement step.

Approximation (v1): RO ranks attackers by cumulative damage. Keeping a per-attacker
cumulative table needs a [MAX_MONSTERS, K] ledger, so v1 keeps `damage_ledger` as the
total damage per monster and `top_damage`/`top_damager` as the best SINGLE tick hit
(lowest attacker slot wins ties). A later worker can swap in a cumulative ledger.

Respawn: `phase` runs after combat each tick and holds `monsters.respawn_in[mvp_slot] = 0`
while `next_spawn[kind] > 0`, so the generic monster respawn never fires for MVP slots; the MVP
is revived in place (its `spawn` tile) at full table hp when `next_spawn` reaches 0.
"""
import jax.numpy as jnp
import numpy as np

from game.config import MAX_MONSTERS, MAX_PLAYERS
from game.data import items as _items
from game.data import monsters as _mon
from game.data.items import ITEM_NONE

N_KINDS = _mon.N_KINDS
N_MVP_KINDS = N_KINDS  # kept for callers; MVP state is indexed by monster kind
MVP_KIND = tuple(bool(m.mvp) for m in _mon.MONSTERS)
MVP_SPAWN_TICKS = tuple(3000 if f else 0 for f in MVP_KIND)
MVP_HP = tuple(m.hp if m.mvp else 0 for m in _mon.MONSTERS)
_DROPS = {
    "Golden Thief Bug": (_items.ITEM_THIEF_BUG_CARD, _items.ITEM_ELUNIUM, _items.ITEM_GOLD_COIN, _items.ITEM_GOLD_COIN),
    "Eddga": (_items.ITEM_ORC_WARRIOR_CARD, _items.ITEM_ORIDECON, _items.ITEM_GOLD_COIN, _items.ITEM_GOLD_COIN),
}
MVP_DROP = tuple(_DROPS.get(m.name, (0,) * 4) for m in _mon.MONSTERS)  # [N_KINDS, 4], padded with ITEM_NONE
EVENT_RING = 8

_KIND = np.array(MVP_KIND, bool)
_TICKS = np.array(MVP_SPAWN_TICKS, np.int32)
_DROP = np.array(MVP_DROP, np.int32)
_BIG = np.int32(2 ** 30)


def make_state():
    return {
        "top_damager": jnp.full((MAX_MONSTERS,), -1, jnp.int32),
        "top_damage": jnp.zeros((MAX_MONSTERS,), jnp.int32),
        "damage_ledger": jnp.zeros((MAX_MONSTERS,), jnp.int32),
        "next_spawn": jnp.zeros((N_KINDS,), jnp.int32),
        "mvp_slot": jnp.full((N_KINDS,), -1, jnp.int32),
        "event": jnp.full((EVENT_RING, 3), -1, jnp.int32),
        "event_head": jnp.zeros((), jnp.int32),
    }


def _mvp_of_kind(kind):
    in_range = (kind >= 0) & (kind < N_KINDS)
    return in_range & jnp.asarray(_KIND)[jnp.clip(kind, 0, N_KINDS - 1)]


def is_mvp(world):
    return _mvp_of_kind(world["monsters"]["kind"])


def record_damage(world, attacker_idx, monster_idx, dmg):
    mvp = world["mvp"]
    attacker_idx = attacker_idx.astype(jnp.int32)
    dmg = dmg.astype(jnp.int32)
    valid = (dmg > 0) & (monster_idx >= 0) & (monster_idx < MAX_MONSTERS) & (attacker_idx >= 0)
    m = jnp.where(valid, monster_idx, MAX_MONSTERS).astype(jnp.int32)
    ms = jnp.where(valid, monster_idx, 0)
    ledger = mvp["damage_ledger"].at[m].add(dmg, mode="drop")
    old_top = mvp["top_damage"]
    new_top = old_top.at[m].max(dmg, mode="drop")
    win = valid & (dmg == new_top[ms])
    cand = jnp.full((MAX_MONSTERS,), _BIG, jnp.int32).at[jnp.where(win, m, MAX_MONSTERS)].min(
        attacker_idx, mode="drop")
    old_who = jnp.where((old_top == new_top) & (mvp["top_damager"] >= 0), mvp["top_damager"], _BIG)
    best = jnp.minimum(cand, old_who)
    who = jnp.where(best < _BIG, best, mvp["top_damager"])
    return {**world, "mvp": {**mvp, "damage_ledger": ledger, "top_damage": new_top, "top_damager": who}}


def phase(world):
    mons, mvp, players = world["monsters"], world["mvp"], world["players"]
    kind = mons["kind"]
    kc = jnp.clip(kind, 0, N_KINDS - 1)
    mvp_died = mons["died"] & _mvp_of_kind(kind)
    any_died = mvp_died.any()
    first = jnp.argmax(mvp_died)
    k0 = kc[first]
    killer = mvp["top_damager"][first]

    # Respawn countdown (uses state from before this tick's deaths).
    live = _mvp_of_kind(kind)
    hit = live[None] & (kc[None] == jnp.arange(N_KINDS)[:, None])  # [N_KINDS, MAX_MONSTERS]
    slot = jnp.where(hit.any(-1), jnp.argmax(hit, -1), -1).astype(jnp.int32)
    sl = jnp.clip(slot, 0, MAX_MONSTERS - 1)
    waiting = (slot >= 0) & ~mons["active"][sl] & (mvp["next_spawn"] > 0)
    nxt = jnp.where(waiting, mvp["next_spawn"] - 1, mvp["next_spawn"])
    # A blocked spawn tile (wall, active monster or living player) holds the timer at 1 and retries next tick,
    # like monsters._respawn.
    tile = mons["spawn"][sl]
    on_wall = world["walls"][tile[:, 0], tile[:, 1]] != 0
    m_on = ((mons["pos"][None] == tile[:, None]).all(-1) & mons["active"][None]).any(-1)
    p_on = ((players["pos"][None] == tile[:, None]).all(-1) & (players["active"] & ~players["dead"])[None]).any(-1)
    blocked = on_wall | m_on | p_on
    nxt = jnp.where(waiting & (nxt == 0) & blocked, 1, nxt)
    revive = waiting & (nxt == 0)

    # Death: start the timer.
    dk = jnp.where(mvp_died, kc, N_KINDS)
    nxt = nxt.at[dk].set(jnp.asarray(_TICKS)[kc], mode="drop")

    # Event row.
    head = mvp["event_head"]
    row = jnp.stack([k0, killer, world["tick"]]).astype(jnp.int32)
    ev = mvp["event"].at[jnp.where(any_died, head % EVENT_RING, EVENT_RING)].set(row, mode="drop")
    head = head + any_died.astype(jnp.int32)

    # Drops into the top damager's first free slots (loop over the 4 constant columns).
    inv = players["inv"]
    give = any_died & (killer >= 0)
    kl = jnp.clip(killer, 0, MAX_PLAYERS - 1)
    for j in range(_DROP.shape[1]):
        item = jnp.asarray(_DROP)[k0, j]
        empty = inv[kl] == ITEM_NONE
        ok = give & (item != ITEM_NONE) & empty.any()
        inv = inv.at[jnp.where(ok, kl, MAX_PLAYERS), jnp.argmax(empty)].set(item, mode="drop")

    # Reset trackers of every monster that died.
    died = mons["died"]
    mvp2 = {
        **mvp,
        "top_damager": jnp.where(died, -1, mvp["top_damager"]),
        "top_damage": jnp.where(died, 0, mvp["top_damage"]),
        "damage_ledger": jnp.where(died, 0, mvp["damage_ledger"]),
        "next_spawn": nxt,
        "mvp_slot": slot,
        "event": ev,
        "event_head": head,
    }

    # Hold generic respawn, revive MVPs whose timer ran out.
    held = jnp.where((slot >= 0) & (nxt > 0), slot, MAX_MONSTERS)
    respawn_in = mons["respawn_in"].at[held].set(0, mode="drop")
    ri = jnp.where(revive, slot, MAX_MONSTERS)
    new_mons = {
        **mons,
        "respawn_in": respawn_in.at[ri].set(0, mode="drop"),
        "active": mons["active"].at[ri].set(True, mode="drop"),
        "hp": mons["hp"].at[ri].set(_mon.HP[jnp.arange(N_KINDS)], mode="drop"),
        "pos": mons["pos"].at[ri].set(tile, mode="drop"),
        "died": mons["died"].at[ri].set(False, mode="drop"),
    }
    return {**world, "monsters": new_mons, "mvp": mvp2, "players": {**players, "inv": inv}}
