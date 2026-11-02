"""MVP bosses: damage tracking, kill event ring, top-damager drops, fixed-tile respawn.

Pure JAX, fixed shapes, int32. All state lives in world["mvp"] (see make_state).

Approximation (v1): RO ranks attackers by cumulative damage. Keeping a per-attacker
cumulative table needs a [MAX_MONSTERS, K] ledger, so v1 keeps `damage_ledger` as the
total damage per monster and `top_damage`/`top_damager` as the best SINGLE tick hit
(lowest attacker slot wins ties). A later worker can swap in a cumulative ledger.

Respawn: `phase` runs after combat each tick and holds `monsters.respawn_in[mvp_slot] = 0`
while `next_spawn > 0`, so the generic monster respawn never fires for MVP slots; the MVP
is revived here on its fixed tile with full MVP_HP when `next_spawn` reaches 0.
Never index config monster tables with an MVP kind (the kind table here is longer).
"""
import jax.numpy as jnp
import numpy as np

from game.config import (
    ITEM_GOLD_COIN,
    ITEM_MEAT,
    ITEM_NONE,
    ITEM_SWORD,
    MAP,
    MAX_MONSTERS,
    MAX_PLAYERS,
)
from game.world_layout import RAT_RING, ROADS, town_centers

MVP_KIND = (False, False, True)
MVP_SPAWN_TICKS = (0, 0, 3000)
MVP_HP = (0, 0, 2000)
N_MVP_KINDS = len(MVP_KIND)
# Orc Lord spawns on the vertical leg of the Center -> town 2 road (roads are L-shaped: horizontal at the
# first town's y, then vertical at the second town's x), RAT_RING * 1.5 tiles north of Center's y. A road tile
# is always walkable and outside both town boxes, so the fixed spawn never lands on a wall or in a town.
_ofs = int(round(RAT_RING * MAP * 1.5))
_c0, _c2 = town_centers()[0], town_centers()[ROADS[1][1]]
MVP_SPAWN_TILE = ((0, 0), (0, 0), (int(_c2[0]), int(_c0[1]) - _ofs))
MVP_DROP = ((0,) * 4, (0,) * 4, (ITEM_SWORD, ITEM_GOLD_COIN, ITEM_GOLD_COIN, ITEM_MEAT))
EVENT_RING = 8

_KIND = np.array(MVP_KIND, bool)
_TICKS = np.array(MVP_SPAWN_TICKS, np.int32)
_HP = np.array(MVP_HP, np.int32)
_TILE = np.array(MVP_SPAWN_TILE, np.int32)
_DROP = np.array(MVP_DROP, np.int32)
_BIG = np.int32(2 ** 30)


def make_state():
    return {
        "top_damager": jnp.full((MAX_MONSTERS,), -1, jnp.int32),
        "top_damage": jnp.zeros((MAX_MONSTERS,), jnp.int32),
        "damage_ledger": jnp.zeros((MAX_MONSTERS,), jnp.int32),
        "next_spawn": jnp.zeros((N_MVP_KINDS,), jnp.int32),
        "mvp_slot": jnp.full((N_MVP_KINDS,), -1, jnp.int32),
        "event": jnp.full((EVENT_RING, 3), -1, jnp.int32),
        "event_head": jnp.zeros((), jnp.int32),
    }


def _mvp_of_kind(kind):
    in_range = (kind >= 0) & (kind < N_MVP_KINDS)
    return in_range & jnp.asarray(_KIND)[jnp.clip(kind, 0, N_MVP_KINDS - 1)]


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


def place_mvps(world):
    """Put each MVP kind k in monster slot MAX_MONSTERS - 1 - k, alive on its tile."""
    mons, mvp = world["monsters"], world["mvp"]
    k = np.arange(N_MVP_KINDS)
    slots = jnp.asarray(MAX_MONSTERS - 1 - k, jnp.int32)
    live = jnp.asarray(_KIND)
    idx = jnp.where(live, slots, MAX_MONSTERS)
    tile = jnp.asarray(_TILE)
    new = {
        **mons,
        "kind": mons["kind"].at[idx].set(jnp.asarray(k, jnp.int32), mode="drop"),
        "hp": mons["hp"].at[idx].set(jnp.asarray(_HP), mode="drop"),
        "pos": mons["pos"].at[idx].set(tile, mode="drop"),
        "spawn": mons["spawn"].at[idx].set(tile, mode="drop"),
        "active": mons["active"].at[idx].set(True, mode="drop"),
        "respawn_in": mons["respawn_in"].at[idx].set(0, mode="drop"),
    }
    mv = {**mvp, "mvp_slot": jnp.where(live, slots, mvp["mvp_slot"])}
    return {**world, "monsters": new, "mvp": mv}


def phase(world):
    mons, mvp, players = world["monsters"], world["mvp"], world["players"]
    kind = mons["kind"]
    kc = jnp.clip(kind, 0, N_MVP_KINDS - 1)
    mvp_died = mons["died"] & _mvp_of_kind(kind)
    any_died = mvp_died.any()
    first = jnp.argmax(mvp_died)
    k0 = kc[first]
    killer = mvp["top_damager"][first]

    # Respawn countdown (uses state from before this tick's deaths).
    slot = mvp["mvp_slot"]
    sl = jnp.clip(slot, 0, MAX_MONSTERS - 1)
    waiting = (slot >= 0) & ~mons["active"][sl] & (mvp["next_spawn"] > 0)
    nxt = jnp.where(waiting, mvp["next_spawn"] - 1, mvp["next_spawn"])
    # A blocked spawn tile (wall, active monster or living player) holds the timer at 1 and retries next tick,
    # like monsters._respawn.
    tile = jnp.asarray(_TILE)
    on_wall = world["walls"][tile[:, 0], tile[:, 1]] != 0
    m_on = ((mons["pos"][None] == tile[:, None]).all(-1) & mons["active"][None]).any(-1)
    p_on = ((players["pos"][None] == tile[:, None]).all(-1) & (players["active"] & ~players["dead"])[None]).any(-1)
    blocked = on_wall | m_on | p_on
    nxt = jnp.where(waiting & (nxt == 0) & blocked, 1, nxt)
    revive = waiting & (nxt == 0)

    # Death: start the timer.
    dk = jnp.where(mvp_died, kc, N_MVP_KINDS)
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
        "hp": mons["hp"].at[ri].set(jnp.asarray(_HP), mode="drop"),
        "pos": mons["pos"].at[ri].set(tile, mode="drop"),
        "spawn": mons["spawn"].at[ri].set(tile, mode="drop"),
        "died": mons["died"].at[ri].set(False, mode="drop"),
    }
    return {**world, "monsters": new_mons, "mvp": mvp2, "players": {**players, "inv": inv}}
