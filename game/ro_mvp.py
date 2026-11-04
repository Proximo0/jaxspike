"""MVP bosses: damage tracking, kill event ring, top-damager drops, in-place respawn.

Pure JAX, fixed shapes, int32. All state lives in world["mvp"] (see make_state). MVP identity comes from the
`MVP` flag of game/data/monsters.py (Gildscarab 18, Scorchstripe 19); state is indexed by monster kind
(`[N_KINDS]`), and the event ring stores the monster kind. MVPs are placed by `ro_spawns`/`spawn_population`
(each MVP kind once, in its dungeon), so the slot's own `spawn` is the respawn tile; there is no placement step.

Damage credit (round 3): RO ranks attackers by CUMULATIVE damage. A [MAX_MONSTERS, MAX_PLAYERS] ledger is
too big, so each player keeps a small fixed ledger of the last CREDIT_SLOTS monsters they hit
(`credit_mon`/`credit_dmg`, int32 [MAX_PLAYERS, CREDIT_SLOTS]); a hit on a monster already in the ledger adds to
that row, a hit on a new monster takes the row with the least damage (empty rows first). Per monster,
`top_damage` is the largest cumulative damage any one player has dealt it and `top_damager` that player
(lowest slot wins ties; an earlier holder keeps an exact tie). `damage_ledger` stays the total damage per monster.
A player who spreads hits over more than CREDIT_SLOTS monsters at once restarts the evicted monster's count
(documented approximation). The mvp phase clears every tracker of a monster that died this tick, including the
players' ledger rows for it, after loot and stats have read the credit.

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
    "Gildscarab": (_items.ITEM_PILFERBUG_CARD, _items.ITEM_MOONSTONE, _items.ITEM_GOLD_COIN, _items.ITEM_GOLD_COIN),
    "Scorchstripe": (_items.ITEM_GRUKK_BRUTE_CARD, _items.ITEM_STARSTEEL, _items.ITEM_GOLD_COIN, _items.ITEM_GOLD_COIN),
}
MVP_DROP = tuple(_DROPS.get(m.name, (0,) * 4) for m in _mon.MONSTERS)  # [N_KINDS, 4], padded with ITEM_NONE
EVENT_RING = 8
CREDIT_SLOTS = 4  # monsters a player accumulates damage on at once (fixed per-player ledger width)

_KIND = np.array(MVP_KIND, bool)
_TICKS = np.array(MVP_SPAWN_TICKS, np.int32)
_DROP = np.array(MVP_DROP, np.int32)
_BIG = np.int32(2 ** 30)


def make_state():
    return {
        "top_damager": jnp.full((MAX_MONSTERS,), -1, jnp.int32),
        "top_damage": jnp.zeros((MAX_MONSTERS,), jnp.int32),
        "damage_ledger": jnp.zeros((MAX_MONSTERS,), jnp.int32),
        "credit_mon": jnp.full((MAX_PLAYERS, CREDIT_SLOTS), -1, jnp.int32),
        "credit_dmg": jnp.zeros((MAX_PLAYERS, CREDIT_SLOTS), jnp.int32),
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


def _record_column(mvp, attacker_idx, monster_idx, dmg):
    """One hit per attacker: [N] vectors. Attackers must be distinct within a column (rows = players)."""
    valid = (dmg > 0) & (monster_idx >= 0) & (monster_idx < MAX_MONSTERS) & (attacker_idx >= 0) & (
        attacker_idx < MAX_PLAYERS)
    m = jnp.where(valid, monster_idx, MAX_MONSTERS).astype(jnp.int32)
    ms = jnp.where(valid, monster_idx, 0)
    a = jnp.where(valid, attacker_idx, MAX_PLAYERS).astype(jnp.int32)
    asafe = jnp.where(valid, attacker_idx, 0)
    ledger = mvp["damage_ledger"].at[m].add(dmg, mode="drop")

    # Per-player cumulative row: the matching monster's row, else the row holding the least damage.
    cmon, cdmg = mvp["credit_mon"], mvp["credit_dmg"]
    rows_m, rows_d = cmon[asafe], cdmg[asafe]  # [N, K]
    match = rows_m == ms[:, None]
    has = match.any(-1)
    row = jnp.where(has, jnp.argmax(match, -1), jnp.argmin(rows_d, -1)).astype(jnp.int32)
    prev = jnp.where(has, jnp.take_along_axis(rows_d, row[:, None], 1)[:, 0], 0)
    cum = prev + dmg
    cmon = cmon.at[a, row].set(ms, mode="drop")
    cdmg = cdmg.at[a, row].set(cum, mode="drop")

    # Per-monster top: cumulative beats the holder; lowest slot wins an exact tie among this column's hitters,
    # and a holder with the same total keeps it.
    old_top = mvp["top_damage"]
    new_top = old_top.at[m].max(cum, mode="drop")
    win = valid & (cum == new_top[ms])
    cand = jnp.full((MAX_MONSTERS,), _BIG, jnp.int32).at[jnp.where(win, m, MAX_MONSTERS)].min(
        a, mode="drop")
    old_who = jnp.where((old_top == new_top) & (mvp["top_damager"] >= 0), mvp["top_damager"], _BIG)
    best = jnp.minimum(cand, old_who)
    who = jnp.where(best < _BIG, best, mvp["top_damager"])
    return {**mvp, "damage_ledger": ledger, "credit_mon": cmon, "credit_dmg": cdmg,
            "top_damage": new_top, "top_damager": who}


def record_damage(world, attacker_idx, monster_idx, dmg):
    """Report landed player hits. Shapes [N] (one hit per attacker) or [N, H] (H hits per attacker, AoE).

    Attacker slots must be distinct within a column (callers pass rows = player slots); a monster hit by one
    attacker in several columns of one call accumulates correctly (columns run in order). Zero damage, -1 or
    out-of-range indices are ignored.
    """
    mvp = world["mvp"]
    attacker_idx = jnp.asarray(attacker_idx).astype(jnp.int32)
    monster_idx = jnp.asarray(monster_idx).astype(jnp.int32)
    dmg = jnp.asarray(dmg).astype(jnp.int32)
    if dmg.ndim == 1:
        attacker_idx, monster_idx, dmg = attacker_idx[:, None], monster_idx[:, None], dmg[:, None]
    attacker_idx = jnp.broadcast_to(attacker_idx, dmg.shape)
    monster_idx = jnp.broadcast_to(monster_idx, dmg.shape)
    for h in range(dmg.shape[1]):  # static column count (1 for auto-attacks, 9 for AoE skills)
        mvp = _record_column(mvp, attacker_idx[:, h], monster_idx[:, h], dmg[:, h])
    return {**world, "mvp": mvp}


def credit_of(world):
    """int32 [MAX_MONSTERS]: the logged-in player with the most cumulative damage on each monster (-1 = nobody)."""
    who = world["mvp"]["top_damager"]
    safe = jnp.clip(who, 0, MAX_PLAYERS - 1)
    return jnp.where((who >= 0) & world["players"]["active"][safe], who, -1).astype(jnp.int32)


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

    # Reset trackers of every monster that died, including the players' ledger rows that point at it.
    died = mons["died"]
    cmon = mvp["credit_mon"]
    stale = (cmon >= 0) & died[jnp.clip(cmon, 0, MAX_MONSTERS - 1)]
    mvp2 = {
        **mvp,
        "top_damager": jnp.where(died, -1, mvp["top_damager"]),
        "top_damage": jnp.where(died, 0, mvp["top_damage"]),
        "damage_ledger": jnp.where(died, 0, mvp["damage_ledger"]),
        "credit_mon": jnp.where(stale, -1, cmon),
        "credit_dmg": jnp.where(stale, 0, mvp["credit_dmg"]),
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
