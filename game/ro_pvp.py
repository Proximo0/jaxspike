"""PvP (ro4-pvp): per-player PK flag, PvP kill/death counters, the toggle phase and the area rules. Pure JAX.

State block world["pvp"] (make_state), int32/bool [MAX_PLAYERS]:
  pk          the player's PK flag (toggled by the one-shot `pvp` input; cleared on an inactive slot)
  pvp_kills   players this slot killed in PvP
  pvp_deaths  times this slot was killed in PvP (these deaths are NOT counted in players.deaths, so the
              ro_stats death penalty never fires for them: PvP deaths cost no exp; respawn is the usual one)

Where PvP is allowed (world_layout): the arena floor (`in_arena`) always, for everyone in it; elsewhere only in a
zone whose `pvp_zone` flag is on AND with both players PK-enabled (towns and dungeons never; fields off in v1).
`pair_ok` is the rule for one attacker/target pair: both in the arena, or both outside it in PvP zones with PK on.

The damage itself is game/combat.py's PvP path (same RO formulas as players vs monsters). It needs at most
PVP_SLOTS PvP-capable players per world at once: combat compacts them into a fixed [PVP_SLOTS] list and matches
them pairwise ([PVP_SLOTS, PVP_SLOTS]); players beyond PVP_SLOTS (lowest slots first) skip PvP that tick.

phase(world, inputs) runs after combat: the `pvp` input (bool [MAX_PLAYERS], one-shot) flips the PK flag; turning it
ON is refused while the player stands in a town (outside the arena) or is dead. Without the input key the phase only
clears the flag of inactive slots.

Round 5 (ro5-pvp2):
  duel     int32 [MAX_PLAYERS]: the explicit PvP target slot (-1 = none). Set by phase from `inputs["target"]` on a
           tick with no skill and no party op: a player entity id (not your own) locks on, any other id >= 0 (a
           monster, your own id) clears it; -1 (no input) keeps it. Cleared when either side dies or logs out.
           combat swings at the locked player first (even with a monster in reach) when the pair is allowed and in
           weapon range; otherwise the old auto-pick (nearest allowed player, only with no monster in reach) applies.
  by, dmg  int32 [MAX_PLAYERS, CREDIT_SLOTS]: per TARGET player, the CREDIT_SLOTS attackers with the most cumulative
           PvP damage on it (ro_mvp's per-player ledger mirrored: keyed by target, since a player is hit by few
           players at once). `record_hits` merges one tick's hits; `top_damager` is the PvP kill credit. A target's
           row clears when it leaves the PvP areas or dies (phase); entries of logged-out attackers drop.
Crit shield: a PvP crit chance (per mille) loses LUK_CRIT_SHIELD x target LUK (classic: target LUK / 5 percent).
"""
import jax
import jax.numpy as jnp

from game import world_layout as wl
from game.config import MAX_PLAYERS

PVP_SLOTS = 64  # fixed PvP candidate list per world (see the module docstring)
CREDIT_SLOTS = 4  # attackers each target player accumulates PvP damage from at once (ro5-pvp2)
LUK_CRIT_SHIELD = 2  # per mille of crit chance removed per point of target LUK (classic: LUK / 5 percent)
_PVP_ZONE = jnp.asarray(wl.pvp_zone())                                         # bool [N_ZONES]
_TOWN_ZONE = jnp.asarray([z[1] == wl.ZONE_TOWN for z in wl.ZONES], jnp.bool_)  # bool [N_ZONES]


def make_state():
    return {
        "pk": jnp.zeros((MAX_PLAYERS,), jnp.bool_),
        "pvp_kills": jnp.zeros((MAX_PLAYERS,), jnp.int32),
        "pvp_deaths": jnp.zeros((MAX_PLAYERS,), jnp.int32),
        "duel": jnp.full((MAX_PLAYERS,), -1, jnp.int32),
        "by": jnp.full((MAX_PLAYERS, CREDIT_SLOTS), -1, jnp.int32),
        "dmg": jnp.zeros((MAX_PLAYERS, CREDIT_SLOTS), jnp.int32),
    }


def fresh_row():
    """{key: value} of one slot of a fresh block (a slot changing hands gets these: server _reset_rows_fn)."""
    return {k: v[0] for k, v in make_state().items()}


def areas(world):
    """(arena bool [MAX_PLAYERS], field bool [MAX_PLAYERS]): on the arena floor; in a PvP zone with PK on (outside
    the arena). Only living, active players count."""
    p = world["players"]
    alive = p["active"] & ~p["dead"]
    arena = wl.in_arena(p["pos"]) & alive
    pk = world["pvp"]["pk"] if "pvp" in world else jnp.zeros_like(alive)
    field = _PVP_ZONE[wl.zone_of(p["pos"])] & pk & alive & ~arena
    return arena, field


def pair_ok(arena_a, field_a, arena_b, field_b):
    """bool, broadcasting: attacker a may hit target b (both in the arena, or both in PvP zones with PK on)."""
    return (arena_a & arena_b) | (field_a & field_b)


def phase(world, inputs):
    """Apply the PK toggle (one-shot `pvp` input) and clear the flag of inactive slots. No-op without the block."""
    if "pvp" not in world:
        return world
    p, v = world["players"], world["pvp"]
    pk = v["pk"]
    toggle = inputs.get("pvp", None)
    if toggle is not None:
        in_town = _TOWN_ZONE[wl.zone_of(p["pos"])] & ~wl.in_arena(p["pos"])
        may_on = p["active"] & ~p["dead"] & ~in_town
        flip = toggle.astype(jnp.bool_) & p["active"] & (pk | may_on)  # turning off always works, on only outside towns
        pk = pk ^ flip
    out = {**v, "pk": pk & p["active"]}
    if "duel" in v:
        out.update(_duel_and_ledger(world, v, inputs))
    return {**world, "pvp": out}


def _duel_and_ledger(world, v, inputs):
    """The round-5 rows after one tick: the explicit duel target from `inputs["target"]`, cleared for the dead and
    the logged out; the damage ledger cleared for targets outside the PvP areas (or dead) and for gone attackers."""
    p = world["players"]
    n = p["active"].shape[0]
    slot = jnp.arange(n, dtype=jnp.int32)
    duel = v["duel"]
    t = inputs.get("target", None)
    if t is not None:
        t = t.astype(jnp.int32)
        plain = jnp.ones((n,), jnp.bool_)
        for k in ("skill", "party"):  # a skill or a party op owns `target` on its tick
            if k in inputs:
                plain = plain & (inputs[k] == 0)
        lock = plain & (t >= 0) & (t < n) & (t != slot)
        clear = plain & ((t >= n) | (t == slot))
        duel = jnp.where(lock, t, jnp.where(clear, -1, duel))
    live = p["active"] & ~p["dead"]
    dc = jnp.clip(duel, 0, n - 1)
    duel = jnp.where((duel >= 0) & live & live[dc], duel, -1).astype(jnp.int32)
    arena, field = areas(world)
    by, dmg = v["by"], v["dmg"]
    keep = (arena | field)[:, None] & (by >= 0) & p["active"][jnp.clip(by, 0, n - 1)] & (dmg > 0)
    return {"duel": duel, "by": jnp.where(keep, by, -1), "dmg": jnp.where(keep, dmg, 0)}


def record_hits(v, idx, tcol, dmg):
    """Merge one tick's landed PvP hits into the ledger. idx int32 [K] compact candidate slots (MAX_PLAYERS = pad),
    tcol int32 [K] each attacker's target column in idx, dmg int32 [K] landed damage (0 = none). Each target row keeps
    its CREDIT_SLOTS largest cumulative attackers (earlier holders first on exact ties). Returns the new block."""
    n = v["by"].shape[0]
    k = idx.shape[0]
    valid = idx < n
    ic = jnp.minimum(idx, n - 1)
    hits = jnp.zeros((k, k), jnp.int32).at[tcol, jnp.arange(k)].add(jnp.where(valid, dmg, 0))  # [target j, attacker i]
    by, dm = v["by"][ic], v["dmg"][ic]                                                          # [K, C]
    match = (by[:, None, :] == idx[None, :, None]) & valid[None, :, None] & (by[:, None, :] >= 0)  # [K, K, C]
    carried = jnp.sum(jnp.where(match, dm[:, None, :], 0), axis=-1)                              # [K, K]
    total = hits + carried
    moved = jnp.any(match, axis=1)                                                               # [K, C]
    vals = jnp.concatenate([jnp.where(moved, 0, dm), total], axis=1)                             # [K, C + K]
    ids = jnp.concatenate([jnp.where(moved, -1, by), jnp.broadcast_to(idx[None, :], (k, k))], axis=1)
    vals = jnp.where((ids >= 0) & (ids < n) & (vals > 0), vals, -1)
    top, pos = jax.lax.top_k(vals, v["by"].shape[1])  # ties: the lower column (earlier holder) first
    new_by = jnp.where(top > 0, jnp.take_along_axis(ids, pos, axis=1), -1)
    new_dm = jnp.maximum(top, 0)
    rows = jnp.where(valid & jnp.any(hits > 0, axis=1), idx, n)
    return {**v, "by": v["by"].at[rows].set(new_by, mode="drop"), "dmg": v["dmg"].at[rows].set(new_dm, mode="drop")}


def top_damager(v, active):
    """int32 [MAX_PLAYERS]: per target, the logged-in attacker with the most cumulative PvP damage (-1 = none)."""
    n = v["by"].shape[0]
    ok = (v["by"] >= 0) & active[jnp.clip(v["by"], 0, n - 1)] & (v["dmg"] > 0)
    d = jnp.where(ok, v["dmg"], -1)
    j = jnp.argmax(d, axis=1)
    best = jnp.take_along_axis(v["by"], j[:, None], axis=1)[:, 0]
    return jnp.where(jnp.any(ok, axis=1), best, -1).astype(jnp.int32)
