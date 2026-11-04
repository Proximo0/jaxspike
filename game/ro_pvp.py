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

Round 6 (ro6-pvp3):
  chase       bool [MAX_PLAYERS]: phase wrote this player's move intent toward its duel target (auto-chase). A
              player with a locked, allowed duel target (pair_ok) farther than its weapon range gets move.path_dir
              one step toward it every tick (ro_move.want, the interface the monsters use; movement applies it on
              the next tick); it stops at weapon range, when the lock clears or the pair leaves the PvP areas (the
              intent it wrote is cleared), and a human's own move input this tick wins (no chase written).
  quiet       int32 [MAX_PLAYERS]: ticks since this target last took a PvP hit; its ledger row clears once it has
              gone LEDGER_DECAY_MS without one (record_hits resets it).
  skill_hit   int32 [MAX_PLAYERS]: PvP skill damage taken this tick (ro_skills_effects writes it before combat,
              combat reads it for the PvP death and zeroes it). A skill that drops a player to 0 hp leaves the
              death to combat, so the corpse shows and respawn waits a tick like an auto-attack kill.
  skill_kills int32 [MAX_PLAYERS]: PvP kills whose killing tick had skill damage on the target (credited to the
              same killer as pvp_kills: the top cumulative damager in the ledger).
  skill_target_ok: the single-target ATK/MATK skills (Bash, the bolts, Double Strafe, Soul Strike, Envenom,
              Mammonite, Frost Diver) may target another player when pair_ok allows; heals target any player.
"""
import jax
import jax.numpy as jnp

from game import world_layout as wl
from game.config import MAX_PLAYERS, TICK_MS
from game.data import skills as S

PVP_SLOTS = 64  # fixed PvP candidate list per world (see the module docstring)
CREDIT_SLOTS = 4  # attackers each target player accumulates PvP damage from at once (ro5-pvp2)
LUK_CRIT_SHIELD = 2  # per mille of crit chance removed per point of target LUK (classic: LUK / 5 percent)
LEDGER_DECAY_MS = 10000  # ro6-pvp3: a target's ledger row clears after this long without a PvP hit on it
LEDGER_DECAY_TICKS = (LEDGER_DECAY_MS + TICK_MS - 1) // TICK_MS
_PVP_SKILL = jnp.asarray((S.SKILL_TARGET == S.TARGET_ENEMY) & (S.SKILL_AOE == 0)
                         & ((S.SKILL_EFFECT == S.EFFECT_ATK) | (S.SKILL_EFFECT == S.EFFECT_MATK)))  # bool [N_SKILLS]
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
        "chase": jnp.zeros((MAX_PLAYERS,), jnp.bool_),
        "quiet": jnp.zeros((MAX_PLAYERS,), jnp.int32),
        "skill_hit": jnp.zeros((MAX_PLAYERS,), jnp.int32),
        "skill_kills": jnp.zeros((MAX_PLAYERS,), jnp.int32),
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


def skill_target_ok(world, skill, target):
    """bool [MAX_PLAYERS]: caster i may cast `skill[i]` (a single-target ATK/MATK skill) on player `target[i]`
    (another living player, pair_ok). False everywhere without the pvp block. ro6-pvp3."""
    skill = jnp.asarray(skill, jnp.int32)
    target = jnp.asarray(target, jnp.int32)
    if "pvp" not in world:
        return jnp.zeros(target.shape, jnp.bool_)
    n = world["players"]["active"].shape[0]
    slot = jnp.arange(n, dtype=jnp.int32)
    tc = jnp.clip(target, 0, n - 1)
    arena, field = areas(world)
    is_pl = (target >= 0) & (target < n) & (target != slot)
    return _PVP_SKILL[jnp.clip(skill, 0, S.N_SKILLS - 1)] & is_pl & pair_ok(arena, field, arena[tc], field[tc])


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
        if "chase" in v and "move" in world:
            world, out["chase"] = _chase(world, inputs, out["duel"], v["chase"])
    return {**world, "pvp": out}


def _chase(world, inputs, duel, was):
    """Auto-chase (ro6-pvp3): (world with move.path_dir written, chase bool [P]). A living player whose locked duel
    target is allowed (pair_ok) and farther than its weapon range (manhattan, as combat) wants one step toward it:
    the diagonal, except when the two stand diagonally adjacent (then the lower slot takes the x step and the higher
    the y step, so two mutual chasers meet instead of swapping diagonals forever), falling back to an axis step when
    the diagonal is walled. A human's own move input this tick wins. Rows this wrote last tick that stop chasing
    get their intent cleared (when no move input came this tick)."""
    from game import ro_items, ro_move  # lazy: ro_items pulls in ro_stats; keep this module light
    p = world["players"]
    n = p["active"].shape[0]
    slot = jnp.arange(n, dtype=jnp.int32)
    dc = jnp.clip(duel, 0, n - 1)
    arena, field = areas(world)
    pos = p["pos"].astype(jnp.int32)
    delta = pos[dc] - pos
    dist = jnp.abs(delta).sum(-1)
    own = inputs["move"] != 0 if "move" in inputs else jnp.zeros((n,), jnp.bool_)
    chase = ((duel >= 0) & (duel != slot) & pair_ok(arena, field, arena[dc], field[dc])
             & (dist > ro_items.weapon_range(world)) & ~own)
    s = jnp.sign(delta)
    sx, sy = s * jnp.asarray([1, 0], jnp.int32), s * jnp.asarray([0, 1], jnp.int32)
    adj_diag = (jnp.abs(delta[:, 0]) == 1) & (jnp.abs(delta[:, 1]) == 1)
    first = jnp.where(adj_diag[:, None], jnp.where((slot < duel)[:, None], sx, sy), s)
    walls = world["walls"]
    hi = walls.shape[0] - 1

    def free(d):
        t = jnp.clip(pos + d, 0, hi)
        ok = walls[t[:, 0], t[:, 1]] == 0
        diag = jnp.all(d != 0, axis=1)
        side = (walls[t[:, 0], pos[:, 1]] == 0) & (walls[pos[:, 0], t[:, 1]] == 0)
        return ok & (~diag | side) & jnp.any(d != 0, axis=1)

    cand = jnp.stack([first, sx, sy], axis=1)  # [P, 3, 2]
    ok = jnp.stack([free(first), free(sx), free(sy)], axis=1)
    pick = jnp.take_along_axis(cand, jnp.argmax(ok, axis=1)[:, None, None], axis=1)[:, 0]
    chase = chase & jnp.any(ok, axis=1)
    pd = jnp.where(chase, ro_move.dir_toward(pick), -1)
    mask = chase | (was & ~own)  # stopped chasers: clear the intent this phase wrote
    n_m = world["move"]["path_dir"].shape[0] - n
    pad = jnp.full((n_m,), -1, jnp.int32)
    world = ro_move.want(world, jnp.concatenate([pd, pad]), jnp.concatenate([mask, jnp.zeros((n_m,), jnp.bool_)]))
    return world, chase


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
    out = {"duel": duel}
    if "quiet" in v:  # ro6-pvp3: the row decays after LEDGER_DECAY_MS without a PvP hit on the target
        quiet = jnp.minimum(v["quiet"] + 1, LEDGER_DECAY_TICKS + 1).astype(jnp.int32)
        keep = keep & (quiet <= LEDGER_DECAY_TICKS)[:, None]
        out["quiet"] = quiet
    out.update({"by": jnp.where(keep, by, -1), "dmg": jnp.where(keep, dmg, 0)})
    return out


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
    out = {**v, "by": v["by"].at[rows].set(new_by, mode="drop"), "dmg": v["dmg"].at[rows].set(new_dm, mode="drop")}
    if "quiet" in v:  # ro6-pvp3: a hit restarts the decay clock of its target's row
        out["quiet"] = v["quiet"].at[rows].set(0, mode="drop")
    return out


def top_damager(v, active):
    """int32 [MAX_PLAYERS]: per target, the logged-in attacker with the most cumulative PvP damage (-1 = none)."""
    n = v["by"].shape[0]
    ok = (v["by"] >= 0) & active[jnp.clip(v["by"], 0, n - 1)] & (v["dmg"] > 0)
    d = jnp.where(ok, v["dmg"], -1)
    j = jnp.argmax(d, axis=1)
    best = jnp.take_along_axis(v["by"], j[:, None], axis=1)[:, 0]
    return jnp.where(jnp.any(ok, axis=1), best, -1).astype(jnp.int32)
