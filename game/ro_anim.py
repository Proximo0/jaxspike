"""RO animation state in the tick (docs/round2_contract.md section 1.2). Pure JAX, fixed shapes, masks.

Entities are players then monsters (N = MAX_PLAYERS + MAX_MONSTERS), the same order as world["move"].
Phases call the mask API while they run; `phase(world)` runs last in the tick and derives the rest:

  set_action(world, mask, action, dur_ms, face=None)  any action (attack, cast, sit, pickup...) starts this tick
  note_hit(world, mask, dmg)                          damage number this tick + hurt flinch over idle/walk/standby
  on_combat(...)                                      the combat phase's swings and hits in one call

phase() rules: dead is sticky while the player is dead / the monster inactive, then idle; a one-shot action
(pickup, attack, hurt, cast) ends after dur_ms and falls to standby when the entity swung within STANDBY_MS, else
idle; standby times out to idle; a step (or the move block's `moving`) sets walk unless an action started this
tick or a cast is running; a walker that is still and has no intent goes idle. `face` follows world["move"]["dir"]
while moving (without the move block: the step's direction); actions keep the face their hook gave them.
`hit_dmg` is 0 at the end of every tick in which the entity was not hit.
"""
import jax.numpy as jnp

from game.config import MAX_MONSTERS, MAX_PLAYERS, MONSTER_ATTACK_COOLDOWN, TICK_MS
from game.data import monsters as mdata
from game.ro_move import dir_toward

N = MAX_PLAYERS + MAX_MONSTERS
IDLE, WALK, SIT, PICKUP, STANDBY, ATTACK, HURT, DEAD, CAST = range(9)
ACTION_NAMES = ("idle", "walk", "sit", "pickup", "standby", "attack", "hurt", "dead", "cast")
HURT_MS = 300
STANDBY_MS = 3000
DEFAULT_SPEED_MS = 150
# Without world["move"], legacy monsters step every MOVE_TICKS ticks: stay "walk" that long between steps.
LEGACY_STILL_TICKS = int(jnp.max(mdata.MOVE_TICKS))
_ELAPSED_CAP = 1 << 16  # ticks; x TICK_MS stays far inside int32
_ONESHOT = (PICKUP, ATTACK, HURT, CAST)
_FLINCHABLE = (IDLE, WALK, STANDBY)
_MON_AMOTION = jnp.asarray(getattr(mdata, "AMOTION_MS", jnp.full(mdata.MOVE_TICKS.shape,
                                                                 (MONSTER_ATTACK_COOLDOWN + 1) * TICK_MS // 2)),
                           jnp.int32)


def make_state():
    """The world["anim"] block: everyone idle facing south, never hit."""
    z = lambda dt: jnp.zeros((N,), dt)
    return {
        "action": z(jnp.int8), "start_tick": z(jnp.int32), "dur_ms": z(jnp.int16), "face": z(jnp.int8),
        "hit_tick": jnp.full((N,), -1, jnp.int32), "hit_dmg": z(jnp.int32),
        # ro-anim's own bookkeeping (not on the wire)
        "swing_tick": jnp.full((N,), -(2**30), jnp.int32), "last_pos": jnp.zeros((N, 2), jnp.int32),
        "still": z(jnp.int16),
    }


def _isin(a, values):
    out = jnp.zeros(a.shape, jnp.bool_)
    for v in values:
        out = out | (a == v)
    return out


def _ent_pos(world):
    return jnp.concatenate([world["players"]["pos"], world["monsters"]["pos"]]).astype(jnp.int32)


def amotion_ms(world, derived=None):
    """Attack clip length in ms per entity, int32 [N]: half the real swing period (RO: delay = 2 x amotion).

    Players: the combat cooldown (derived aspd_ticks with the buff aspd_pct); monsters: AMOTION_MS per kind when
    game/data/monsters.py has it, else half of the MONSTER_ATTACK_COOLDOWN period. `derived` defaults to ro_stats.derived(world)."""
    from game import ro_skills, ro_stats  # ro_skills imports this module
    d = ro_stats.derived(world) if derived is None else derived
    mod = ro_skills.buff_stat_mods(world)
    period = jnp.maximum(1, d["aspd_ticks"] * 100 // (100 + mod["aspd_pct"]))
    return jnp.concatenate([period * TICK_MS // 2, _MON_AMOTION[world["monsters"]["kind"]]]).astype(jnp.int32)


def set_action(world, mask, action, dur_ms, face=None):
    """Start `action` this tick where mask (bool [N]); dur_ms and face are scalars or [N]. No-op without the block."""
    if "anim" not in world:
        return world
    a, t = world["anim"], world["tick"]
    act = jnp.broadcast_to(jnp.asarray(action, jnp.int32), (N,))
    out = {
        **a,
        "action": jnp.where(mask, act, a["action"].astype(jnp.int32)).astype(jnp.int8),
        "start_tick": jnp.where(mask, t, a["start_tick"]).astype(jnp.int32),
        "dur_ms": jnp.where(mask, jnp.clip(jnp.asarray(dur_ms, jnp.int32), 0, 32767),
                            a["dur_ms"].astype(jnp.int32)).astype(jnp.int16),
        "swing_tick": jnp.where(mask & (act == ATTACK), t, a["swing_tick"]).astype(jnp.int32),
    }
    if face is not None:
        f = jnp.asarray(face, jnp.int32)
        out["face"] = jnp.where(mask & (f >= 0), f, a["face"].astype(jnp.int32)).astype(jnp.int8)
    return {**world, "anim": out}


def note_hit(world, mask, dmg):
    """Record damage taken this tick where mask (summed over several hits in one tick) and flinch (hurt, HURT_MS)
    the entities that are idle, walking or in standby. No-op without the block."""
    if "anim" not in world:
        return world
    a, t = world["anim"], world["tick"]
    dmg = jnp.asarray(dmg, jnp.int32)
    hit_dmg = jnp.where(mask, jnp.where(a["hit_tick"] == t, a["hit_dmg"] + dmg, dmg), a["hit_dmg"])
    world = {**world, "anim": {**a, "hit_dmg": hit_dmg.astype(jnp.int32),
                               "hit_tick": jnp.where(mask, t, a["hit_tick"]).astype(jnp.int32)}}
    flinch = mask & _isin(world["anim"]["action"].astype(jnp.int32), _FLINCHABLE)
    return set_action(world, flinch, HURT, HURT_MS)


def on_combat(world, p_swing, p_tgt, m_swing, m_tgt, dmg_to_monsters, dmg_to_players):
    """Combat's hook: attack on swingers (dur = amotion, facing the target), then hits on the damaged.

    p_swing bool [MAX_PLAYERS] with p_tgt monster index; m_swing bool [MAX_MONSTERS] with m_tgt player index;
    dmg_* int [MAX_MONSTERS] / [MAX_PLAYERS] damage taken this tick."""
    if "anim" not in world:
        return world
    p_pos, m_pos = world["players"]["pos"], world["monsters"]["pos"]
    swing = jnp.concatenate([p_swing, m_swing])
    tgt_pos = jnp.concatenate([m_pos[p_tgt], p_pos[m_tgt]])
    face = dir_toward(tgt_pos - _ent_pos(world))
    world = set_action(world, swing, ATTACK, amotion_ms(world), face)
    dmg = jnp.concatenate([dmg_to_players, dmg_to_monsters]).astype(jnp.int32)
    return note_hit(world, dmg > 0, dmg)


def phase(world):
    """Derive dead / idle / walk / standby transitions and facing; runs last in the tick. No-op without the block."""
    if "anim" not in world:
        return world
    a, t = world["anim"], world["tick"]
    p, m = world["players"], world["monsters"]
    act = a["action"].astype(jnp.int32)
    pos = _ent_pos(world)
    stepped = jnp.any(pos != a["last_pos"], axis=1)
    still = jnp.where(stepped, 0, jnp.minimum(a["still"].astype(jnp.int32) + 1, 1000))
    if "move" in world:
        mv = world["move"]
        moving = mv["moving"] | stepped
        step_face = mv["dir"].astype(jnp.int32)
        still_need = 1
    else:
        moving = stepped
        step_face = dir_toward(pos - a["last_pos"])
        still_need = LEGACY_STILL_TICKS

    started_now = a["start_tick"] == t
    # elapsed ms in clipped tick deltas: no int32 overflow on old start ticks or the swing_tick sentinel
    since_start = jnp.minimum(t - a["start_tick"], _ELAPSED_CAP) * TICK_MS
    expired = _isin(act, _ONESHOT) & ~started_now & (since_start >= a["dur_ms"].astype(jnp.int32))
    recent = jnp.minimum(t - a["swing_tick"], _ELAPSED_CAP) * TICK_MS < STANDBY_MS
    rest = jnp.where(recent, STANDBY, IDLE)
    new = jnp.where(expired, rest, act)
    new = jnp.where((new == STANDBY) & ~recent, IDLE, new)
    new = jnp.where(moving & ~started_now & (new != CAST), WALK, new)
    new = jnp.where((new == WALK) & ~moving & (still >= still_need), rest, new)
    dead_now = jnp.concatenate([p["active"] & p["dead"], ~m["active"]])
    new = jnp.where(dead_now, DEAD, jnp.where(act == DEAD, IDLE, new))

    changed = new != act
    keep_face = started_now & _isin(act, (ATTACK, CAST))
    face = jnp.where(stepped & ~keep_face, step_face, a["face"].astype(jnp.int32))
    if "move" in world:
        face = jnp.where(moving & ~keep_face, step_face, face)
    anim = {
        **a,
        "action": new.astype(jnp.int8),
        "start_tick": jnp.where(changed, t, a["start_tick"]).astype(jnp.int32),
        "dur_ms": jnp.where(changed, 0, a["dur_ms"].astype(jnp.int32)).astype(jnp.int16),
        "face": jnp.where(face >= 0, face, a["face"].astype(jnp.int32)).astype(jnp.int8),
        "hit_dmg": jnp.where(a["hit_tick"] == t, a["hit_dmg"], 0).astype(jnp.int32),
        "last_pos": pos,
        "still": still.astype(jnp.int16),
    }
    return {**world, "anim": anim}


def wire_fields(world):
    """Per-entity wire columns over N (players, monsters) for game/view.py: action, face, start_tick, dur_ms,
    speed_ms, hit. Zeros (speed: the legacy pace) when the blocks are absent."""
    if "move" in world:
        speed = world["move"]["speed_ms"].astype(jnp.int32)
    else:
        speed = jnp.concatenate([jnp.full((MAX_PLAYERS,), TICK_MS, jnp.int32),
                                 mdata.MOVE_TICKS[world["monsters"]["kind"]].astype(jnp.int32) * TICK_MS])
    if "anim" not in world:
        z = jnp.zeros((N,), jnp.int32)
        return {"action": z, "face": z, "start_tick": z, "dur_ms": z, "speed_ms": speed, "hit": z}
    a = world["anim"]
    return {"action": a["action"].astype(jnp.int32), "face": a["face"].astype(jnp.int32),
            "start_tick": a["start_tick"], "dur_ms": a["dur_ms"].astype(jnp.int32), "speed_ms": speed,
            "hit": a["hit_dmg"]}
