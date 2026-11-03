"""Tests for game/ro_anim.py (docs/round2_contract.md 1.2): the block, the mask API, phase transitions, the combat and
skills hooks, and the wire fields in game/view.py and net/protocol.py (docs/anim_wire.md)."""
import jax
import jax.numpy as jnp
import numpy as np

from game import combat, ro_anim, ro_move, ro_skills
from game.config import MAX_MONSTERS, MAX_PLAYERS, MONSTER_HP, N_ENTITIES, TICK_MS
from game.data import skills as sk
from game.state import empty_inputs, make_empty_world
from game.view import changed_mask, entity_table
from net.protocol import entity_rows, ro_you

N = ro_anim.N
M0 = MAX_PLAYERS
PHASE = jax.jit(ro_anim.phase)
A = ro_anim


def _world(players=((0, 20, 20),), monsters=(), move=False, tick=10):
    """players: [(slot, x, y)], monsters: [(i, x, y, kind)]; the anim block (and the move block when move)."""
    w = make_empty_world(jax.random.PRNGKey(0))
    p, m = dict(w["players"]), dict(w["monsters"])
    for s, x, y in players:
        p["pos"] = p["pos"].at[s].set(jnp.array([x, y], jnp.int32))
        p["spawn"] = p["spawn"].at[s].set(jnp.array([x, y], jnp.int32))
        p["active"] = p["active"].at[s].set(True)
    for i, x, y, k in monsters:
        m["pos"] = m["pos"].at[i].set(jnp.array([x, y], jnp.int32))
        m["kind"] = m["kind"].at[i].set(k)
        m["hp"] = m["hp"].at[i].set(MONSTER_HP[k])
        m["active"] = m["active"].at[i].set(True)
    w = {**w, "players": p, "monsters": m, "tick": jnp.asarray(tick, jnp.int32)}
    an = ro_anim.make_state()
    an["last_pos"] = jnp.concatenate([p["pos"], m["pos"]]).astype(jnp.int32)
    w["anim"] = an
    if move:
        w["move"] = ro_move.make_state()
    else:
        w.pop("move", None)  # the legacy path: anim derives walking from position changes
    return w


def _mask(*ents):
    return jnp.zeros((N,), bool).at[jnp.asarray(ents, jnp.int32)].set(True)


def _tick(w, n=1):
    """phase, then the tick increment, n times (the step order: phases ... ro_anim.phase, tick + 1)."""
    for _ in range(n):
        w = PHASE(w)
        w = {**w, "tick": w["tick"] + 1}
    return w


def _act(w, e=0):
    return int(w["anim"]["action"][e])


def _move_player(w, s, dx, dy):
    p = w["players"]
    return {**w, "players": {**p, "pos": p["pos"].at[s].add(jnp.array([dx, dy], jnp.int32))}}


# ---------------------------------------------------------------- block and API
def test_make_state_shapes_and_dtypes():
    s = ro_anim.make_state()
    want = {"action": jnp.int8, "start_tick": jnp.int32, "dur_ms": jnp.int16, "face": jnp.int8,
            "hit_tick": jnp.int32, "hit_dmg": jnp.int32}
    for k, dt in want.items():
        assert s[k].shape == (N,) and s[k].dtype == dt, k
    assert N == MAX_PLAYERS + MAX_MONSTERS
    assert int(s["action"].max()) == A.IDLE and int(s["hit_dmg"].max()) == 0
    assert len(A.ACTION_NAMES) == 9 and A.ACTION_NAMES[A.CAST] == "cast"


def test_set_action_writes_only_masked_rows_and_noop_without_block():
    w = _world()
    w2 = A.set_action(w, _mask(0, M0 + 3), A.ATTACK, 400, face=jnp.full((N,), 6, jnp.int32))
    an = w2["anim"]
    assert _act(w2, 0) == A.ATTACK and _act(w2, M0 + 3) == A.ATTACK and _act(w2, 1) == A.IDLE
    assert int(an["start_tick"][0]) == 10 and int(an["dur_ms"][0]) == 400 and int(an["face"][0]) == 6
    assert int(an["face"][1]) == 0 and int(an["swing_tick"][0]) == 10
    bare = {k: v for k, v in w.items() if k != "anim"}
    assert A.set_action(bare, _mask(0), A.ATTACK, 400) is bare
    assert A.note_hit(bare, _mask(0), 5) is bare and A.phase(bare) is bare


def test_note_hit_sums_one_tick_and_flinches_only_idle_walk_standby():
    w = _world(players=((0, 20, 20), (1, 22, 20)))
    w = A.set_action(w, _mask(1), A.ATTACK, 500)
    w = A.note_hit(w, _mask(0, 1), 7)
    w = A.note_hit(w, _mask(0), 5)
    an = w["anim"]
    assert int(an["hit_dmg"][0]) == 12 and int(an["hit_tick"][0]) == 10 and int(an["hit_dmg"][1]) == 7
    assert _act(w, 0) == A.HURT and int(an["dur_ms"][0]) == A.HURT_MS
    assert _act(w, 1) == A.ATTACK  # an attack in progress is not interrupted
    w = A.set_action(w, _mask(2), A.CAST, 1000)
    w = A.note_hit(w, _mask(2), 3)
    assert _act(w, 2) == A.CAST


def test_hit_dmg_cleared_after_the_tick_it_landed():
    w = A.note_hit(_world(), _mask(0), 9)
    w = PHASE(w)
    assert int(w["anim"]["hit_dmg"][0]) == 9  # end of the hit tick: still visible to the view
    w = _tick({**w, "tick": w["tick"] + 1})
    assert int(w["anim"]["hit_dmg"][0]) == 0
    w = A.note_hit(w, _mask(0), 4)  # a new tick's hit replaces, does not sum with the old one
    assert int(w["anim"]["hit_dmg"][0]) == 4


# ---------------------------------------------------------------- phase transitions
def test_attack_ends_in_standby_then_idle_after_3s():
    w = A.set_action(_world(), _mask(0), A.ATTACK, 2 * TICK_MS)
    w = _tick(w)
    assert _act(w) == A.ATTACK
    w = _tick(w, 2)
    assert _act(w) == A.STANDBY and int(w["anim"]["start_tick"][0]) == 12
    standby_ticks = -(-A.STANDBY_MS // TICK_MS)
    w = _tick(w, standby_ticks - 4)
    assert _act(w) == A.STANDBY
    w = _tick(w, 3)
    assert _act(w) == A.IDLE and int(w["anim"]["dur_ms"][0]) == 0


def test_hurt_ends_in_idle_without_a_recent_swing():
    w = A.note_hit(_world(), _mask(0), 3)
    w = _tick(w, -(-A.HURT_MS // TICK_MS) + 1)
    assert _act(w) == A.IDLE


def test_legacy_walk_from_pos_change_with_face_then_idle():
    w = _world()
    w = _tick(_move_player(w, 0, 1, 0))      # east
    assert _act(w) == A.WALK and int(w["anim"]["face"][0]) == 6
    t0 = int(w["anim"]["start_tick"][0])
    w = _tick(_move_player(w, 0, -1, 1))     # south-west: still walking, same clip start, new face
    assert _act(w) == A.WALK and int(w["anim"]["start_tick"][0]) == t0 and int(w["anim"]["face"][0]) == 1
    w = _tick(w, A.LEGACY_STILL_TICKS - 1)
    assert _act(w) == A.WALK                 # legacy monsters step every MOVE_TICKS: no flicker
    w = _tick(w)
    assert _act(w) == A.IDLE and int(w["anim"]["face"][0]) == 1


def test_move_block_walk_follows_moving_and_dir():
    w = _world(move=True)
    mv = w["move"]
    w = {**w, "move": {**mv, "moving": mv["moving"].at[0].set(True), "dir": mv["dir"].at[0].set(4)}}
    w = _tick(w)                              # wants to move, no step yet: walk already (accumulating)
    assert _act(w) == A.WALK and int(w["anim"]["face"][0]) == 4
    w = _tick(_move_player(w, 0, 0, -1))
    assert _act(w) == A.WALK
    mv = w["move"]
    w = {**w, "move": {**mv, "moving": mv["moving"].at[0].set(False)}}
    w = _tick(w)
    assert _act(w) == A.IDLE and int(w["anim"]["face"][0]) == 4


def test_walk_does_not_override_cast_or_this_ticks_action():
    w = _world()
    w = A.set_action(w, _mask(0), A.CAST, 5 * TICK_MS)
    w = _tick(_move_player(w, 0, 1, 0))
    assert _act(w) == A.CAST
    w = _world(players=((0, 20, 20),))
    w = _move_player(w, 0, 1, 0)
    w = A.set_action(w, _mask(0), A.ATTACK, 300, face=jnp.full((N,), 2, jnp.int32))
    w = PHASE(w)
    assert _act(w) == A.ATTACK and int(w["anim"]["face"][0]) == 2
    w = _tick(_move_player({**w, "tick": w["tick"] + 1}, 0, 1, 0))
    assert _act(w) == A.WALK                  # a later step cancels the attack clip


def test_dead_is_sticky_until_respawn_then_idle():
    w = _world(monsters=((0, 30, 30, 1),))
    assert _act(_tick(w), M0 + 1) == A.DEAD   # inactive monster slots are dead
    w = {**w, "players": {**w["players"], "dead": w["players"]["dead"].at[0].set(True)}}
    w = _tick(w)
    assert _act(w) == A.DEAD and _act(w, M0) == A.IDLE
    w = A.note_hit(w, _mask(0), 5)
    w = _tick(w, 3)
    assert _act(w) == A.DEAD
    w = {**w, "players": {**w["players"], "dead": w["players"]["dead"].at[0].set(False)}}
    w = _tick(w)
    assert _act(w) == A.IDLE and int(w["anim"]["start_tick"][0]) == int(w["tick"]) - 1


def test_phase_is_jittable_and_keeps_dtypes():
    w = _world(move=True)
    w2 = PHASE(w)
    for k, v in w["anim"].items():
        assert w2["anim"][k].dtype == v.dtype and w2["anim"][k].shape == v.shape, k


# ---------------------------------------------------------------- hooks
def test_combat_hook_attack_faces_target_and_hits_show():
    w = _world(players=((0, 20, 20),), monsters=((0, 21, 20, 1),))
    inp = {**empty_inputs(), "attack": jnp.zeros((MAX_PLAYERS,), bool).at[0].set(True)}
    w1 = jax.jit(combat.combat)(w, inp)
    assert _act(w1, 0) == A.ATTACK and int(w1["anim"]["face"][0]) == 6  # E toward the monster
    amo = int(A.amotion_ms(w1)[0])
    assert int(w1["anim"]["dur_ms"][0]) == amo > 0
    dealt = MONSTER_HP[1] - int(w1["monsters"]["hp"][0])
    assert int(w1["anim"]["hit_dmg"][M0]) == dealt
    if dealt:
        assert _act(w1, M0) in (A.HURT, A.ATTACK)


def test_on_combat_monster_swing_and_player_damage():
    w = _world(players=((0, 20, 20),), monsters=((0, 20, 19, 1),))
    z = jnp.zeros
    w = A.on_combat(w, z((MAX_PLAYERS,), bool), z((MAX_PLAYERS,), jnp.int32),
                    z((MAX_MONSTERS,), bool).at[0].set(True), z((MAX_MONSTERS,), jnp.int32),
                    z((MAX_MONSTERS,), jnp.int32), z((MAX_PLAYERS,), jnp.int32).at[0].set(6))
    assert _act(w, M0) == A.ATTACK and int(w["anim"]["face"][M0]) == 0  # south toward the player
    assert _act(w, 0) == A.HURT and int(w["anim"]["hit_dmg"][0]) == 6


def _skill_world():
    w = _world(players=((0, 20, 20),), monsters=((0, 21, 20, 1),))
    w = {**w, "skills": {**w["skills"], "learned": w["skills"]["learned"].at[0, sk.SK_FIRE_BOLT].set(1)
                                                                       .at[0, sk.SK_BASH].set(1)}}
    return {**w, "stats": {**w["stats"], "sp": w["stats"]["sp"].at[0].set(200)}}


def _skill_inp(s):
    i = empty_inputs()
    return {**i, "skill": i["skill"].at[0].set(s), "target": i["target"].at[0].set(M0)}


def test_skill_cast_sets_cast_clip_and_after_delay_blocks():
    w = _skill_world()
    ph = jax.jit(ro_skills.phase)
    w1 = ph(w, _skill_inp(sk.SK_FIRE_BOLT))
    cast_ticks = int(w1["skills"]["cast_left"][0]) + 1
    assert _act(w1, 0) == A.CAST and int(w1["anim"]["dur_ms"][0]) == cast_ticks * TICK_MS
    assert int(w1["anim"]["face"][0]) == 6
    w2 = ph(w, _skill_inp(sk.SK_BASH))        # instant: attack motion, damage number, after-cast delay
    assert _act(w2, 0) == A.ATTACK
    dealt = MONSTER_HP[1] - int(w2["monsters"]["hp"][0])
    assert dealt > 0 and int(w2["anim"]["hit_dmg"][M0]) == dealt
    delay = int(w2["skills"]["after_delay"][0])
    assert delay >= 1
    w2 = {**w2, "skills": {**w2["skills"], "cooldown": jnp.zeros_like(w2["skills"]["cooldown"])}}
    w3 = ph(w2, _skill_inp(sk.SK_BASH))
    assert int(w3["skills"]["casts"][0]) == 1  # blocked by after_delay
    for _ in range(delay - 1):
        w3 = ph(w3, empty_inputs())
    w3 = {**w3, "skills": {**w3["skills"], "cooldown": jnp.zeros_like(w3["skills"]["cooldown"])}}
    w4 = ph(w3, _skill_inp(sk.SK_BASH))
    assert int(w4["skills"]["casts"][0]) == 2


# ---------------------------------------------------------------- wire
def test_entity_table_anim_columns_and_changed_mask():
    w = _world(players=((0, 20, 20), (1, 25, 20)), move=True)
    t0 = entity_table(w)
    for k in ("action", "face", "start_tick", "dur_ms", "speed_ms", "hit"):
        assert t0[k].shape == t0["vis"].shape and t0[k].shape[0] >= N_ENTITIES, k
    assert int(t0["speed_ms"][0]) == 150 and int(t0["speed_ms"][N]) == 0  # drops carry zeros
    assert not bool(changed_mask(t0, entity_table(w)).any())
    w1 = A.set_action(w, _mask(0), A.ATTACK, 300)
    w1 = A.note_hit(w1, _mask(1), 4)
    ch = np.asarray(changed_mask(t0, entity_table(w1)))
    assert ch[0] and ch[1] and not ch[2]
    t1 = entity_table(w1)
    assert int(t1["action"][1]) == A.HURT and int(t1["hit"][1]) == 4
    bare = entity_table({k: v for k, v in w.items() if k not in ("anim", "move")})
    assert int(bare["action"].max()) == 0 and int(bare["speed_ms"][0]) == TICK_MS


def test_entity_rows_and_you_anim_fields():
    w = _world(players=((0, 20, 20), (1, 25, 20)), monsters=((0, 21, 20, 1),), move=True)
    w = A.set_action(w, _mask(0), A.ATTACK, 300, face=jnp.full((N,), 6, jnp.int32))
    w = A.note_hit(w, _mask(M0), 11)
    tab = {k: np.asarray(v) for k, v in entity_table(w).items()}
    rows = {r["id"]: r for r in entity_rows(np.array([0, 1, M0]), tab, {})}
    assert (rows[0]["a"], rows[0]["d"], rows[0]["t"], rows[0]["m"], rows[0]["s"]) == (A.ATTACK, 6, 10, 300, 150)
    assert "h" not in rows[0] and "h" not in rows[1]
    assert rows[M0]["h"] == 11 and rows[M0]["a"] == A.HURT
    pull = {"tab": tab, "anim": {"amotion_ms": np.asarray(A.amotion_ms(w))},
            "players": {"inv": np.asarray(w["players"]["inv"])},
            "ro": {"stats": {k: np.asarray(v) for k, v in w["stats"].items()},
                   "skills": {k: np.asarray(v) for k, v in w["skills"].items()},
                   "items": {k: np.asarray(v) for k, v in w["items"].items()},
                   "craft": {k: np.asarray(v) for k, v in w["craft"].items()},
                   "party": {k: np.asarray(v) for k, v in w["party"].items()},
                   "derived": {"max_hp": np.full(MAX_PLAYERS, 100), "max_sp": np.full(MAX_PLAYERS, 10)}}}
    you = ro_you(pull, 0)
    assert you["speed_ms"] == 150 and you["amotion_ms"] == int(A.amotion_ms(w)[0]) > 0 and you["after_delay"] == 0
    del pull["anim"]
    assert ro_you(pull, 0)["amotion_ms"] == 0
