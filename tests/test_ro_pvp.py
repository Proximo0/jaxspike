"""ro4-pvp: arena layout, PK toggle, the PvP damage path in combat, PvP kill/death counters, crit on the wire,
the protocol fields and the duelling bot policy."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from game import combat as combat_mod
from game import map_gen, ro_anim, ro_items, ro_pvp, ro_stats, world_layout as wl
from game.combat import combat
from game.config import MAP
from game.data.stats import JOB_SWORDSMAN
from game.state import empty_inputs, make_empty_world

ARENA = wl.arena_box()
A0 = (int(ARENA[0]) + 3, int(ARENA[1]) + 3)   # a tile on the arena floor
A1 = (A0[0] + 1, A0[1])                        # its east neighbour
TOWN = tuple(int(v) for v in wl.town_centers()[0] + np.array([0, 3]))   # Solmera plaza, outside the arena
FIELD = (40, 40)                               # a field tile (lun_fld00)


def _fake_equipped(monkeypatch, weapon_atk=0, weapon_elem=0, armor_def=0):
    real = ro_items.equipped

    def fake(world):
        e = real(world)
        return {**e, "weapon_atk": e["weapon_atk"] + weapon_atk, "weapon_elem": e["weapon_elem"] + weapon_elem,
                "armor_def": e["armor_def"] + armor_def}

    monkeypatch.setattr(ro_items, "equipped", fake)


def _world(pos0=A0, pos1=A1, pk=(False, False)):
    w = make_empty_world(jax.random.PRNGKey(0))
    p = w["players"]
    p = {**p, "pos": p["pos"].at[0].set(jnp.array(pos0)).at[1].set(jnp.array(pos1)),
         "spawn": p["spawn"].at[0].set(jnp.array([5, 5])).at[1].set(jnp.array([7, 7])),
         "active": p["active"].at[0].set(True).at[1].set(True),
         "hp": p["hp"].at[0].set(10 ** 6).at[1].set(10 ** 6)}  # no deaths unless a test lowers hp
    v = ro_pvp.make_state()
    v = {**v, "pk": v["pk"].at[0].set(pk[0]).at[1].set(pk[1])}
    return {**w, "players": p, "pvp": v}


def _seed(w, s):
    return {**w, "rng": jax.random.key_data(jax.random.PRNGKey(s))}


def _attack(*slots):
    i = empty_inputs()
    a = i["attack"]
    for s in slots:
        a = a.at[s].set(True)
    return {**i, "attack": a}


def _dmg_to_1(w, seeds=range(30), slots=(0,)):
    out = []
    for s in seeds:
        o = combat(_seed(w, s), _attack(*slots))
        out.append(int(w["players"]["hp"][1]) - int(o["players"]["hp"][1]))
    return out


# --- layout -------------------------------------------------------------------------------------------------------

def test_zone_flags_and_arena_inside_solmera():
    flags = wl.pvp_zone()
    assert flags.shape == (wl.N_ZONES,) and not flags.any()  # v1: towns never, dungeons never, fields off
    box = wl.arena_box()
    corners = jnp.asarray([[box[0], box[1]], [box[2], box[3]]], jnp.int32)
    assert {wl.ZONE_NAMES[int(z)] for z in wl.zone_of(corners)} == {"solmera"}
    assert bool(wl.in_arena(jnp.asarray([A0]))[0]) and not bool(wl.in_arena(jnp.asarray([TOWN]))[0])
    assert not bool(wl.in_arena(jnp.asarray(wl.arena_gate()))[0])  # the gate itself is not arena floor


def test_arena_is_walled_with_an_open_gate_and_reachable_from_the_temple():
    from collections import deque
    ter = np.asarray(map_gen.make_terrain_ro(jax.random.PRNGKey(0)))
    free = np.asarray(map_gen.blocking(jnp.asarray(ter))) == 0
    x0, y0, x1, y1 = wl.arena_ring()
    gate = {tuple(g) for g in wl.arena_gate().tolist()}
    ring = [(x, y) for x in range(x0, x1 + 1) for y in (y0, y1)] + [(x, y) for y in range(y0, y1 + 1) for x in (x0, x1)]
    for t in ring:
        assert free[t] == (t in gate), t
    ax0, ay0, ax1, ay1 = wl.arena_box().tolist()
    assert free[ax0:ax1 + 1, ay0:ay1 + 1].all()  # open floor inside
    start = tuple(int(v) for v in np.asarray(wl.temple_tiles())[0, 0])
    seen, q = {start}, deque([start])
    while q:
        x, y = q.popleft()
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= n[0] < MAP and 0 <= n[1] < MAP and free[n] and n not in seen:
                seen.add(n)
                q.append(n)
    assert (ax0, ay0) in seen and (ax1, ay1) in seen


# --- toggle phase -------------------------------------------------------------------------------------------------

def test_toggle_refused_in_town_allowed_in_field_and_off_anywhere():
    w = _world(pos0=TOWN, pos1=FIELD)
    toggle = {"pvp": jnp.zeros(empty_inputs()["attack"].shape, jnp.bool_).at[0].set(True).at[1].set(True)}
    o = ro_pvp.phase(w, toggle)
    assert not bool(o["pvp"]["pk"][0]) and bool(o["pvp"]["pk"][1])
    o2 = ro_pvp.phase(o, toggle)  # flips 1 back off; 0 still refused
    assert not bool(o2["pvp"]["pk"][0]) and not bool(o2["pvp"]["pk"][1])
    on_in_town = {**w, "pvp": {**w["pvp"], "pk": w["pvp"]["pk"].at[0].set(True)}}
    assert not bool(ro_pvp.phase(on_in_town, toggle)["pvp"]["pk"][0])  # turning off works in town


def test_phase_without_input_only_clears_inactive_slots():
    w = _world(pk=(True, True))
    w = {**w, "players": {**w["players"], "active": w["players"]["active"].at[1].set(False)}}
    o = ro_pvp.phase(w, empty_inputs())
    assert bool(o["pvp"]["pk"][0]) and not bool(o["pvp"]["pk"][1])
    bare = {k: v for k, v in w.items() if k != "pvp"}
    assert "pvp" not in ro_pvp.phase(bare, empty_inputs())


# --- damage path --------------------------------------------------------------------------------------------------

def test_arena_pvp_damage_uses_the_ro_formula(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=100)
    w = _world()
    d = {k: int(v[0]) for k, v in ro_stats.derived(w).items()}
    atk = d["atk"] + 100
    soft = d["def"]
    dmgs = _dmg_to_1(w)
    hits = [x for x in dmgs if x > 0]
    # hit% = clip(80 + hit - flee, 5, 95) = 80 for two fresh Novices
    assert 18 <= len(hits) <= 30
    for x in hits:
        assert atk * 90 // 100 - soft <= x <= atk * 110 // 100 * 2
    o = combat(_seed(w, 0), _attack(0))
    assert int(o["players"]["attack_cd"][0]) == d["aspd_ticks"] - 1  # a PvP swing costs the ASPD cooldown
    assert int(o["players"]["hp"][0]) == int(w["players"]["hp"][0])  # player 1 did not attack


def test_no_pvp_in_town_even_with_pk_and_none_in_fields_in_v1(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=100)
    town = _world(pos0=TOWN, pos1=(TOWN[0] + 1, TOWN[1]), pk=(True, True))
    assert all(x == 0 for x in _dmg_to_1(town, range(8)))
    field = _world(pos0=FIELD, pos1=(FIELD[0] + 1, FIELD[1]), pk=(True, True))
    assert all(x == 0 for x in _dmg_to_1(field, range(8)))


def test_field_pvp_needs_both_pk_flags(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=100)
    monkeypatch.setattr(ro_pvp, "_PVP_ZONE", jnp.asarray([z[1] == wl.ZONE_FIELD for z in wl.ZONES]))
    both = _world(pos0=FIELD, pos1=(FIELD[0] + 1, FIELD[1]), pk=(True, True))
    assert sum(x > 0 for x in _dmg_to_1(both, range(12))) >= 5
    one = _world(pos0=FIELD, pos1=(FIELD[0] + 1, FIELD[1]), pk=(True, False))
    assert all(x == 0 for x in _dmg_to_1(one, range(8)))


def test_no_pvp_without_the_block_or_out_of_range(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=100)
    bare = {k: v for k, v in _world().items() if k != "pvp"}
    assert all(x == 0 for x in _dmg_to_1(bare, range(8)))
    far = _world(pos1=(A0[0] + 2, A0[1]))
    assert all(x == 0 for x in _dmg_to_1(far, range(8)))


def test_armor_def_and_weapon_element_apply(monkeypatch):
    from game.data.elements import ELEM_GHOST, elem_mult
    _fake_equipped(monkeypatch, weapon_atk=200)
    seeds = range(40)
    plain = sorted(x for x in _dmg_to_1(_world(), seeds) if x > 0)
    _fake_equipped(monkeypatch, armor_def=50)  # stacks on the first patch: the target wears 50% hard DEF
    armored = sorted(x for x in _dmg_to_1(_world(), seeds) if x > 0)
    assert armored[len(armored) // 2] < plain[len(plain) // 2] * 70 // 100
    ghost = int(elem_mult(jnp.int32(ELEM_GHOST), jnp.int32(0)))
    _fake_equipped(monkeypatch, weapon_elem=ELEM_GHOST)  # ghost weapon vs a neutral body
    g = sorted(x for x in _dmg_to_1(_world(), seeds) if x > 0)
    if ghost == 0:
        assert not g
    else:
        assert g and ghost < 100 and g[len(g) // 2] < armored[len(armored) // 2]


def test_kill_counts_pvp_kill_and_death_not_players_deaths_and_no_exp_loss(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=1000)
    w = _world()
    w = {**w, "players": {**w["players"], "hp": w["players"]["hp"].at[1].set(5)}}
    st = w["stats"]
    w = {**w, "stats": {**st, "job": st["job"].at[1].set(JOB_SWORDSMAN), "base_level": st["base_level"].at[1].set(30),
                        "base_exp": st["base_exp"].at[1].set(1000)}}
    for s in range(30):
        o = combat(_seed(w, s), _attack(0))
        if bool(o["players"]["dead"][1]):
            break
    assert bool(o["players"]["dead"][1])
    assert int(o["pvp"]["pvp_kills"][0]) == 1 and int(o["pvp"]["pvp_deaths"][1]) == 1
    assert int(o["players"]["deaths"][1]) == 0 and int(o["players"]["kills"][0]) == 0
    after = ro_stats.phase(o, empty_inputs())
    assert int(after["stats"]["base_exp"][1]) == 1000  # no death penalty
    pve = {**o, "players": {**o["players"], "deaths": o["players"]["deaths"].at[1].add(1)}}  # control: a PvE death
    assert int(ro_stats.phase(pve, empty_inputs())["stats"]["base_exp"][1]) < 1000


def test_monster_death_still_counts_and_penalises():
    from tests.test_combat import ORC, _world as pve_world
    w = pve_world(kind=ORC)
    w = {**w, "players": {**w["players"], "hp": w["players"]["hp"].at[0].set(1)}, "pvp": ro_pvp.make_state()}
    for s in range(40):
        o = combat(_seed(w, s), empty_inputs())
        if bool(o["players"]["dead"][0]):
            break
    assert int(o["players"]["deaths"][0]) == 1 and int(o["pvp"]["pvp_deaths"][0]) == 0


def test_pvp_swing_animates_and_misses_show():
    w = _world()
    w = {**w, "stats": {**w["stats"], "dex": w["stats"]["dex"].at[0].set(-200)}}  # hit 5%: mostly misses
    misses = 0
    for s in range(10):
        o = combat(_seed(w, s), _attack(0))
        assert int(o["anim"]["action"][0]) == ro_anim.ATTACK
        misses += int(o["anim"]["hit_dmg"][1]) == -1
    assert misses >= 5


def test_crit_is_flagged_and_sent_as_negative_hit(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=50)
    w = _world()
    w = {**w, "stats": {**w["stats"], "luk": w["stats"]["luk"].at[0].set(400)}}  # crit per mille 10 + 3 luk >= 1000
    o = combat(_seed(w, 0), _attack(0))
    o = ro_anim.phase(o)
    hit = int(o["anim"]["hit_dmg"][1])
    assert hit >= 2 and bool(o["anim"]["crit"][1])
    assert int(ro_anim.wire_fields(o)["hit"][1]) == -hit


def test_pve_crit_on_a_monster_is_negative_on_the_wire(monkeypatch):
    from tests.test_combat import _world as pve_world
    _fake_equipped(monkeypatch, weapon_atk=50)
    w = pve_world()
    w = {**w, "stats": {**w["stats"], "luk": w["stats"]["luk"].at[0].set(400)}}
    o = ro_anim.phase(combat(_seed(w, 0), _attack(0)))
    from game.config import MAX_PLAYERS
    assert int(ro_anim.wire_fields(o)["hit"][MAX_PLAYERS]) <= -2


def test_monster_kill_credit_is_the_top_damager(monkeypatch):
    from tests.test_combat import _world as pve_world
    _fake_equipped(monkeypatch, weapon_atk=1000)
    w = pve_world(player_pos=(20, 20), monster_pos=(21, 20))
    p = w["players"]
    w = {**w, "players": {**p, "pos": p["pos"].at[1].set(jnp.array([22, 20])), "active": p["active"].at[1].set(True)}}
    mv = w["mvp"]
    # slot 1 already dealt more cumulative damage than one hit of slot 0 can add
    w = {**w, "mvp": {**mv, "credit_mon": mv["credit_mon"].at[1, 0].set(0), "credit_dmg": mv["credit_dmg"].at[1, 0].set(10 ** 6),
                      "top_damage": mv["top_damage"].at[0].set(10 ** 6), "top_damager": mv["top_damager"].at[0].set(1)}}
    w = {**w, "monsters": {**w["monsters"], "hp": w["monsters"]["hp"].at[0].set(5)}}
    for s in range(30):
        o = combat(_seed(w, s), _attack(0))
        if not bool(o["monsters"]["active"][0]):
            break
    assert not bool(o["monsters"]["active"][0])
    assert int(o["players"]["kills"][1]) == 1 and int(o["players"]["kills"][0]) == 0


def test_jit_and_vmap_with_pvp(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=100)
    ws = jax.tree_util.tree_map(lambda *x: jnp.stack(x), _seed(_world(), 1), _seed(_world(), 2))
    inp = _attack(0, 1)
    out = jax.jit(jax.vmap(combat, in_axes=(0, None)))(ws, inp)
    assert out["players"]["hp"].shape[0] == 2
    o2 = jax.jit(ro_pvp.phase)(_world(pos0=TOWN, pos1=FIELD), {"pvp": jnp.ones_like(inp["attack"])})
    assert bool(o2["pvp"]["pk"][1]) and not bool(o2["pvp"]["pk"][0])


def test_more_candidates_than_pvp_slots_is_safe(monkeypatch):
    monkeypatch.setattr(ro_pvp, "PVP_SLOTS", 2)
    _fake_equipped(monkeypatch, weapon_atk=100)
    w = _world()
    p = w["players"]
    w = {**w, "players": {**p, "pos": p["pos"].at[2].set(jnp.array([A0[0], A0[1] + 1])),
                          "active": p["active"].at[2].set(True)}}
    for s in range(5):
        o = combat(_seed(w, s), _attack(0, 1, 2))
        assert int(o["players"]["hp"][2]) == int(w["players"]["hp"][2])  # slot 2 is beyond the list: skipped
        assert int(o["players"]["attack_cd"][2]) == 0  # and never swung


# --- wire ---------------------------------------------------------------------------------------------------------

def test_protocol_pvp_message_wire_hit_and_you_fields():
    from net import protocol
    assert protocol.validate_client({"t": "pvp"}) == {"t": "pvp"}
    assert protocol.wire_hit(-1) == 0 and protocol.wire_hit(-7) == -7 and protocol.wire_hit(12) == 12
    pv = {"pk": np.array([True, False]), "pvp_kills": np.array([3, 0]), "pvp_deaths": np.array([1, 2])}
    assert protocol._pvp_you({"pvp": pv}, 0) == {"pvp": True, "pvp_kills": 3, "pvp_deaths": 1}
    assert protocol._pvp_you({}, 0) == {}
    pw = protocol.pvp_world()
    assert pw["arena"] == wl.arena_box().tolist() and len(pw["zones"]) == wl.N_ZONES


def test_entity_rows_send_crit_negative_and_miss_zero():
    from net import protocol
    n = 3
    tab = {"pos": np.zeros((n, 2), int), "hp": np.ones(n, int), "max_hp": np.ones(n, int), "kind": np.ones(n, int),
           "action": np.zeros(n, int), "face": np.zeros(n, int), "start_tick": np.zeros(n, int),
           "dur_ms": np.zeros(n, int), "speed_ms": np.zeros(n, int), "hit": np.array([-1, -9, 4])}
    rows = protocol.entity_rows(np.arange(n), tab, {})
    assert [r["h"] for r in rows] == [0, -9, 4]


# --- bot policy ---------------------------------------------------------------------------------------------------

def test_duellists_walk_from_the_temple_into_the_arena_and_attack():
    from bots import ro_pvp_policy as pol
    from bots.ro_nav_policy import DELTAS
    ter = map_gen.make_terrain_ro(jax.random.PRNGKey(0))
    w = make_empty_world(jax.random.PRNGKey(0))
    temples = np.asarray(wl.temple_tiles())[0]
    slots = [0, 10, 20, 30]
    p = w["players"]
    pos = p["pos"]
    for k, s in enumerate(slots):
        pos = pos.at[s].set(jnp.asarray(temples[k * 37 % len(temples)]))
    act = p["active"].at[jnp.asarray(slots)].set(True)
    w = {**w, "terrain": ter, "walls": map_gen.blocking(ter), "players": {**p, "pos": pos, "active": act}}
    moves = jax.jit(pol.duel_moves)
    for _ in range(80):  # no occupancy here: a plain walk of the codes
        mv, attack, mask = moves(w)
        new = w["players"]["pos"] + DELTAS[mv]
        w = {**w, "players": {**w["players"], "pos": new}}
    inside = np.asarray(wl.in_arena(w["players"]["pos"]))[slots]
    assert inside.all()
    assert np.asarray(mask)[slots].all() and not bool(mask[1])
    assert np.asarray(attack)[slots].all()


def test_overrides_only_touch_duellists():
    from bots import ro_pvp_policy as pol
    w = _world()
    inp = {**empty_inputs(), "move": jnp.full((empty_inputs()["move"].shape[0],), 3, jnp.int32)}
    out = pol.pvp_overrides(w, inp)
    assert int(out["move"][1]) == 3  # slot 1 is not a duellist
    assert bool(out["attack"][0])     # slot 0 is in the arena with a foe adjacent
    assert int(out["move"][0]) == 0
