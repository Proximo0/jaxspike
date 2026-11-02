import jax
import jax.numpy as jnp

from game import ro_items, ro_stats
from game.combat import combat
from game.config import MONSTER_ATTACK_COOLDOWN, MONSTER_HP, MONSTER_RESPAWN_TICKS
from game.data import monsters as mdata
from game.state import empty_inputs, make_empty_world

ORC = 1  # Orc Warrior: aggro, atk 6..9, def 5, level 14


def _world(player_pos=(20, 20), monster_pos=(21, 20), kind=0):
    w = make_empty_world(jax.random.PRNGKey(0))
    p, m = w["players"], w["monsters"]
    p = {**p, "pos": p["pos"].at[0].set(jnp.array(player_pos)),
         "spawn": p["spawn"].at[0].set(jnp.array([30, 30])),
         "active": p["active"].at[0].set(True)}
    m = {**m, "pos": m["pos"].at[0].set(jnp.array(monster_pos)),
         "hp": m["hp"].at[0].set(MONSTER_HP[kind]),
         "kind": m["kind"].at[0].set(kind),
         "active": m["active"].at[0].set(True)}
    return {**w, "players": p, "monsters": m}


def _seed(w, s):
    return {**w, "rng": jax.random.key_data(jax.random.PRNGKey(s))}


def _stat(w, **kv):
    return {**w, "stats": {**w["stats"], **{k: w["stats"][k].at[0].set(v) for k, v in kv.items()}}}


def _fake_equipped(monkeypatch, weapon_atk=0, weapon_elem=0, armor_def=0):
    real = ro_items.equipped

    def fake(world):
        e = real(world)
        return {**e, "weapon_atk": e["weapon_atk"] + weapon_atk, "weapon_elem": e["weapon_elem"] + weapon_elem,
                "armor_def": e["armor_def"] + armor_def}

    monkeypatch.setattr(ro_items, "equipped", fake)


def _fresh():
    """Derived values of player 0, a fresh Novice."""
    return {k: int(v[0]) for k, v in ro_stats.derived(make_empty_world(jax.random.PRNGKey(0))).items()}


def _dmg_to_monster(w, seeds=range(30)):
    """Damage player 0 deals to monster 0 with one attack, per rng seed."""
    out = []
    for s in seeds:
        o = combat(_seed(w, s), _attack())
        out.append(int(w["monsters"]["hp"][0]) - int(o["monsters"]["hp"][0]))
    return out


def _dmg_to_player(w, seeds=range(30)):
    out = []
    for s in seeds:
        o = combat(_seed(w, s), empty_inputs())
        out.append(int(w["players"]["hp"][0]) - int(o["players"]["hp"][0]))
    return out


def _attack():
    i = empty_inputs()
    return {**i, "attack": i["attack"].at[0].set(True)}


def test_player_hit_in_damage_range_and_sets_aspd_cooldown(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=100)  # a Novice with a weapon: atk = derived atk + 100
    w = _world()
    atk = _fresh()["atk"] + 100
    dmgs = _dmg_to_monster(w)
    hits = [d for d in dmgs if d > 0]
    assert len(hits) >= 20  # 95% hit chance vs a level 1 Poring
    for d in hits:
        assert atk * 90 // 100 <= d <= atk * 110 // 100 * 2  # roll, doubled on a crit
    o = combat(_seed(w, 0), _attack())
    assert int(o["players"]["attack_cd"][0]) == _fresh()["aspd_ticks"] - 1


def test_kill_gets_credit_and_flags(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=1000)
    w = _world()
    w = {**w, "monsters": {**w["monsters"], "hp": w["monsters"]["hp"].at[0].set(5)}}
    for s in range(30):
        o = combat(_seed(w, s), _attack())
        if not bool(o["monsters"]["active"][0]):
            break
    assert not bool(o["monsters"]["active"][0])
    assert int(o["players"]["kills"][0]) == 1
    assert bool(o["monsters"]["died"][0])
    assert int(o["monsters"]["respawn_in"][0]) == MONSTER_RESPAWN_TICKS


def test_hit_reports_damage_to_mvp_ledger(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=100)
    w = _world()
    for s in range(30):
        o = combat(_seed(w, s), _attack())
        if int(o["monsters"]["hp"][0]) < int(w["monsters"]["hp"][0]):
            break
    dealt = int(w["monsters"]["hp"][0]) - int(o["monsters"]["hp"][0])
    assert dealt > 0
    assert int(o["mvp"]["damage_ledger"][0]) == dealt
    assert int(o["mvp"]["top_damager"][0]) == 0


def test_aspd_cooldown_sets_attack_period():
    # One attack every aspd_ticks ticks while the button is held; a monster with huge hp never dies.
    for agi in (1, 80):
        w = _stat(_world(), agi=agi)
        w = {**w, "monsters": {**w["monsters"], "hp": w["monsters"]["hp"].at[0].set(10 ** 6)}}
        ticks = ro_stats.derived(w)["aspd_ticks"][0].item()
        swings = []
        for t in range(3 * ticks + 1):
            w = combat(_seed(w, t), _attack())
            swings.append(int(w["players"]["attack_cd"][0]) == ticks - 1)
        assert [i for i, x in enumerate(swings) if x] == [0, ticks, 2 * ticks, 3 * ticks]
    assert ro_stats.derived(_stat(_world(), agi=80))["aspd_ticks"][0] < _fresh()["aspd_ticks"]


def test_cooldown_blocks_second_attack(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=100)
    w = _seed(_world(), 0)
    w1 = combat(w, _attack())
    assert int(w1["monsters"]["hp"][0]) < MONSTER_HP[0]
    w2 = combat(w1, _attack())
    assert int(w2["monsters"]["hp"][0]) == int(w1["monsters"]["hp"][0])
    assert int(w2["players"]["attack_cd"][0]) == _fresh()["aspd_ticks"] - 2


def test_element_multiplier_applies(monkeypatch):
    # Poring is Water; a Wind weapon deals 175%, a Neutral one 100%. Same seed = same hit, crit and roll.
    assert int(mdata.ELEMENT[mdata.KIND_BY_NAME["Poring"]]) == 1
    _fake_equipped(monkeypatch, weapon_atk=1000, weapon_elem=4)
    wind = _dmg_to_monster(_world())
    monkeypatch.undo()
    _fake_equipped(monkeypatch, weapon_atk=1000, weapon_elem=0)
    neutral = _dmg_to_monster(_world())
    pairs = [(a, b) for a, b in zip(wind, neutral) if a and b]
    assert len(pairs) >= 20
    for a, b in pairs:
        assert abs(a * 100 - b * 175) <= 175 * 2


def test_monster_def_reduces_damage(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=1000)
    w = _world(kind=ORC)
    real = _dmg_to_monster(w)
    monkeypatch.setattr(mdata, "DEF", jnp.zeros_like(mdata.DEF))
    flat = _dmg_to_monster(w)
    pairs = [(a, b) for a, b in zip(real, flat) if a and b]
    assert len(pairs) >= 10
    assert all(a <= b for a, b in pairs)
    assert sum(1 for a, b in pairs if a < b) >= 5  # crits (x2, ignore DEF) are the equal ones


def test_hit_chance_falls_with_monster_level():
    from game.combat import hit_percent_player
    hit = _fresh()["hit"]
    low = int(hit_percent_player(hit, mdata.LEVEL[0]))
    high = int(hit_percent_player(hit, mdata.LEVEL[mdata.KIND_BY_NAME["Golden Thief Bug"]]))
    assert low == 95 and high == 5
    assert int(hit_percent_player(hit + 100, 35)) > high


def test_player_misses_and_a_miss_costs_the_cooldown():
    w = _world(kind=mdata.KIND_BY_NAME["Golden Thief Bug"])  # level 35: 5% hit chance for a Novice
    outs = [combat(_seed(w, s), _attack()) for s in range(20)]
    assert sum(int(o["monsters"]["hp"][0]) < int(w["monsters"]["hp"][0]) for o in outs) <= 6
    assert all(int(o["players"]["attack_cd"][0]) == _fresh()["aspd_ticks"] - 1 for o in outs)


def test_not_adjacent_no_attack():
    w = combat(_world(monster_pos=(23, 20)), _attack())
    assert int(w["monsters"]["hp"][0]) == MONSTER_HP[0]


def test_monster_hits_adjacent_player_in_atk_range():
    w = _world(kind=ORC)
    hits = [d for d in _dmg_to_player(w) if d]
    assert len(hits) >= 20  # 95% against a fresh Novice
    assert all(int(mdata.ATK_MIN[ORC]) <= d <= int(mdata.ATK_MAX[ORC]) for d in hits)
    assert len(set(hits)) > 1  # uniform in a range, not a constant
    o = combat(w, empty_inputs())
    assert int(o["monsters"]["attack_cd"][0]) == MONSTER_ATTACK_COOLDOWN
    hp1 = int(o["players"]["hp"][0])
    o2 = combat(o, empty_inputs())
    assert int(o2["players"]["hp"][0]) == hp1  # on cooldown


def test_player_def_reduces_monster_damage(monkeypatch):
    w = _stat(_world(kind=ORC), vit=120)  # soft def far above the 80% cap
    hits = [d for d in _dmg_to_player(w) if d]
    assert hits and max(hits) <= 2  # 9 * 20 / 100 = 1, min 1
    _fake_equipped(monkeypatch, armor_def=50)
    hits = [d for d in _dmg_to_player(_world(kind=ORC)) if d]
    assert hits and max(hits) <= int(mdata.ATK_MAX[ORC]) * 50 // 100


def test_monster_flee_miss():
    from game.combat import hit_percent_monster
    assert int(hit_percent_monster(14, 102)) == 95
    assert int(hit_percent_monster(1, 300)) == 5
    w = _stat(_world(kind=mdata.KIND_BY_NAME["Lunatic"]), agi=200)  # flee 300
    w = {**w, "monsters": {**w["monsters"], "hp": w["monsters"]["hp"].at[0].set(1)}}
    assert sum(1 for d in _dmg_to_player(w) if d) <= 6


def test_passive_monster_only_fights_back():
    poring = _world(kind=0)
    assert not bool(mdata.AGGRO[0])
    assert all(d == 0 for d in _dmg_to_player(poring))  # full hp: never starts a fight
    hurt = {**poring, "monsters": {**poring["monsters"], "hp": poring["monsters"]["hp"].at[0].set(10)}}
    assert any(d > 0 for d in _dmg_to_player(hurt))
    assert any(d > 0 for d in _dmg_to_player(_world(kind=ORC)))


def test_monster_cannot_hit_player_on_pz():
    w = _world(kind=ORC)
    w = {**w, "pz": w["pz"].at[20, 20].set(True)}
    assert all(d == 0 for d in _dmg_to_player(w, range(10)))
    assert int(combat(w, empty_inputs())["monsters"]["attack_cd"][0]) == 0


def test_player_on_pz_can_still_attack(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=100)
    w = _world()
    w = {**w, "pz": w["pz"].at[20, 20].set(True)}
    assert max(_dmg_to_monster(w)) > 0


def test_player_death_counts():
    w = _seed(_world(kind=ORC), 1)  # seed 1: the Orc's swing lands
    w = {**w, "players": {**w["players"], "hp": w["players"]["hp"].at[0].set(1)}}
    w = combat(w, empty_inputs())
    assert bool(w["players"]["dead"][0])
    assert int(w["players"]["deaths"][0]) == 1
    assert bool(w["players"]["active"][0])
    w = combat(w, empty_inputs())  # respawn, no double count
    assert int(w["players"]["deaths"][0]) == 1


def test_dead_player_respawns():
    w = _world(monster_pos=(100, 100))
    p = w["players"]
    p = {**p, "dead": p["dead"].at[0].set(True), "hp": p["hp"].at[0].set(0)}
    w = combat({**w, "players": p}, empty_inputs())
    assert not bool(w["players"]["dead"][0])
    assert int(w["players"]["hp"][0]) == _fresh()["max_hp"]
    assert w["players"]["pos"][0].tolist() == [30, 30]


def _two_dead(second_pos, second_dead):
    """Player 0 dead with spawn (30, 30); player 1 at second_pos, dead or living."""
    w = _world(monster_pos=(100, 100))
    p = w["players"]
    p = {**p, "dead": p["dead"].at[0].set(True).at[1].set(second_dead),
         "hp": p["hp"].at[0].set(0),
         "pos": p["pos"].at[1].set(jnp.array(second_pos)),
         "spawn": p["spawn"].at[1].set(jnp.array([40, 40])),
         "active": p["active"].at[1].set(True)}
    return {**w, "players": p}


def test_respawn_blocked_by_living_player_on_spawn():
    w = combat(_two_dead((30, 30), False), empty_inputs())
    assert bool(w["players"]["dead"][0])
    assert w["players"]["pos"][0].tolist() == [20, 20]


def test_respawn_allowed_onto_waiting_dead_player():
    # Dead players are not physical: player 1 is dead on player 0's spawn tile, and its own
    # spawn (40, 40) is free, so both come back; neither blocks the other.
    w = combat(_two_dead((30, 30), True), empty_inputs())
    assert not bool(w["players"]["dead"][0]) and not bool(w["players"]["dead"][1])
    assert w["players"]["pos"][0].tolist() == [30, 30]
    assert w["players"]["pos"][1].tolist() == [40, 40]


def test_jit_and_vmap(monkeypatch):
    _fake_equipped(monkeypatch, weapon_atk=100)
    w = _seed(_world(), 0)
    batch = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), w)
    inp = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), _attack())
    out = jax.jit(jax.vmap(combat))(batch, inp)
    single = jax.jit(combat)(w, _attack())
    eager = combat(w, _attack())
    assert int(single["monsters"]["hp"][0]) == int(eager["monsters"]["hp"][0])
    assert out["monsters"]["hp"][:, 0].tolist() == [int(eager["monsters"]["hp"][0])] * 2


def test_monster_hits_reachable_player_not_pz_neighbour():
    w = _world(kind=ORC)
    p = w["players"]
    p = {**p, "pos": p["pos"].at[1].set(jnp.array([21, 21])),
         "spawn": p["spawn"].at[1].set(jnp.array([40, 40])),
         "active": p["active"].at[1].set(True)}
    w = {**w, "players": p, "pz": w["pz"].at[20, 20].set(True)}
    for s in range(10):
        o = combat(_seed(w, s), empty_inputs())
        assert int(o["players"]["hp"][0]) == int(w["players"]["hp"][0])
        if int(o["players"]["hp"][1]) < int(w["players"]["hp"][1]):
            return
    raise AssertionError("monster never hit the reachable player")
