"""Playtest cheats (net/cheats.py, server --cheats): validation, every op through a real server tick, the refusal
with cheats off, and god mode refilling HP each tick."""
import asyncio
import json

import numpy as np

from game import ro_stats
from game.data import items as item_data
from game.data import skills as skill_data
from game.data import stats as stats_data
from game.world_layout import TOWN_NAMES, temple_tiles
from net import cheats
from net.protocol import validate_client
from net.server import Client, GameServer


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(json.loads(text))

    async def close(self):
        pass


def _server(tmp_path, cheats_on=True):
    s = GameServer(seed=0, chars_dir=tmp_path, tick_ms=50, cheats=cheats_on)
    s.prepare()
    return s


def _human(s, name):
    slot, err = s.login(name)
    assert err is None
    ws = FakeWS()
    s.conn[slot] = Client(ws)
    return slot, ws


def _tick(s, n=1):
    async def body():
        for _ in range(n):
            s.tick_once()
            await asyncio.sleep(0.005)
    asyncio.run(body())


def _cheat(s, slot, **msg):
    m = validate_client({"t": "cheat", **msg})
    assert m is not None, msg
    s._on_message(slot, m)


def _notices(ws):
    return [m["notice"] for m in ws.sent if m.get("t") == "snap" and m.get("notice")]


def _inv(s, slot):
    return (np.asarray(s.world["players"]["inv"][slot]), np.asarray(s.world["items"]["inv_count"][slot]),
            np.asarray(s.world["items"]["inv_refine"][slot]))


# ---- validation -------------------------------------------------------------------------------------------------
def test_validation_accepts_each_op_and_rejects_bad_fields():
    good = [{"op": "level", "base": 50, "job": 0}, {"op": "ryn", "amount": 100000},
            {"op": "item", "item": item_data.ITEM_BLADE, "count": 1, "refine": 7}, {"op": "heal"},
            {"op": "teleport", "town": 1}, {"op": "kit"}, {"op": "allskills"}, {"op": "godmode", "on": True}]
    for g in good:
        assert validate_client({"t": "cheat", **g}) == {"t": "cheat", **g}, g
    bad = [{"op": "level", "base": 100, "job": 1}, {"op": "level", "base": "5", "job": 1}, {"op": "level", "base": 5},
           {"op": "ryn", "amount": 3_000_000_000}, {"op": "ryn", "amount": True},
           {"op": "item", "item": 0, "count": 1, "refine": 0}, {"op": "item", "item": 99, "count": 1, "refine": 0},
           {"op": "item", "item": 3, "count": 0, "refine": 0}, {"op": "item", "item": 3, "count": 1, "refine": 11},
           {"op": "teleport", "town": len(TOWN_NAMES)}, {"op": "godmode", "on": 1}, {"op": "nuke"}, {}]
    for b in bad:
        assert validate_client({"t": "cheat", **b}) is None, b
    assert validate_client({"t": "cheat", "op": "heal", "extra": 1}) == {"t": "cheat", "op": "heal"}


def test_kit_table_has_a_job_weapon_and_the_supplies():
    for job in range(stats_data.NUM_JOBS):
        kit = cheats.kit_items(job)
        weapon = [i for i, _, eq in kit if eq == item_data.EQ_WEAPON]
        assert len(weapon) == 1 and bool(item_data.EQUIP_JOBS[weapon[0], job]), job
        assert all(bool(item_data.EQUIP_JOBS[i, job]) for i, _, eq in kit if eq >= 0)
        assert (item_data.ITEM_RED_POTION, 20, -1) in kit and (item_data.ITEM_BLUE_POTION, 10, -1) in kit
        assert (item_data.ITEM_FLIT_FEATHER, 100, -1) in kit
    sw = dict((eq, i) for i, _, eq in cheats.kit_items(stats_data.JOB_SWORDSMAN) if eq >= 0)
    assert item_data.CLASS[sw[item_data.EQ_WEAPON]] == item_data.WC_SWORD
    assert sw[item_data.EQ_SHIELD] == item_data.ITEM_SHIELD and sw[item_data.EQ_HEAD] == item_data.ITEM_HELM
    arc = dict((eq, i) for i, _, eq in cheats.kit_items(stats_data.JOB_ARCHER) if eq >= 0)
    assert item_data.EQ_SHIELD not in arc  # a bow takes both hands


# ---- each op through a real tick --------------------------------------------------------------------------------
def test_cheats_off_refuses_with_a_notice(tmp_path):
    s = _server(tmp_path, cheats_on=False)
    slot, ws = _human(s, "nocheat")
    _cheat(s, slot, op="ryn", amount=5000)
    _cheat(s, slot, op="godmode", on=True)
    _tick(s, 2)
    assert int(s.world["items"]["ryn"][slot]) == 0
    assert not s.godmode and not s.cheat_queue
    assert "cheats are off" in _notices(ws)


def test_level_ryn_heal_and_teleport(tmp_path):
    s = _server(tmp_path)
    slot, ws = _human(s, "cheater")
    st0 = int(s.world["stats"]["stat_points"][slot])
    _cheat(s, slot, op="level", base=50, job=50)
    _cheat(s, slot, op="ryn", amount=100000)
    _tick(s)
    st = s.world["stats"]
    assert int(st["base_level"][slot]) == 50
    assert int(st["job_level"][slot]) == stats_data.MAX_JOB_LEVEL_NOVICE  # a Novice's job cap
    cum = np.cumsum(stats_data.STAT_POINTS_PER_LEVEL)
    assert int(st["stat_points"][slot]) == st0 + int(cum[50] - cum[1])
    assert int(s.world["items"]["ryn"][slot]) == 100000
    max_hp = int(ro_stats.derived(s.world)["max_hp"][slot])
    assert max_hp > 100  # derived follows the new level; hp itself is untouched until healed
    assert int(s.world["players"]["hp"][slot]) < max_hp
    _cheat(s, slot, op="heal")
    _cheat(s, slot, op="teleport", town=2)
    _tick(s)
    assert int(s.world["players"]["hp"][slot]) == int(ro_stats.derived(s.world)["max_hp"][slot])
    pos = tuple(int(v) for v in np.asarray(s.world["players"]["pos"][slot]))
    assert pos in {tuple(int(v) for v in t) for t in np.asarray(temple_tiles())[2]}
    _tick(s)
    n = _notices(ws)
    assert "level 50 / job 10; ryn 100,000" in n and f"fully healed; teleported to {TOWN_NAMES[2]}" in n


def test_item_gear_and_stack(tmp_path):
    s = _server(tmp_path)
    slot, _ = _human(s, "itemer")
    inv0, _, _ = _inv(s, slot)
    free0 = int((inv0 == 0).sum())
    _cheat(s, slot, op="item", item=item_data.ITEM_BLADE, count=2, refine=7)
    _cheat(s, slot, op="item", item=item_data.ITEM_RED_POTION, count=30, refine=5)
    _tick(s)
    _cheat(s, slot, op="item", item=item_data.ITEM_RED_POTION, count=30, refine=0)
    _tick(s, 2)
    inv, cnt, ref = _inv(s, slot)
    blades = np.flatnonzero(inv == item_data.ITEM_BLADE)
    assert len(blades) == 2 and (ref[blades] == 7).all() and (cnt[blades] == 1).all()
    reds = np.flatnonzero(inv == item_data.ITEM_RED_POTION)
    assert len(reds) == 1 and ref[reds[0]] == 0  # one merged stack, no refine on a potion
    assert int((inv == 0).sum()) == free0 - 3 + (1 if (inv0 == item_data.ITEM_RED_POTION).any() else 0)
    assert cnt[reds[0]] >= 60


def test_kit_is_worn_and_counts(tmp_path):
    s = _server(tmp_path)
    slot, _ = _human(s, "kitter")
    _cheat(s, slot, op="kit")
    _tick(s, 2)
    inv, cnt, _ = _inv(s, slot)
    equip = np.asarray(s.world["items"]["equip"][slot])
    for item, n, eq in cheats.kit_items(stats_data.JOB_NOVICE):
        where = np.flatnonzero(inv == item)
        assert len(where) >= 1, item
        if eq >= 0:
            assert inv[equip[eq]] == item, (item, eq)
        else:
            assert cnt[where[0]] >= n


def test_allskills_learns_job_skills_and_grants_points(tmp_path):
    s = _server(tmp_path)
    slot, _ = _human(s, "skiller")
    _cheat(s, slot, op="level", base=40, job=10)
    _tick(s)
    s.world["stats"]["job"] = s.world["stats"]["job"].at[slot].set(stats_data.JOB_MAGE)
    s.world["stats"]["stat_points"] = s.world["stats"]["stat_points"].at[slot].set(0)
    _cheat(s, slot, op="allskills")
    _tick(s)
    learned = np.asarray(s.world["skills"]["learned"][slot])
    sj, mx = np.asarray(skill_data.SKILL_JOB), np.asarray(skill_data.SKILL_MAX_LEVEL)
    mine = ((sj == skill_data.JOB_NOVICE) | (sj == stats_data.JOB_MAGE)) & (np.arange(len(sj)) > 0)
    assert (learned[mine] == mx[mine]).all()
    assert (learned[~mine] == 0).all()
    cum = np.cumsum(stats_data.STAT_POINTS_PER_LEVEL)
    assert int(s.world["stats"]["stat_points"][slot]) == stats_data.STAT_POINTS_START + int(cum[40] - cum[1])


def test_godmode_refills_hp_every_tick_and_turns_off(tmp_path):
    s = _server(tmp_path)
    slot, ws = _human(s, "godly")
    _cheat(s, slot, op="godmode", on=True)
    _tick(s)
    assert slot in s.godmode
    for _ in range(3):
        s.world["players"]["hp"] = s.world["players"]["hp"].at[slot].set(1)
        _tick(s)
        assert int(s.world["players"]["hp"][slot]) == int(ro_stats.derived(s.world)["max_hp"][slot])
    _cheat(s, slot, op="godmode", on=False)
    _tick(s)
    s.world["players"]["hp"] = s.world["players"]["hp"].at[slot].set(1)
    _tick(s)
    assert int(s.world["players"]["hp"][slot]) < int(ro_stats.derived(s.world)["max_hp"][slot])
    assert "god mode on" in _notices(ws) and "god mode off" in _notices(ws)
    _cheat(s, slot, op="godmode", on=True)
    _tick(s)
    s.conn.pop(slot)  # no event loop here to close a socket
    s.logout(slot)
    assert slot not in s.godmode  # the next owner of the slot is mortal
