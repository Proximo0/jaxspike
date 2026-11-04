"""bots.ro_stats_policy.bot_allot: the allot plan, DEX included (ro5-balance)."""
import jax.numpy as jnp
import numpy as np

from bots import ro_stats_policy as sp
from game.config import MAX_PLAYERS
from game.data.stats import STAT_COST, STAT_POINTS_PER_LEVEL, STAT_POINTS_START

_KEYS = ("str", "agi", "vit", "int", "dex", "luk")


def _world(job=0, points=STAT_POINTS_START, level=1, **stats):
    one = jnp.ones((MAX_PLAYERS,), jnp.int32)
    s = {k: one * stats.get(k, 1) for k in _KEYS}
    s.update(job=one * job, stat_points=one * points, base_level=one * level)
    return {"stats": s, "players": {"active": jnp.zeros((MAX_PLAYERS,), jnp.bool_).at[0].set(True)}}


def _spend(job, points, level=1, v=None):
    """Apply bot_allot to slot 0 until it stops; return the final stats of slot 0."""
    v = dict(v or {k: 1 for k in _KEYS})
    for _ in range(400):
        a = int(sp.bot_allot(_world(job, points, level, **v), None)[0])
        if a < 0:
            break
        k = _KEYS[a]
        points -= STAT_COST[v[k]]
        v[k] += 1
    assert points >= 0
    return v


def _level_up(job, top):
    """Stats after allotting level by level from a fresh character to base level `top`."""
    v = _spend(job, STAT_POINTS_START)
    for lv in range(2, top + 1):
        v = _spend(job, int(STAT_POINTS_PER_LEVEL[lv]) + 1, lv, v)  # +1: leftovers are not tracked here
    return v


def test_fresh_novice_keeps_its_strength_build():
    v = _spend(0, STAT_POINTS_START)
    assert (v["str"], v["vit"], v["dex"]) == (18, 4, 1)  # DEX = base level 1: level-1 damage unchanged


def test_dex_follows_base_level():
    for top in (5, 10, 20):
        v = _level_up(0, top)
        assert v["dex"] >= top - 1 and v["str"] > v["vit"] > 1, (top, v)
    lo, hi = _level_up(0, 5), _level_up(0, 10)
    assert hi["dex"] > lo["dex"] and hi["str"] >= lo["str"]


def test_dex_is_raised_before_the_primary():
    w = _world(0, 10, level=6, str=20, vit=5, dex=3)
    assert int(sp.bot_allot(w, None)[0]) == sp.STAT_DEX


def test_jobs_with_dex_in_build_keep_their_plan():
    mage = 2  # JOB_BUILD (INT, DEX): no extra DEX rule
    assert sp.JOB_BUILD[mage] == (3, 4) and int(sp._DEX_PCT[mage]) == 0
    v = _spend(mage, STAT_POINTS_START, level=30)
    assert v["int"] > v["dex"] > 1 and v["str"] == 1


def test_inactive_and_broke_slots_allot_nothing():
    assert (np.asarray(sp.bot_allot(_world(0, 0, level=10), None)) == -1).all()
    assert (np.asarray(sp.bot_allot(_world(0), None))[1:] == -1).all()
