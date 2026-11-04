"""RO fidelity: the data tables match classic pre-renewal Ragnarok Online (docs/ro_fidelity.md cites each source).

Sources: rAthena db/pre-re (job_exp.yml, job_basepoints.yml, job_aspd.yml, attr_fix.yml, mob_db.yml, item_db_*.yml,
skill_db.yml) and iRO wiki classic pages. Tests marked xfail check engine formulas the lead / owners still have to
change (the exact lines are in docs/ro_fidelity.md); they turn into passes when the engine is fixed.
"""
import jax
import numpy as np
import pytest

from game.config import TICK_MS
from game.data import elements as el
from game.data import items as I
from game.data import monsters as M
from game.data import skills as sk
from game.data import stats as S

NOV, SWD, MAG, ARC, ACO, THF, MER = range(7)


# ---- exp, status points, HP/SP ---------------------------------------------------------------------------------
def test_base_exp_table_matches_rathena():
    t = S.BASE_EXP_TABLE
    assert (t[1], t[2], t[10], t[20], t[50], t[70], t[90], t[98], t[99]) == (
        9, 16, 320, 1620, 115254, 1473746, 9738720, 99999998, 0)
    assert sum(t[1:10]) == 881  # exp from level 1 to 10


def test_job_exp_tables_match_rathena():
    assert S.JOB_EXP_NOVICE_TABLE[1:10] == (10, 18, 28, 40, 91, 151, 205, 268, 340)
    f = S.JOB_EXP_FIRST_TABLE
    assert (f[1], f[10], f[40], f[49], f[50]) == (30, 520, 125915, 509596, 0)
    assert S.MAX_JOB_LEVEL == (10, 50, 50, 50, 50, 50, 50)


def test_status_points_and_costs():
    assert S.STAT_POINTS_START == 48
    assert S.STAT_POINTS_START + sum(S.STAT_POINTS_PER_LEVEL[2:100]) == 1273  # RO total at base 99
    assert (S.STAT_POINTS_PER_LEVEL[2], S.STAT_POINTS_PER_LEVEL[6], S.STAT_POINTS_PER_LEVEL[99]) == (3, 4, 22)
    assert (S.STAT_COST[1], S.STAT_COST[10], S.STAT_COST[11], S.STAT_COST[50], S.STAT_COST[98]) == (2, 2, 3, 6, 11)


@pytest.mark.parametrize("job,l1,l50,l99", [(NOV, 40, 285, 530), (SWD, 40, 1179, 3999), (MAG, 40, 670, 2020),
                                           (ARC, 40, 934, 3029), (ACO, 40, 795, 2510), (THF, 40, 934, 3029),
                                           (MER, 40, 795, 2510)])
def test_job_base_hp_matches_rathena(job, l1, l50, l99):
    row = S.JOB_BASE_HP_TABLE[job]
    assert (row[1], row[50], row[99]) == (l1, l50, l99)


def test_job_base_sp_matches_rathena():
    assert [S.JOB_BASE_SP_TABLE[j][1] for j in range(7)] == [11, 12, 16, 12, 15, 12, 13]
    assert [S.JOB_BASE_SP_TABLE[j][99] for j in range(7)] == [109, 208, 604, 208, 505, 208, 307]


def test_interim_hp_sp_factors_stay_close_to_exact():
    for j in range(1, 7):
        for lv in (10, 30, 60, 99):
            hp = S.BASE_HP_TABLE[lv] * S.JOB_HP_FACTOR[j] // 1000
            sp = S.BASE_SP_TABLE[lv] * S.JOB_SP_FACTOR[j] // 1000
            assert abs(hp - S.JOB_BASE_HP_TABLE[j][lv]) <= 0.2 * S.JOB_BASE_HP_TABLE[j][lv], (j, lv)
            assert abs(sp - S.JOB_BASE_SP_TABLE[j][lv]) <= 0.3 * S.JOB_BASE_SP_TABLE[j][lv], (j, lv)


def test_sp_regen_every_eight_seconds():
    assert abs(S.SP_REGEN_TICKS * TICK_MS - 8000) <= TICK_MS // 2
    assert abs(sk.SP_REGEN_TICKS * TICK_MS - 8000) <= TICK_MS // 2


# ---- ASPD --------------------------------------------------------------------------------------------------------
def test_weapon_aspd_matches_rathena_job_aspd():
    w = S.JOB_WEAPON_ASPD  # columns: fist, dagger, 1h sword, bow, staff, mace, 1h axe
    assert w[NOV][0] == 150 and w[NOV][1] == 135
    assert w[SWD][0] == 160 and w[SWD][2] == 145 and w[SWD][3] == 0
    assert w[MAG][4] == 130 and w[ARC][3] == 130 and w[ACO][5] == 140 and w[THF][1] == 150 and w[MER][6] == 130


def swings_per_s_x1000(aspd):
    """RO pre-re: amotion = (200 - ASPD) x 10 ms, attack delay = 2 x amotion, so swings/s = 50 / (200 - ASPD)."""
    return 50_000 // (200 - aspd)


def test_feel_target_swings_per_second():
    assert [swings_per_s_x1000(a) for a in (150, 170, 180, 190)] == [1000, 1666, 2500, 5000]


def _derived(job=NOV, level=1, **stats):
    from game import ro_stats
    from game.state import make_empty_world
    st = ro_stats.make_state()
    for k, v in {"job": job, "base_level": level, **stats}.items():
        st[k] = st[k].at[0].set(v)
    w = {**make_empty_world(jax.random.PRNGKey(0)), "stats": st}
    return {k: int(v[0]) for k, v in ro_stats.derived(w).items()}


def test_engine_attack_delay_is_two_amotions():
    # Thief bare-handed (fist 160), AGI 1 DEX 1: ASPD 200 - 40*995//1000 = 161 -> amotion 390 ms, delay 780 ms
    d = _derived(job=THF)
    assert d["aspd"] == 161 and d["aspd_ticks"] == (780 + TICK_MS // 2) // TICK_MS
    # with a knife (150): ASPD 151 -> amotion 490 ms, delay 980 ms
    from game import ro_stats
    from game.state import make_empty_world
    from game.ro_items import make_state as make_items
    st = ro_stats.make_state()
    st["job"] = st["job"].at[0].set(THF)
    w = {**make_empty_world(jax.random.PRNGKey(0)), "stats": st}
    it = make_items()
    it["equip"] = it["equip"].at[0, I.EQ_WEAPON].set(0)
    it["inv_count"] = it["inv_count"].at[0, 0].set(1)
    w["players"] = {**w["players"], "inv": w["players"]["inv"].at[0, 0].set(I.ITEM_KNIFE)}
    d = {k: int(v[0]) for k, v in ro_stats.derived({**w, "items": it}).items()}
    assert d["aspd"] == 151 and d["aspd_ticks"] == (980 + TICK_MS // 2) // TICK_MS


def test_engine_max_hp_exact():
    assert _derived(job=NOV, level=1)["max_hp"] == 40
    assert _derived(job=SWD, level=50, vit=30)["max_hp"] == 1179 * 130 // 100


def test_engine_hit_flee_pre_renewal():
    d = _derived(level=10, dex=20, agi=30)
    assert d["hit"] == 10 + 20 and d["flee"] == 10 + 30


# ---- skills ------------------------------------------------------------------------------------------------------
def _lv(a, s):
    return [int(x) for x in np.asarray(a[s, 1:int(sk.SKILL_MAX_LEVEL[s]) + 1])]


def _ms(ticks):
    return [t * TICK_MS for t in ticks]


def test_skill_list_covers_novice_and_first_jobs():
    want = {"Basic Skill", "First Aid", "Play Dead", "Bash", "Magnum Break", "Provoke", "Endure", "Sword Mastery",
            "HP Recovery", "Fire Bolt", "Cold Bolt", "Lightning Bolt", "Napalm Beat", "Soul Strike", "Fire Wall",
            "Frost Diver", "Stone Curse", "Double Strafe", "Arrow Shower", "Owl's Eye", "Vulture's Eye",
            "Improve Concentration", "Heal", "Blessing", "Increase AGI", "Angelus", "Cure", "Teleport",
            "Warp Portal", "Double Attack", "Steal", "Hiding", "Envenom", "Detoxify", "Improve Dodge", "Mammonite",
            "Discount", "Overcharge", "Pushcart", "Vending", "Item Appraisal"}
    assert want <= set(sk.SKILL_NAME)
    assert sk.N_SKILLS == len(sk.SKILL_NAME) == 42
    assert int(sk.SKILL_MAX_LEVEL[sk.SK_TELEPORT]) == 2 and int(sk.SKILL_MAX_LEVEL[sk.SK_WARP_PORTAL]) == 4


def test_swordsman_skills():
    assert _lv(sk.SKILL_SP, sk.SK_BASH) == [8] * 5 + [15] * 5
    assert _lv(sk.SKILL_POWER, sk.SK_BASH) == [1000 + 300 * n for n in range(1, 11)]  # 130% .. 400%
    assert _lv(sk.SKILL_POWER, sk.SK_MAGNUM_BREAK) == [1000 + 200 * n for n in range(1, 11)]
    assert int(sk.SKILL_AOE[sk.SK_MAGNUM_BREAK]) == 2 and _lv(sk.SKILL_AFTER_DELAY_MS, sk.SK_MAGNUM_BREAK)[0] == 2000
    assert _lv(sk.SKILL_SP, sk.SK_PROVOKE) == list(range(4, 14))
    assert _ms(_lv(sk.SKILL_COOLDOWN, sk.SK_PROVOKE))[0] == pytest.approx(1000, abs=TICK_MS)
    assert _ms(_lv(sk.SKILL_COOLDOWN, sk.SK_ENDURE))[0] == pytest.approx(10_000, abs=TICK_MS)
    assert _ms(_lv(sk.SKILL_BUFF_TICKS, sk.SK_ENDURE)) == pytest.approx(
        [7000 + 3000 * n for n in range(1, 11)], abs=TICK_MS)
    assert int(sk.SKILL_DELAY_ASPD[sk.SK_BASH]) == 1


def test_mage_skills():
    assert _lv(sk.SKILL_SP, sk.SK_FIRE_BOLT) == [12 + 2 * i for i in range(10)]
    assert _ms(_lv(sk.SKILL_CAST, sk.SK_COLD_BOLT)) == pytest.approx([700 * n for n in range(1, 11)], abs=TICK_MS)
    assert _lv(sk.SKILL_AFTER_DELAY_MS, sk.SK_LIGHTNING_BOLT) == [800 + 200 * n for n in range(1, 11)]
    assert _lv(sk.SKILL_SP, sk.SK_NAPALM_BEAT) == [9, 9, 9, 12, 12, 12, 15, 15, 15, 18]
    assert _lv(sk.SKILL_POWER, sk.SK_NAPALM_BEAT) == [700 + 100 * n for n in range(1, 11)]  # 80% .. 170%
    assert _lv(sk.SKILL_SP, sk.SK_SOUL_STRIKE) == [18, 14, 24, 20, 30, 26, 36, 32, 42, 38]
    assert _lv(sk.SKILL_POWER, sk.SK_SOUL_STRIKE) == [1000 * ((n + 1) // 2) for n in range(1, 11)]
    assert _ms(_lv(sk.SKILL_CAST, sk.SK_SOUL_STRIKE)) == pytest.approx([500] * 10, abs=TICK_MS)
    assert _ms(_lv(sk.SKILL_CAST, sk.SK_FIRE_WALL)) == pytest.approx(
        [2150 - 150 * n for n in range(1, 11)], abs=TICK_MS)
    assert _lv(sk.SKILL_SP, sk.SK_FROST_DIVER) == list(range(25, 15, -1))
    assert _lv(sk.SKILL_SP, sk.SK_STONE_CURSE) == list(range(25, 15, -1))
    assert _lv(sk.SKILL_RANGE, sk.SK_STONE_CURSE)[0] == 2


def test_archer_skills():
    assert _lv(sk.SKILL_SP, sk.SK_DOUBLE_STRAFE) == [12] * 10 and int(sk.SKILL_HITS[sk.SK_DOUBLE_STRAFE]) == 2
    assert _lv(sk.SKILL_POWER, sk.SK_DOUBLE_STRAFE) == [900 + 100 * n for n in range(1, 11)]  # 100% .. 190% x2
    assert _lv(sk.SKILL_POWER, sk.SK_ARROW_SHOWER) == [750 + 50 * n for n in range(1, 11)]   # 80% .. 125%
    assert int(sk.SKILL_AOE[sk.SK_ARROW_SHOWER]) == 2
    assert _lv(sk.SKILL_SP, sk.SK_IMPROVE_CONCENTRATION) == [20 + 5 * n for n in range(1, 11)]


def test_acolyte_skills():
    assert _lv(sk.SKILL_SP, sk.SK_HEAL) == [10 + 3 * n for n in range(1, 11)]
    assert _lv(sk.SKILL_CAST, sk.SK_HEAL) == [0] * 10 and _lv(sk.SKILL_AFTER_DELAY_MS, sk.SK_HEAL)[0] == 1000
    assert _lv(sk.SKILL_SP, sk.SK_BLESSING) == [24 + 4 * n for n in range(1, 11)]
    assert _ms(_lv(sk.SKILL_BUFF_TICKS, sk.SK_BLESSING)) == pytest.approx(
        [(40 + 20 * n) * 1000 for n in range(1, 11)], abs=TICK_MS)
    assert _lv(sk.SKILL_SP, sk.SK_INCREASE_AGI) == [15 + 3 * n for n in range(1, 11)]
    assert _ms(_lv(sk.SKILL_CAST, sk.SK_INCREASE_AGI))[0] == pytest.approx(1000, abs=TICK_MS)
    assert _lv(sk.SKILL_SP, sk.SK_ANGELUS) == [20 + 3 * n for n in range(1, 11)]
    assert _lv(sk.SKILL_AFTER_DELAY_MS, sk.SK_ANGELUS)[0] == 3500
    assert _lv(sk.SKILL_SP, sk.SK_CURE) == [15]
    assert _lv(sk.SKILL_SP, sk.SK_TELEPORT) == [10, 9] and _lv(sk.SKILL_SP, sk.SK_WARP_PORTAL) == [35, 32, 29, 26]
    assert int(sk.SKILL_ELEMENT[sk.SK_HEAL]) == el.ELEM_HOLY


def test_thief_and_merchant_skills():
    assert _lv(sk.SKILL_SP, sk.SK_STEAL) == [10] * 10 and _lv(sk.SKILL_RANGE, sk.SK_STEAL)[0] == 1
    assert _ms(_lv(sk.SKILL_BUFF_TICKS, sk.SK_HIDING)) == pytest.approx(
        [30_000 * n for n in range(1, 11)], abs=TICK_MS)
    assert _lv(sk.SKILL_SP, sk.SK_ENVENOM) == [12] * 10 and int(sk.SKILL_ELEMENT[sk.SK_ENVENOM]) == el.ELEM_POISON
    assert _lv(sk.SKILL_SP, sk.SK_DETOXIFY) == [10]
    assert _lv(sk.SKILL_SP, sk.SK_MAMMONITE) == [5] * 10
    assert _lv(sk.SKILL_POWER, sk.SK_MAMMONITE) == [1000 + 500 * n for n in range(1, 11)]  # 150% .. 600%
    assert _lv(sk.SKILL_SP, sk.SK_VENDING) == [30] * 10 and _lv(sk.SKILL_SP, sk.SK_ITEM_APPRAISAL) == [10]
    for s in (sk.SK_DOUBLE_ATTACK, sk.SK_IMPROVE_DODGE, sk.SK_DISCOUNT, sk.SK_OVERCHARGE, sk.SK_PUSHCART):
        assert int(sk.SKILL_EFFECT[s]) == sk.EFFECT_PASSIVE


def test_every_skill_use_has_a_minimum_delay():
    for s in range(1, sk.N_SKILLS):
        assert min(_lv(sk.SKILL_AFTER_DELAY_MS, s)) >= 100, sk.SKILL_NAME[s]


# ---- elements ----------------------------------------------------------------------------------------------------
def test_element_table_level_one_matches_attr_fix():
    E = el
    m = lambda a, d: int(E.ELEM_MULT[a, d])  # noqa: E731
    assert m(E.ELEM_WATER, E.ELEM_FIRE) == 150 and m(E.ELEM_FIRE, E.ELEM_EARTH) == 150
    assert m(E.ELEM_WIND, E.ELEM_WATER) == 175 and m(E.ELEM_EARTH, E.ELEM_WIND) == 150
    assert m(E.ELEM_EARTH, E.ELEM_POISON) == 100 and m(E.ELEM_HOLY, E.ELEM_UNDEAD) == 150
    assert m(E.ELEM_FIRE, E.ELEM_UNDEAD) == 125 and m(E.ELEM_GHOST, E.ELEM_SHADOW) == 75
    assert m(E.ELEM_UNDEAD, E.ELEM_UNDEAD) == 0 and m(E.ELEM_NEUTRAL, E.ELEM_GHOST) == 25
    assert m(E.ELEM_POISON, E.ELEM_UNDEAD) == 0  # RO -25 (heals), clamped


# ---- monsters ----------------------------------------------------------------------------------------------------
# name: (level, hp, atk_min, atk_max, def, base_exp, job_exp, element, size)   rAthena db/pre-re/mob_db.yml
RO_MOBS = {
    "Blobbin": (1, 50, 7, 10, 0, 2, 1, "Water", 1), "Grukk Brute": (24, 1400, 104, 126, 10, 408, 160, "Earth", 1),
    "Wrigglet": (2, 63, 8, 11, 0, 3, 2, "Earth", 0), "Moonbun": (3, 60, 9, 12, 0, 6, 2, "Neutral", 0),
    "Emberblob": (3, 55, 10, 13, 0, 4, 3, "Fire", 1), "Vexblob": (14, 344, 59, 72, 0, 81, 44, "Poison", 1),
    "Bramblet": (4, 95, 9, 12, 5, 5, 4, "Earth", 1), "Fiddlehop": (9, 198, 24, 29, 5, 20, 16, "Earth", 1),
    "Capling": (16, 510, 24, 48, 0, 66, 108, "Water", 1), "Pilferbug": (6, 126, 18, 24, 5, 17, 5, "Neutral", 0),
    "Rotling": (15, 534, 67, 79, 0, 50, 33, "Undead", 1), "Snarlfang": (25, 919, 37, 46, 0, 329, 199, "Earth", 1),
    "Clackbone": (10, 234, 39, 47, 10, 18, 14, "Undead", 1), "Gallopeck": (19, 531, 50, 64, 0, 159, 72, "Fire", 2),
    "Sandgulper": (26, 2282, 146, 177, 0, 550, 300, "Earth", 1),
    "Clackbone Guard": (29, 2334, 221, 245, 10, 372, 226, "Undead", 1),
    "Grumblebear": (25, 1619, 198, 220, 10, 310, 188, "Earth", 2), "Tombwrap": (37, 5176, 305, 360, 0, 800, 602, "Undead", 1),
    "Gildscarab": (64, 126000, 870, 1145, 60, 14300, 7150, "Fire", 2),
    "Scorchstripe": (65, 152000, 1215, 1565, 15, 25025, 12870, "Fire", 2),
}


@pytest.mark.parametrize("name", list(RO_MOBS))
def test_monster_numbers_match_rathena(name):
    m = M.MONSTERS[M.KIND_BY_NAME[name]]
    lv, hp, amin, amax, d, bexp, jexp, elem, size = RO_MOBS[name]
    assert (m.level, m.hp, m.atk_min, m.atk_max, m.def_, m.base_exp, m.job_exp) == (lv, hp, amin, amax, d, bexp, jexp)
    assert M.ELEMENTS[m.element] == elem and m.size == size


def test_monster_aggro_follows_rathena_ai():
    aggressive = {"Grukk Brute", "Pilferbug", "Rotling", "Clackbone Guard", "Tombwrap", "Gildscarab", "Scorchstripe"}
    assert {m.name for m in M.MONSTERS if m.aggro} == aggressive


def test_monster_walk_speed_matches_rathena():
    ro = {"Blobbin": 400, "Wrigglet": 400, "Emberblob": 400, "Rotling": 400, "Pilferbug": 150, "Gildscarab": 100,
          "Vexblob": 300, "Grumblebear": 300, "Tombwrap": 300, "Scorchstripe": 300, "Snarlfang": 200, "Grukk Brute": 200}
    for name, ms in ro.items():
        assert int(M.SPEED_MS[M.KIND_BY_NAME[name]]) == ms, name


# ---- items -------------------------------------------------------------------------------------------------------
def _row(item):
    return {c: int(np.asarray(I.ITEM_TABLE[c])[item]) for c in I.ITEM_TABLE}


@pytest.mark.parametrize("item,atk,weight,buy,slots", [
    (I.ITEM_KNIFE, 17, 40, 50, 3), (I.ITEM_CUTTER, 30, 50, 1250, 3), (I.ITEM_MAIN_GAUCHE, 43, 60, 2400, 3),
    (I.ITEM_SWORD, 25, 50, 100, 3), (I.ITEM_FALCHION, 39, 60, 1500, 3), (I.ITEM_BLADE, 53, 70, 2900, 3),
    (I.ITEM_BOW, 15, 50, 1000, 3), (I.ITEM_ROD, 15, 40, 50, 3), (I.ITEM_MACE, 37, 80, 1600, 3),
    (I.ITEM_AXE, 38, 80, 500, 3)])
def test_starter_weapons(item, atk, weight, buy, slots):
    r = _row(item)
    assert (r["ATK"], r["WEIGHT"], r["BUY"], r["SLOTS"], r["WEAPON_LEVEL"]) == (atk, weight, buy, slots, 1)
    assert r["SELL"] == buy // 2 and r["KIND"] == I.KIND_WEAPON


@pytest.mark.parametrize("item,df,weight,buy", [
    (I.ITEM_COTTON_SHIRT, 1, 10, 10), (I.ITEM_GUARD, 3, 30, 500), (I.ITEM_BUCKLER, 4, 60, 14000),
    (I.ITEM_HOOD, 1, 20, 1000), (I.ITEM_SANDALS, 1, 20, 400), (I.ITEM_CLIP, 0, 10, 30000)])
def test_starter_armor(item, df, weight, buy):
    r = _row(item)
    assert (r["DEF"], r["WEIGHT"], r["BUY"], r["SELL"]) == (df, weight, buy, buy // 2)


def test_consumables():
    heal = {I.ITEM_RED_POTION: (45, 65, 50, 7), I.ITEM_ORANGE_POTION: (105, 145, 200, 10),
            I.ITEM_YELLOW_POTION: (175, 235, 550, 13), I.ITEM_WHITE_POTION: (325, 405, 1200, 15)}
    for item, (lo, hi, buy, w) in heal.items():
        r = _row(item)
        assert lo <= r["HEAL_HP"] <= hi and r["BUY"] == buy and r["WEIGHT"] == w
    assert (_row(I.ITEM_FLIT_FEATHER)["BUY"], _row(I.ITEM_HOMING_FEATHER)["BUY"]) == (60, 300)
    assert _row(I.ITEM_FLIT_FEATHER)["KIND"] == I.KIND_CONSUMABLE and bool(I.STACKABLE[I.ITEM_FLIT_FEATHER])


def test_cards_sell_for_ten_ryn():
    cards = np.flatnonzero(np.asarray(I.KIND) == I.KIND_CARD)
    assert len(cards) >= 16 and (np.asarray(I.SELL)[cards] == 10).all()
