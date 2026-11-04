import numpy as np

from game.data import monsters as M
from game.world_layout import DUNGEONS, FIELD_BANDS

ORDER = ("Blobbin", "Grukk Brute", "Wrigglet", "Moonbun", "Emberblob", "Vexblob", "Bramblet", "Fiddlehop", "Capling",
         "Pilferbug", "Rotling", "Snarlfang", "Clackbone", "Gallopeck", "Sandgulper", "Clackbone Guard", "Grumblebear",
         "Tombwrap", "Gildscarab", "Scorchstripe")


def test_table_shape_and_order():
    assert M.N_KINDS == 20 and len(M.MONSTERS) == 20
    assert M.NAMES == ORDER
    assert M.KIND_BY_NAME["Blobbin"] == 0 and M.KIND_BY_NAME["Grukk Brute"] == 1
    for a in (M.HP, M.ATK_MIN, M.ATK_MAX, M.DEF, M.ELEMENT, M.LEVEL, M.BASE_EXP, M.JOB_EXP, M.AGGRO,
              M.MOVE_TICKS, M.SIZE, M.MVP):
        assert a.shape == (20,)
    assert M.HP.dtype == np.int32 and M.AGGRO.dtype == bool and M.MVP.dtype == bool


def test_legacy_slots_and_tuples():
    assert M.MONSTER_HP_TUPLE[:2] == (50, 1400)  # RO pre-re Blobbin / Grukk Brute (ro-fidelity)
    assert M.MONSTER_DAMAGE_TUPLE[:2] == (10, 126)
    assert len(M.MONSTER_HP_TUPLE) == len(M.MONSTER_DAMAGE_TUPLE) == 20


def test_values_sane():
    assert (np.asarray(M.ATK_MIN) <= np.asarray(M.ATK_MAX)).all()
    assert (np.asarray(M.MOVE_TICKS) >= 1).all() and (np.asarray(M.ELEMENT) < len(M.ELEMENTS)).all()
    assert (np.asarray(M.SIZE) <= 2).all()
    lv = np.asarray(M.LEVEL)  # RO levels (ro-fidelity): rows are not sorted by level (Vexblob 14 before Bramblet 4)
    assert (lv >= 1).all() and lv[-1] > lv[0]
    mvp = [M.NAMES[i] for i in np.flatnonzero(np.asarray(M.MVP))]
    assert mvp == ["Gildscarab", "Scorchstripe"]
    for i in (18, 19):
        assert M.MONSTER_HP_TUPLE[i] >= 100_000 and M.MONSTER_DAMAGE_TUPLE[i] >= 1000  # RO pre-re MVPs


def test_layout_names_exist():
    for *_, names in FIELD_BANDS:
        assert all(n in M.KIND_BY_NAME for n in names)
    for d in DUNGEONS:
        assert all(n in M.KIND_BY_NAME for n in d[4]) and d[5] in M.KIND_BY_NAME
        assert M.MVP[M.KIND_BY_NAME[d[5]]]
