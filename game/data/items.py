"""Item table: ids, constants and per-column int32 arrays (M3 contract: docs/items_brief.md).

One list of row tuples is the single source; every column array, the NAMES tuple and the client table are built from
it, so ids and columns cannot drift. Plain numpy at import, then jnp arrays.
"""
import numpy as np
import jax.numpy as jnp

from game import config

# --- kinds, weapon classes, equip slots, slot types, elements, bonuses -------------------------------------------
KIND_NONE, KIND_ETC, KIND_CONSUMABLE, KIND_WEAPON, KIND_ARMOR, KIND_CARD = 0, 1, 2, 3, 4, 5

WC_NONE, WC_DAGGER, WC_SWORD, WC_BOW, WC_STAFF, WC_MACE, WC_AXE = 0, 1, 2, 3, 4, 5, 6

N_EQUIP_SLOTS = 8
EQ_WEAPON, EQ_HEAD, EQ_BODY, EQ_SHIELD, EQ_GARMENT, EQ_SHOES, EQ_ACC1, EQ_ACC2 = 0, 1, 2, 3, 4, 5, 6, 7

SLOT_WEAPON, SLOT_HEAD, SLOT_BODY, SLOT_SHIELD, SLOT_GARMENT, SLOT_SHOES, SLOT_ACCESSORY = 0, 1, 2, 3, 4, 5, 6

(ELEM_NEUTRAL, ELEM_WATER, ELEM_EARTH, ELEM_FIRE, ELEM_WIND, ELEM_POISON, ELEM_HOLY, ELEM_SHADOW, ELEM_GHOST,
 ELEM_UNDEAD) = range(10)

N_BONUS = 14
(BONUS_STR, BONUS_AGI, BONUS_VIT, BONUS_INT, BONUS_DEX, BONUS_LUK, BONUS_ATK, BONUS_DEF, BONUS_MATK, BONUS_HIT,
 BONUS_FLEE, BONUS_CRIT, BONUS_MAX_HP, BONUS_MAX_SP) = range(14)

MAX_CARDS = 4
MAX_STACK = 9999
MAX_REFINE = 10
ZENY_PER_COIN = 10
MAX_ZENY = 2_000_000_000

N_ITEMS = 73

# --- item ids ----------------------------------------------------------------------------------------------------
(ITEM_NONE, ITEM_GOLD_COIN, ITEM_MEAT, ITEM_SWORD, ITEM_RED_POTION, ITEM_ORANGE_POTION, ITEM_YELLOW_POTION,
 ITEM_WHITE_POTION, ITEM_BLUE_POTION, ITEM_APPLE, ITEM_JELLOPY, ITEM_CLOVER, ITEM_FEATHER, ITEM_FLUFF,
 ITEM_STICKY_MUCUS, ITEM_SHELL, ITEM_ANIMAL_SKIN, ITEM_ORCISH_VOUCHER, ITEM_PHRACON, ITEM_EMVERETARCON,
 ITEM_ORIDECON, ITEM_ELUNIUM, ITEM_KNIFE, ITEM_MAIN_GAUCHE, ITEM_BLADE, ITEM_FALCHION, ITEM_SCIMITAR, ITEM_BOW,
 ITEM_COMPOSITE_BOW, ITEM_ROD, ITEM_WAND, ITEM_CLUB, ITEM_MACE, ITEM_AXE, ITEM_BATTLE_AXE, ITEM_DAMASCUS,
 ITEM_HAT, ITEM_CAP, ITEM_HELM, ITEM_COTTON_SHIRT, ITEM_JACKET, ITEM_ADVENTURER_SUIT, ITEM_MANTLE, ITEM_GUARD,
 ITEM_BUCKLER, ITEM_SHIELD, ITEM_HOOD, ITEM_MUFFLER, ITEM_SANDALS, ITEM_SHOES, ITEM_BOOTS, ITEM_RING, ITEM_CLIP,
 ITEM_NECKLACE, ITEM_GLOVE, ITEM_BROOCH, ITEM_ROSARY, ITEM_PORING_CARD, ITEM_FABRE_CARD, ITEM_PUPA_CARD,
 ITEM_LUNATIC_CARD, ITEM_ROCKER_CARD, ITEM_THIEF_BUG_CARD, ITEM_ORC_WARRIOR_CARD, ITEM_ANDRE_CARD,
 ITEM_DROPS_CARD, ITEM_CHONCHON_CARD, ITEM_PICKY_CARD, ITEM_RODA_FROG_CARD, ITEM_SPORE_CARD, ITEM_HORNET_CARD,
 ITEM_CONDOR_CARD, ITEM_WILLOW_CARD) = range(N_ITEMS)

# Row column order:
# (name, kind, cls, atk, def, matk, element, slots, weight, buy, sell, heal_hp, heal_sp, weapon_level, stackable,
#  bonus, bonus_value, card_bonus, card_value)
_E, _C, _W, _A, _K = KIND_ETC, KIND_CONSUMABLE, KIND_WEAPON, KIND_ARMOR, KIND_CARD
_rows = [
    ("none",           KIND_NONE, 0,           0,   0, 0,  0, 0, 0,   0,     0,   0,  0, 0, False, 0, 0, 0, 0),
    ("gold coin",      _E, 0,                  0,   0, 0,  0, 0, 0,   0,     10,  0,  0, 0, True,  0, 0, 0, 0),
    ("meat",           _C, 0,                  0,   0, 0,  0, 0, 15,  50,    25,  config.MEAT_HEAL, 0, 0, False, 0, 0, 0, 0),
    ("sword",          _W, WC_SWORD,           25,  0, 0,  0, 3, 50,  100,   50,  0,  0, 1, False, 0, 0, 0, 0),
    ("red potion",     _C, 0,                  0,   0, 0,  0, 0, 7,   50,    25,  45, 0, 0, True,  0, 0, 0, 0),
    ("orange potion",  _C, 0,                  0,   0, 0,  0, 0, 10,  200,   100, 105, 0, 0, True, 0, 0, 0, 0),
    ("yellow potion",  _C, 0,                  0,   0, 0,  0, 0, 13,  550,   275, 175, 0, 0, True, 0, 0, 0, 0),
    ("white potion",   _C, 0,                  0,   0, 0,  0, 0, 15,  1200,  600, 325, 0, 0, True, 0, 0, 0, 0),
    ("blue potion",    _C, 0,                  0,   0, 0,  0, 0, 15,  5000,  2500, 0, 60, 0, True, 0, 0, 0, 0),
    ("apple",          _C, 0,                  0,   0, 0,  0, 0, 2,   15,    7,   16, 0, 0, True,  0, 0, 0, 0),
    ("jellopy",        _E, 0,                  0,   0, 0,  0, 0, 1,   6,     3,   0,  0, 0, True,  0, 0, 0, 0),
    ("clover",         _E, 0,                  0,   0, 0,  0, 0, 1,   8,     4,   0,  0, 0, True,  0, 0, 0, 0),
    ("feather",        _E, 0,                  0,   0, 0,  0, 0, 1,   10,    5,   0,  0, 0, True,  0, 0, 0, 0),
    ("fluff",          _E, 0,                  0,   0, 0,  0, 0, 1,   8,     4,   0,  0, 0, True,  0, 0, 0, 0),
    ("sticky mucus",   _E, 0,                  0,   0, 0,  0, 0, 2,   140,   70,  0,  0, 0, True,  0, 0, 0, 0),
    ("shell",          _E, 0,                  0,   0, 0,  0, 0, 3,   26,    13,  0,  0, 0, True,  0, 0, 0, 0),
    ("animal skin",    _E, 0,                  0,   0, 0,  0, 0, 5,   40,    20,  0,  0, 0, True,  0, 0, 0, 0),
    ("orcish voucher", _E, 0,                  0,   0, 0,  0, 0, 1,   34,    17,  0,  0, 0, True,  0, 0, 0, 0),
    ("phracon",        _E, 0,                  0,   0, 0,  0, 0, 7,   200,   100, 0,  0, 0, True,  0, 0, 0, 0),
    ("emveretarcon",   _E, 0,                  0,   0, 0,  0, 0, 7,   1000,  500, 0,  0, 0, True,  0, 0, 0, 0),
    ("oridecon",       _E, 0,                  0,   0, 0,  0, 0, 20,  2000,  1000, 0, 0, 0, True,  0, 0, 0, 0),
    ("elunium",        _E, 0,                  0,   0, 0,  0, 0, 20,  2000,  1000, 0, 0, 0, True,  0, 0, 0, 0),
    ("knife",          _W, WC_DAGGER,          17,  0, 0,  0, 3, 40,  50,    25,  0,  0, 1, False, 0, 0, 0, 0),
    ("main gauche",    _W, WC_DAGGER,          43,  0, 0,  0, 3, 50,  1000,  500, 0,  0, 1, False, 0, 0, 0, 0),
    ("blade",          _W, WC_SWORD,           53,  0, 0,  0, 3, 60,  1500,  750, 0,  0, 1, False, 0, 0, 0, 0),
    ("falchion",       _W, WC_SWORD,           39,  0, 0,  0, 4, 100, 1200,  600, 0,  0, 1, False, 0, 0, 0, 0),
    ("scimitar",       _W, WC_SWORD,           85,  0, 0,  0, 2, 80,  5000,  2500, 0, 0, 3, False, 0, 0, 0, 0),
    ("bow",            _W, WC_BOW,             15,  0, 0,  0, 3, 50,  200,   100, 0,  0, 1, False, 0, 0, 0, 0),
    ("composite bow",  _W, WC_BOW,             29,  0, 0,  0, 4, 100, 1500,  750, 0,  0, 1, False, 0, 0, 0, 0),
    ("rod",            _W, WC_STAFF,           15,  0, 15, 0, 3, 40,  50,    25,  0,  0, 1, False, 0, 0, 0, 0),
    ("wand",           _W, WC_STAFF,           25,  0, 25, 0, 2, 50,  2500,  1250, 0, 0, 2, False, 0, 0, 0, 0),
    ("club",           _W, WC_MACE,            23,  0, 0,  0, 3, 70,  15,    7,   0,  0, 1, False, 0, 0, 0, 0),
    ("mace",           _W, WC_MACE,            37,  0, 0,  0, 3, 120, 3000,  1500, 0, 0, 1, False, 0, 0, 0, 0),
    ("axe",            _W, WC_AXE,             38,  0, 0,  0, 3, 150, 3000,  1500, 0, 0, 1, False, 0, 0, 0, 0),
    ("battle axe",     _W, WC_AXE,             80,  0, 0,  0, 3, 200, 7000,  3500, 0, 0, 2, False, 0, 0, 0, 0),
    ("damascus",       _W, WC_DAGGER,          118, 0, 0,  0, 1, 70,  8000,  4000, 0, 0, 3, False, 0, 0, 0, 0),
    ("hat",            _A, SLOT_HEAD,          0,   2, 0,  0, 1, 20,  80,    40,  0,  0, 0, False, 0, 0, 0, 0),
    ("cap",            _A, SLOT_HEAD,          0,   3, 0,  0, 1, 40,  200,   100, 0,  0, 0, False, 0, 0, 0, 0),
    ("helm",           _A, SLOT_HEAD,          0,   6, 0,  0, 0, 80,  1500,  750, 0,  0, 0, False, 0, 0, 0, 0),
    ("cotton shirt",   _A, SLOT_BODY,          0,   1, 0,  0, 1, 100, 10,    5,   0,  0, 0, False, 0, 0, 0, 0),
    ("jacket",         _A, SLOT_BODY,          0,   2, 0,  0, 1, 150, 200,   100, 0,  0, 0, False, 0, 0, 0, 0),
    ("adventurer suit", _A, SLOT_BODY,         0,   2, 0,  0, 1, 100, 300,   150, 0,  0, 0, False, 0, 0, 0, 0),
    ("mantle",         _A, SLOT_BODY,          0,   4, 0,  0, 1, 300, 1000,  500, 0,  0, 0, False, 0, 0, 0, 0),
    ("guard",          _A, SLOT_SHIELD,        0,   3, 0,  0, 1, 100, 200,   100, 0,  0, 0, False, 0, 0, 0, 0),
    ("buckler",        _A, SLOT_SHIELD,        0,   4, 0,  0, 1, 200, 1000,  500, 0,  0, 0, False, 0, 0, 0, 0),
    ("shield",         _A, SLOT_SHIELD,        0,   6, 0,  0, 1, 300, 2400,  1200, 0, 0, 0, False, 0, 0, 0, 0),
    ("hood",           _A, SLOT_GARMENT,       0,   1, 0,  0, 1, 40,  100,   50,  0,  0, 0, False, 0, 0, 0, 0),
    ("muffler",        _A, SLOT_GARMENT,       0,   2, 0,  0, 1, 40,  500,   250, 0,  0, 0, False, 0, 0, 0, 0),
    ("sandals",        _A, SLOT_SHOES,         0,   1, 0,  0, 1, 40,  200,   100, 0,  0, 0, False, 0, 0, 0, 0),
    ("shoes",          _A, SLOT_SHOES,         0,   2, 0,  0, 1, 100, 800,   400, 0,  0, 0, False, 0, 0, 0, 0),
    ("boots",          _A, SLOT_SHOES,         0,   4, 0,  0, 1, 200, 3500,  1750, 0, 0, 0, False, 0, 0, 0, 0),
    ("ring",           _A, SLOT_ACCESSORY,     0,   0, 0,  0, 0, 10,  1000,  500, 0,  0, 0, False, BONUS_STR, 2, 0, 0),
    ("clip",           _A, SLOT_ACCESSORY,     0,   0, 0,  0, 1, 10,  1000,  500, 0,  0, 0, False, 0, 0, 0, 0),
    ("necklace",       _A, SLOT_ACCESSORY,     0,   0, 0,  0, 0, 10,  1000,  500, 0,  0, 0, False, BONUS_VIT, 2, 0, 0),
    ("glove",          _A, SLOT_ACCESSORY,     0,   0, 0,  0, 0, 10,  1000,  500, 0,  0, 0, False, BONUS_DEX, 2, 0, 0),
    ("brooch",         _A, SLOT_ACCESSORY,     0,   0, 0,  0, 0, 10,  1000,  500, 0,  0, 0, False, BONUS_AGI, 2, 0, 0),
    ("rosary",         _A, SLOT_ACCESSORY,     0,   0, 0,  0, 0, 10,  1000,  500, 0,  0, 0, False, BONUS_LUK, 2, 0, 0),
    ("poring card",      _K, SLOT_BODY,        0, 0, 0, 0, 0, 1, 0, 500,  0, 0, 0, True, 0, 0, BONUS_LUK, 2),
    ("fabre card",       _K, SLOT_BODY,        0, 0, 0, 0, 0, 1, 0, 300,  0, 0, 0, True, 0, 0, BONUS_VIT, 1),
    ("pupa card",        _K, SLOT_BODY,        0, 0, 0, 0, 0, 1, 0, 300,  0, 0, 0, True, 0, 0, BONUS_MAX_HP, 70),
    ("lunatic card",     _K, SLOT_BODY,        0, 0, 0, 0, 0, 1, 0, 300,  0, 0, 0, True, 0, 0, BONUS_CRIT, 1),
    ("rocker card",      _K, SLOT_BODY,        0, 0, 0, 0, 0, 1, 0, 300,  0, 0, 0, True, 0, 0, BONUS_VIT, 1),
    ("thief bug card",   _K, SLOT_ACCESSORY,   0, 0, 0, 0, 0, 1, 0, 300,  0, 0, 0, True, 0, 0, BONUS_AGI, 1),
    ("orc warrior card", _K, SLOT_WEAPON,      0, 0, 0, 0, 0, 1, 0, 500,  0, 0, 0, True, 0, 0, BONUS_ATK, 5),
    ("andre card",       _K, SLOT_WEAPON,      0, 0, 0, 0, 0, 1, 0, 500,  0, 0, 0, True, 0, 0, BONUS_ATK, 20),
    ("drops card",       _K, SLOT_ACCESSORY,   0, 0, 0, 0, 0, 1, 0, 500,  0, 0, 0, True, 0, 0, BONUS_HIT, 3),
    ("chonchon card",    _K, SLOT_ACCESSORY,   0, 0, 0, 0, 0, 1, 0, 300,  0, 0, 0, True, 0, 0, BONUS_FLEE, 2),
    ("picky card",       _K, SLOT_WEAPON,      0, 0, 0, 0, 0, 1, 0, 500,  0, 0, 0, True, 0, 0, BONUS_ATK, 10),
    ("roda frog card",   _K, SLOT_BODY,        0, 0, 0, 0, 0, 1, 0, 500,  0, 0, 0, True, 0, 0, BONUS_MAX_HP, 40),
    ("spore card",       _K, SLOT_BODY,        0, 0, 0, 0, 0, 1, 0, 500,  0, 0, 0, True, 0, 0, BONUS_VIT, 2),
    ("hornet card",      _K, SLOT_WEAPON,      0, 0, 0, 0, 0, 1, 0, 300,  0, 0, 0, True, 0, 0, BONUS_ATK, 3),
    ("condor card",      _K, SLOT_SHIELD,      0, 0, 0, 0, 0, 1, 0, 500,  0, 0, 0, True, 0, 0, BONUS_FLEE, 10),
    ("willow card",      _K, SLOT_HEAD,        0, 0, 0, 0, 0, 1, 0, 500,  0, 0, 0, True, 0, 0, BONUS_MAX_SP, 80),
]
ROWS = _rows

assert len(_rows) == N_ITEMS, f"item rows {len(_rows)} != N_ITEMS {N_ITEMS}"
assert all(len(r) == 19 for r in _rows), "every item row needs 19 columns"
for _i, _r in enumerate(_rows):  # ids and rows cannot drift
    assert globals()["ITEM_" + _r[0].upper().replace(" ", "_")] == _i, f"id/row mismatch at {_i} ({_r[0]})"
assert (ITEM_NONE, ITEM_GOLD_COIN, ITEM_MEAT, ITEM_SWORD) == (
    config.ITEM_NONE, config.ITEM_GOLD_COIN, config.ITEM_MEAT, config.ITEM_SWORD)

NAMES = tuple(r[0] for r in _rows)

_COLS = ("KIND", "CLASS", "ATK", "DEF", "MATK", "ELEMENT", "SLOTS", "WEIGHT", "BUY", "SELL", "HEAL_HP", "HEAL_SP",
         "WEAPON_LEVEL", "STACKABLE", "BONUS", "BONUS_VALUE", "CARD_BONUS", "CARD_VALUE")
_np = {c: np.array([r[1 + j] for r in _rows], dtype=np.bool_ if c == "STACKABLE" else np.int32)
       for j, c in enumerate(_COLS)}

KIND = jnp.asarray(_np["KIND"])
CLASS = jnp.asarray(_np["CLASS"])
ATK = jnp.asarray(_np["ATK"])
DEF = jnp.asarray(_np["DEF"])
MATK = jnp.asarray(_np["MATK"])
ELEMENT = jnp.asarray(_np["ELEMENT"])
SLOTS = jnp.asarray(_np["SLOTS"])
WEIGHT = jnp.asarray(_np["WEIGHT"])
BUY = jnp.asarray(_np["BUY"])
SELL = jnp.asarray(_np["SELL"])
HEAL_HP = jnp.asarray(_np["HEAL_HP"])
HEAL_SP = jnp.asarray(_np["HEAL_SP"])
WEAPON_LEVEL = jnp.asarray(_np["WEAPON_LEVEL"])
STACKABLE = jnp.asarray(_np["STACKABLE"])
BONUS = jnp.asarray(_np["BONUS"])
BONUS_VALUE = jnp.asarray(_np["BONUS_VALUE"])
CARD_BONUS = jnp.asarray(_np["CARD_BONUS"])
CARD_VALUE = jnp.asarray(_np["CARD_VALUE"])

ITEM_TABLE = {
    "KIND": KIND, "CLASS": CLASS, "ATK": ATK, "DEF": DEF, "MATK": MATK, "ELEMENT": ELEMENT, "SLOTS": SLOTS,
    "WEIGHT": WEIGHT, "BUY": BUY, "SELL": SELL, "HEAL_HP": HEAL_HP, "HEAL_SP": HEAL_SP,
    "WEAPON_LEVEL": WEAPON_LEVEL, "STACKABLE": STACKABLE, "BONUS": BONUS, "BONUS_VALUE": BONUS_VALUE,
    "CARD_BONUS": CARD_BONUS, "CARD_VALUE": CARD_VALUE,
}

assert int(HEAL_HP[ITEM_MEAT]) == config.MEAT_HEAL

# Refine ore by weapon level (0 = armor); zeny cost per attempt by weapon level.
REFINE_ORE = jnp.asarray([ITEM_ELUNIUM, ITEM_PHRACON, ITEM_EMVERETARCON, ITEM_ORIDECON, ITEM_ORIDECON], jnp.int32)
REFINE_ZENY = jnp.asarray([2000, 50, 200, 5000, 20000], jnp.int32)


def table_for_client() -> list:
    """The item table as plain Python (list of dicts, index == item id) for the welcome message."""
    keys = ("id", "name", "kind", "cls", "atk", "def", "matk", "element", "slots", "weight", "buy", "sell",
            "heal_hp", "heal_sp", "weapon_level", "stackable", "bonus", "bonus_value", "card_bonus", "card_value")
    out = []
    for i, r in enumerate(_rows):
        d = dict(zip(keys, (i,) + tuple(r)))
        d["stackable"] = bool(d["stackable"])
        out.append(d)
    return out
