"""Item table: ids, constants and per-column int32 arrays (M3 contract: docs/items_brief.md).

One list of row tuples is the single source; every column array, the NAMES tuple and the client table are built from
it, so ids and columns cannot drift. Plain numpy at import, then jnp arrays.
"""
import numpy as np
import jax.numpy as jnp

from game import config
from game.data import craft as _craft

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
RYN_PER_COIN = 1  # a gold coin stack's count IS its ryn (a 73-ryn kill is one coin drop of count 73)
MAX_RYN = 2_000_000_000

# Id space (round 2): ids 0..N_BASE_ITEMS-1 are RO items (append rows below and bump N_BASE_ITEMS; keep it at most
# CRAFT_ID_BASE), ids 100..299 are game/data/craft.py's materials and products (craft owns those ids; their rows are
# built from craft's names below), every other id is a reserved empty row. Every column covers the full space
# N_ITEMS == config.ITEM_ID_LIMIT, so craft ids no longer clamp onto the last RO row.
N_BASE_ITEMS = 98
CRAFT_ID_BASE = 100
N_ITEMS = 300

# --- item ids ----------------------------------------------------------------------------------------------------
(ITEM_NONE, ITEM_GOLD_COIN, ITEM_MEAT, ITEM_SWORD, ITEM_RED_POTION, ITEM_ORANGE_POTION, ITEM_YELLOW_POTION,
 ITEM_WHITE_POTION, ITEM_BLUE_POTION, ITEM_APPLE, ITEM_JELLY_BEAD, ITEM_CLOVER, ITEM_FEATHER, ITEM_FLUFF,
 ITEM_STICKY_MUCUS, ITEM_SHELL, ITEM_ANIMAL_SKIN, ITEM_GRUKK_TOKEN, ITEM_TIN_LODE, ITEM_BRONZE_LODE,
 ITEM_STARSTEEL, ITEM_MOONSTONE, ITEM_KNIFE, ITEM_MAIN_GAUCHE, ITEM_BLADE, ITEM_FALCHION, ITEM_SCIMITAR, ITEM_BOW,
 ITEM_COMPOSITE_BOW, ITEM_ROD, ITEM_WAND, ITEM_CLUB, ITEM_MACE, ITEM_AXE, ITEM_BATTLE_AXE, ITEM_DAMASCUS,
 ITEM_HAT, ITEM_CAP, ITEM_HELM, ITEM_COTTON_SHIRT, ITEM_JACKET, ITEM_ADVENTURER_SUIT, ITEM_MANTLE, ITEM_GUARD,
 ITEM_BUCKLER, ITEM_SHIELD, ITEM_HOOD, ITEM_MUFFLER, ITEM_SANDALS, ITEM_SHOES, ITEM_BOOTS, ITEM_RING, ITEM_CLIP,
 ITEM_NECKLACE, ITEM_GLOVE, ITEM_BROOCH, ITEM_ROSARY, ITEM_BLOBBIN_CARD, ITEM_WRIGGLET_CARD, ITEM_COCOONLING_CARD,
 ITEM_MOONBUN_CARD, ITEM_FIDDLEHOP_CARD, ITEM_PILFERBUG_CARD, ITEM_GRUKK_BRUTE_CARD, ITEM_ANTLET_CARD,
 ITEM_EMBERBLOB_CARD, ITEM_BUZZEYE_CARD, ITEM_PEEPIT_CARD, ITEM_CROAKLE_CARD, ITEM_CAPLING_CARD, ITEM_NEEDLEBUZZ_CARD,
 ITEM_CRAGWING_CARD, ITEM_BRAMBLET_CARD,
 # round 2 (ro-items2): etc loot and the cards of the 20 monster kinds' drop tables
 ITEM_TRUNK, ITEM_DECAYED_NAIL, ITEM_SNARLFANG_CLAW, ITEM_SKEL_BONE, ITEM_ROTTEN_BANDAGE, ITEM_TIGER_SKIN,
 ITEM_VEXBLOB_CARD, ITEM_ROTLING_CARD, ITEM_SNARLFANG_CARD, ITEM_CLACKBONE_CARD, ITEM_GALLOPECK_CARD, ITEM_SANDGULPER_CARD,
 ITEM_CLACKBONE_GUARD_CARD, ITEM_GRUMBLEBEAR_CARD, ITEM_TOMBWRAP_CARD, ITEM_GILDSCARAB_CARD,
 ITEM_SCORCHSTRIPE_CARD,
 # ro-fidelity: early RO staples that were missing (rAthena 1204, 601, 602)
 ITEM_CUTTER, ITEM_FLIT_FEATHER, ITEM_HOMING_FEATHER,
 # ro-items2: signature etc loot of Fiddlehop, Capling, Gallopeck, Sandgulper, Grumblebear (docs/ro_fidelity.md section 2)
 ITEM_GRASSHOPPER_LEG, ITEM_MUSHROOM_SPORE, ITEM_BILL_OF_BIRDS, ITEM_EARTHWORM_PEELING,
 ITEM_BEAR_FOOT) = range(N_BASE_ITEMS)

# Row column order:
# (name, kind, cls, atk, def, matk, element, slots, weight, buy, sell, heal_hp, heal_sp, weapon_level, stackable,
#  bonus, bonus_value, card_bonus, card_value)
_E, _C, _W, _A, _K = KIND_ETC, KIND_CONSUMABLE, KIND_WEAPON, KIND_ARMOR, KIND_CARD
_rows = [
    ("none",           KIND_NONE, 0,           0,   0, 0,  0, 0, 0,   0,     0,   0,  0, 0, False, 0, 0, 0, 0),
    ("gold coin",      _E, 0,                  0,   0, 0,  0, 0, 0,   0,     10,  0,  0, 0, True,  0, 0, 0, 0),
    ("meat",           _C, 0,                  0,   0, 0,  0, 0, 15,  50,    25,  config.MEAT_HEAL, 0, 0, False, 0, 0, 0, 0),
    ("sword",          _W, WC_SWORD,           25,  0, 0,  0, 3, 50,  100,   50,  0,  0, 1, False, 0, 0, 0, 0),
    ("red potion", _C, 0, 0, 0, 0, 0, 0, 7, 50, 25, 55, 0, 0, True, 0, 0, 0, 0),
    ("orange potion", _C, 0, 0, 0, 0, 0, 0, 10, 200, 100, 125, 0, 0, True, 0, 0, 0, 0),
    ("yellow potion", _C, 0, 0, 0, 0, 0, 0, 13, 550, 275, 205, 0, 0, True, 0, 0, 0, 0),
    ("white potion", _C, 0, 0, 0, 0, 0, 0, 15, 1200, 600, 365, 0, 0, True, 0, 0, 0, 0),
    ("blue potion", _C, 0, 0, 0, 0, 0, 0, 15, 5000, 2500, 0, 50, 0, True, 0, 0, 0, 0),
    ("apple", _C, 0, 0, 0, 0, 0, 0, 2, 15, 7, 19, 0, 0, True, 0, 0, 0, 0),
    ("jelly bead",        _E, 0,                  0,   0, 0,  0, 0, 1,   6,     3,   0,  0, 0, True,  0, 0, 0, 0),
    ("clover", _E, 0, 0, 0, 0, 0, 0, 1, 10, 5, 0, 0, 0, True, 0, 0, 0, 0),
    ("feather", _E, 0, 0, 0, 0, 0, 0, 1, 20, 10, 0, 0, 0, True, 0, 0, 0, 0),
    ("fluff",          _E, 0,                  0,   0, 0,  0, 0, 1,   8,     4,   0,  0, 0, True,  0, 0, 0, 0),
    ("sticky mucus", _E, 0, 0, 0, 0, 0, 0, 1, 70, 35, 0, 0, 0, True, 0, 0, 0, 0),
    ("shell", _E, 0, 0, 0, 0, 0, 0, 1, 14, 7, 0, 0, 0, True, 0, 0, 0, 0),
    ("animal skin", _E, 0, 0, 0, 0, 0, 0, 1, 36, 18, 0, 0, 0, True, 0, 0, 0, 0),
    ("grukk token", _E, 0, 0, 0, 0, 0, 0, 1, 142, 71, 0, 0, 0, True, 0, 0, 0, 0),
    ("tin lode", _E, 0, 0, 0, 0, 0, 0, 20, 200, 100, 0, 0, 0, True, 0, 0, 0, 0),
    ("bronze lode", _E, 0, 0, 0, 0, 0, 0, 20, 1000, 500, 0, 0, 0, True, 0, 0, 0, 0),
    ("starsteel", _E, 0, 0, 0, 0, 0, 0, 20, 1100, 550, 0, 0, 0, True, 0, 0, 0, 0),
    ("moonstone", _E, 0, 0, 0, 0, 0, 0, 20, 1100, 550, 0, 0, 0, True, 0, 0, 0, 0),
    ("knife",          _W, WC_DAGGER,          17,  0, 0,  0, 3, 40,  50,    25,  0,  0, 1, False, 0, 0, 0, 0),
    ("main gauche", _W, WC_DAGGER, 43, 0, 0, 0, 3, 60, 2400, 1200, 0, 0, 1, False, 0, 0, 0, 0),
    ("blade", _W, WC_SWORD, 53, 0, 0, 0, 3, 70, 2900, 1450, 0, 0, 1, False, 0, 0, 0, 0),
    ("falchion", _W, WC_SWORD, 39, 0, 0, 0, 3, 60, 1500, 750, 0, 0, 1, False, 0, 0, 0, 0),
    ("scimitar", _W, WC_SWORD, 85, 0, 0, 0, 2, 70, 17000, 8500, 0, 0, 2, False, 0, 0, 0, 0),
    ("bow", _W, WC_BOW, 15, 0, 0, 0, 3, 50, 1000, 500, 0, 0, 1, False, 0, 0, 0, 0),
    ("composite bow", _W, WC_BOW, 29, 0, 0, 0, 3, 60, 2500, 1250, 0, 0, 1, False, 0, 0, 0, 0),
    ("rod",            _W, WC_STAFF,           15,  0, 15, 0, 3, 40,  50,    25,  0,  0, 1, False, 0, 0, 0, 0),
    ("wand", _W, WC_STAFF, 25, 0, 25, 0, 2, 40, 2500, 1250, 0, 0, 2, False, 0, 0, 0, 0),
    ("club", _W, WC_MACE, 23, 0, 0, 0, 3, 70, 120, 60, 0, 0, 1, False, 0, 0, 0, 0),
    ("mace", _W, WC_MACE, 37, 0, 0, 0, 3, 80, 1600, 800, 0, 0, 1, False, 0, 0, 0, 0),
    ("axe", _W, WC_AXE, 38, 0, 0, 0, 3, 80, 500, 250, 0, 0, 1, False, 0, 0, 0, 0),
    ("battle axe", _W, WC_AXE, 80, 0, 0, 0, 3, 150, 5400, 2700, 0, 0, 1, False, 0, 0, 0, 0),
    ("damascus", _W, WC_DAGGER, 118, 0, 0, 0, 1, 80, 49000, 24500, 0, 0, 3, False, 0, 0, 0, 0),
    ("hat", _A, SLOT_HEAD, 0, 2, 0, 0, 1, 20, 1000, 500, 0, 0, 0, False, 0, 0, 0, 0),
    ("cap", _A, SLOT_HEAD, 0, 4, 0, 0, 1, 40, 12000, 6000, 0, 0, 0, False, 0, 0, 0, 0),
    ("helm", _A, SLOT_HEAD, 0, 6, 0, 0, 0, 60, 44000, 22000, 0, 0, 0, False, 0, 0, 0, 0),
    ("cotton shirt", _A, SLOT_BODY, 0, 1, 0, 0, 1, 10, 10, 5, 0, 0, 0, False, 0, 0, 0, 0),
    ("jacket", _A, SLOT_BODY, 0, 2, 0, 0, 1, 20, 200, 100, 0, 0, 0, False, 0, 0, 0, 0),
    ("adventurer suit", _A, SLOT_BODY, 0, 3, 0, 0, 1, 30, 1000, 500, 0, 0, 0, False, 0, 0, 0, 0),
    ("mantle", _A, SLOT_BODY, 0, 4, 0, 0, 1, 60, 10000, 5000, 0, 0, 0, False, 0, 0, 0, 0),
    ("guard", _A, SLOT_SHIELD, 0, 3, 0, 0, 1, 30, 500, 250, 0, 0, 0, False, 0, 0, 0, 0),
    ("buckler", _A, SLOT_SHIELD, 0, 4, 0, 0, 1, 60, 14000, 7000, 0, 0, 0, False, 0, 0, 0, 0),
    ("shield", _A, SLOT_SHIELD, 0, 6, 0, 0, 1, 130, 56000, 28000, 0, 0, 0, False, 0, 0, 0, 0),
    ("hood", _A, SLOT_GARMENT, 0, 1, 0, 0, 1, 20, 1000, 500, 0, 0, 0, False, 0, 0, 0, 0),
    ("muffler", _A, SLOT_GARMENT, 0, 2, 0, 0, 1, 40, 5000, 2500, 0, 0, 0, False, 0, 0, 0, 0),
    ("sandals", _A, SLOT_SHOES, 0, 1, 0, 0, 1, 20, 400, 200, 0, 0, 0, False, 0, 0, 0, 0),
    ("shoes", _A, SLOT_SHOES, 0, 2, 0, 0, 1, 40, 3500, 1750, 0, 0, 0, False, 0, 0, 0, 0),
    ("boots", _A, SLOT_SHOES, 0, 4, 0, 0, 1, 60, 18000, 9000, 0, 0, 0, False, 0, 0, 0, 0),
    ("ring", _A, SLOT_ACCESSORY, 0, 0, 0, 0, 0, 10, 30000, 15000, 0, 0, 0, False, BONUS_STR, 2, 0, 0),
    ("clip", _A, SLOT_ACCESSORY, 0, 0, 0, 0, 1, 10, 30000, 15000, 0, 0, 0, False, BONUS_MAX_SP, 10, 0, 0),
    ("necklace", _A, SLOT_ACCESSORY, 0, 0, 0, 0, 0, 10, 30000, 15000, 0, 0, 0, False, BONUS_VIT, 2, 0, 0),
    ("glove", _A, SLOT_ACCESSORY, 0, 0, 0, 0, 0, 10, 30000, 15000, 0, 0, 0, False, BONUS_DEX, 2, 0, 0),
    ("brooch", _A, SLOT_ACCESSORY, 0, 0, 0, 0, 0, 10, 30000, 15000, 0, 0, 0, False, BONUS_AGI, 2, 0, 0),
    ("rosary", _A, SLOT_ACCESSORY, 0, 0, 0, 0, 0, 10, 15000, 7500, 0, 0, 0, False, BONUS_LUK, 2, 0, 0),
    ("blobbin card", _K, SLOT_BODY, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_LUK, 2),
    ("wrigglet card", _K, SLOT_WEAPON, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_VIT, 1),
    ("cocoonling card", _K, SLOT_BODY, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_MAX_HP, 700),
    ("moonbun card", _K, SLOT_WEAPON, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_CRIT, 1),
    ("fiddlehop card", _K, SLOT_BODY, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_ATK, 5),
    ("pilferbug card", _K, SLOT_BODY, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_AGI, 1),
    ("grukk brute card", _K, SLOT_SHIELD, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_DEF, 3),
    ("antlet card", _K, SLOT_WEAPON, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_ATK, 20),
    ("emberblob card", _K, SLOT_WEAPON, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_HIT, 3),
    ("buzzeye card", _K, SLOT_SHOES, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_FLEE, 2),
    ("peepit card", _K, SLOT_BODY, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_ATK, 10),
    ("croakle card", _K, SLOT_BODY, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_MAX_HP, 400),
    ("capling card", _K, SLOT_ACCESSORY, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_VIT, 2),
    ("needlebuzz card", _K, SLOT_WEAPON, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_ATK, 3),
    ("cragwing card", _K, SLOT_GARMENT, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_FLEE, 10),
    ("bramblet card", _K, SLOT_HEAD, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_MAX_SP, 80),
    # round 2 (ro-items2): drop-table etc items and cards (classic names; numbers approximate, ro-fidelity audits)
    ("trunk", _E, 0, 0, 0, 0, 0, 0, 1, 12, 6, 0, 0, 0, True, 0, 0, 0, 0),
    ("decayed nail", _E, 0, 0, 0, 0, 0, 0, 1, 82, 41, 0, 0, 0, True, 0, 0, 0, 0),
    ("snarlfang claw", _E, 0, 0, 0, 0, 0, 0, 1, 58, 29, 0, 0, 0, True, 0, 0, 0, 0),
    ("skel bone", _E, 0, 0, 0, 0, 0, 0, 1, 232, 116, 0, 0, 0, True, 0, 0, 0, 0),
    ("rotten bandage", _E, 0, 0, 0, 0, 0, 0, 1, 350, 175, 0, 0, 0, True, 0, 0, 0, 0),
    ("tiger skin", _E, 0, 0, 0, 0, 0, 0, 1, 548, 274, 0, 0, 0, True, 0, 0, 0, 0),
    ("vexblob card", _K, SLOT_ACCESSORY, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_AGI, 1),
    ("rotling card", _K, SLOT_SHOES, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_MAX_HP, 300),
    ("snarlfang card", _K, SLOT_WEAPON, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_ATK, 15),
    ("clackbone card", _K, SLOT_WEAPON, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_ATK, 10),
    ("gallopeck card", _K, SLOT_BODY, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_MAX_HP, 100),
    ("sandgulper card", _K, SLOT_GARMENT, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_FLEE, 5),
    ("clackbone guard card", _K, SLOT_WEAPON, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_CRIT, 9),
    ("grumblebear card", _K, SLOT_SHIELD, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_VIT, 2),
    ("tombwrap card", _K, SLOT_WEAPON, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_HIT, 20),
    ("gildscarab card", _K, SLOT_SHIELD, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_DEF, 10),  # MVP (CARD_EXTRA)
    ("scorchstripe card", _K, SLOT_SHOES, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0, True, 0, 0, BONUS_STR, 5),  # MVP (CARD_EXTRA)
    # ro-fidelity: early RO staples (rAthena db/pre-re). Wings teleport in RO (random / save point): no hook yet.
    ("cutter",         _W, WC_DAGGER,          30,  0, 0,  0, 3, 50,  1250,  625, 0,  0, 1, False, 0, 0, 0, 0),
    ("flit feather",       _C, 0,                  0,   0, 0,  0, 0, 5,   60,    30,  0,  0, 0, True,  0, 0, 0, 0),
    ("homing feather", _C, 0,                  0,   0, 0,  0, 0, 5,   300,   150, 0,  0, 0, True,  0, 0, 0, 0),
    # ro-items2: drop-table etc loot (classic names; prices approximate)
    ("grasshopper leg", _E, 0,                 0,   0, 0,  0, 0, 1,   36,    18,  0,  0, 0, True,  0, 0, 0, 0),
    ("mushroom spore", _E, 0,                  0,   0, 0,  0, 0, 1,   30,    15,  0,  0, 0, True,  0, 0, 0, 0),
    ("bill of birds",  _E, 0,                  0,   0, 0,  0, 0, 1,   70,    35,  0,  0, 0, True,  0, 0, 0, 0),
    ("earthworm peeling", _E, 0,               0,   0, 0,  0, 0, 1,   66,    33,  0,  0, 0, True,  0, 0, 0, 0),
    ("bear foot",      _E, 0,                  0,   0, 0,  0, 0, 1,   130,   65,  0,  0, 0, True,  0, 0, 0, 0),
]
assert len(_rows) == N_BASE_ITEMS <= CRAFT_ID_BASE, f"item rows {len(_rows)} != N_BASE_ITEMS {N_BASE_ITEMS}"
assert all(len(r) == 19 for r in _rows), "every item row needs 19 columns"
for _i, _r in enumerate(_rows):  # ids and rows cannot drift
    assert globals()["ITEM_" + _r[0].upper().replace(" ", "_")] == _i, f"id/row mismatch at {_i} ({_r[0]})"
assert (ITEM_NONE, ITEM_GOLD_COIN, ITEM_MEAT, ITEM_SWORD) == (
    config.ITEM_NONE, config.ITEM_GOLD_COIN, config.ITEM_MEAT, config.ITEM_SWORD)
assert N_ITEMS == config.ITEM_ID_LIMIT

# --- craft items (ids from game/data/craft.py, never renumbered here) ---------------------------------------------
# Materials and products are NOT stackable: ro_craft counts inventory slots per recipe input and removes a whole
# slot per unit, so a merged stack would read as one unit and vanish whole. Flip once craft counts items.inv_count.
# Row tail after the name: kind, cls, atk, def, matk, element, slots, weight, buy, sell, heal_hp, heal_sp,
# weapon_level, stackable (bonus columns are 0).
_RESERVED = ("", KIND_NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, 0, 0, 0, 0)
_CRAFT_ROWS = {
    _craft.ITEM_COPPER_ORE:    (_E, 0,         0,  0, 0, 0, 0, 10, 40,   20,   0,   0, 0, False),
    _craft.ITEM_IRON_ORE:      (_E, 0,         0,  0, 0, 0, 0, 15, 100,  50,   0,   0, 0, False),
    _craft.ITEM_COAL:          (_E, 0,         0,  0, 0, 0, 0, 10, 60,   30,   0,   0, 0, False),
    _craft.ITEM_RAW_FISH:      (_E, 0,         0,  0, 0, 0, 0, 5,  30,   15,   0,   0, 0, False),
    _craft.ITEM_HERB:          (_E, 0,         0,  0, 0, 0, 0, 1,  20,   10,   0,   0, 0, False),
    _craft.ITEM_LOG:           (_E, 0,         0,  0, 0, 0, 0, 20, 20,   10,   0,   0, 0, False),
    _craft.ITEM_COPPER_BAR:    (_E, 0,         0,  0, 0, 0, 0, 20, 200,  100,  0,   0, 0, False),
    _craft.ITEM_IRON_BAR:      (_E, 0,         0,  0, 0, 0, 0, 25, 500,  250,  0,   0, 0, False),
    _craft.ITEM_STEEL_BAR:     (_E, 0,         0,  0, 0, 0, 0, 25, 1500, 750,  0,   0, 0, False),
    _craft.ITEM_COOKED_FISH:   (_C, 0,         0,  0, 0, 0, 0, 5,  100,  50,   70,  0, 0, False),
    _craft.ITEM_HERB_POTION:   (_C, 0,         0,  0, 0, 0, 0, 7,  120,  60,   60,  0, 0, False),
    _craft.ITEM_PLANK:         (_E, 0,         0,  0, 0, 0, 0, 10, 60,   30,   0,   0, 0, False),
    _craft.ITEM_COPPER_DAGGER: (_W, WC_DAGGER, 30, 0, 0, 0, 2, 50, 1200, 600,  0,   0, 1, False),
    _craft.ITEM_IRON_SWORD:    (_W, WC_SWORD,  60, 0, 0, 0, 2, 90, 3000, 1500, 0,   0, 2, False),
    _craft.ITEM_FISH_STEW:     (_C, 0,         0,  0, 0, 0, 0, 10, 400,  200,  200, 0, 0, False),
}
assert set(_CRAFT_ROWS) == set(_craft.CRAFT_ITEM_IDS)
assert all(CRAFT_ID_BASE <= i < N_ITEMS for i in _CRAFT_ROWS)
_full = list(_rows) + [_RESERVED] * (N_ITEMS - len(_rows))
for _i, _r in _CRAFT_ROWS.items():
    _full[_i] = (_craft.ITEM_NAMES[_i],) + _r + (0, 0, 0, 0)
ROWS = _full
assert len(ROWS) == N_ITEMS and all(len(r) == 19 for r in ROWS)

NAMES = tuple(r[0] for r in ROWS)

_COLS = ("KIND", "CLASS", "ATK", "DEF", "MATK", "ELEMENT", "SLOTS", "WEIGHT", "BUY", "SELL", "HEAL_HP", "HEAL_SP",
         "WEAPON_LEVEL", "STACKABLE", "BONUS", "BONUS_VALUE", "CARD_BONUS", "CARD_VALUE")
_np = {c: np.array([r[1 + j] for r in ROWS], dtype=np.bool_ if c == "STACKABLE" else np.int32)
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

# --- card effects beyond the row's (card_bonus, card_value) pair (round 4, ro4-cards) --------------------------------
# Classic cards often carry two or three bonuses (rAthena item_db scripts); a card row's own pair is effect 0 and
# CARD_EXTRA adds up to CARD_EXTRA_N more (bonus id, value). Negative values are drawbacks (MVP cards trade power).
# ro_items.equipped folds every pair into card_bonus in the same scatter-add. MVP cards: Gildscarab (the golden
# beetle's shell: a wall of DEF and HP, at an SP cost) and Scorchstripe (the ember tiger: STR, ATK and FLEE, less HP).
CARD_EXTRA_N = 3
CARD_EXTRA = {
    ITEM_BLOBBIN_CARD: ((BONUS_FLEE, 1),),
    ITEM_WRIGGLET_CARD: ((BONUS_MAX_HP, 100),),
    ITEM_MOONBUN_CARD: ((BONUS_LUK, 1), (BONUS_FLEE, 1)),
    ITEM_FIDDLEHOP_CARD: ((BONUS_DEX, 1),),
    ITEM_PILFERBUG_CARD: ((BONUS_FLEE, 1),),
    ITEM_GRUKK_BRUTE_CARD: ((BONUS_MAX_HP, 200),),
    ITEM_EMBERBLOB_CARD: ((BONUS_DEX, 1),),
    ITEM_BUZZEYE_CARD: ((BONUS_AGI, 1),),
    ITEM_PEEPIT_CARD: ((BONUS_STR, 1),),
    ITEM_CROAKLE_CARD: ((BONUS_MAX_SP, 50),),
    ITEM_CAPLING_CARD: ((BONUS_MAX_HP, 100),),
    ITEM_VEXBLOB_CARD: ((BONUS_FLEE, 3),),
    ITEM_ROTLING_CARD: ((BONUS_VIT, 1),),
    ITEM_SANDGULPER_CARD: ((BONUS_AGI, 1),),
    ITEM_GRUMBLEBEAR_CARD: ((BONUS_DEF, 2),),
    ITEM_GILDSCARAB_CARD: ((BONUS_MAX_HP, 1000), (BONUS_VIT, 3), (BONUS_MAX_SP, -60)),
    ITEM_SCORCHSTRIPE_CARD: ((BONUS_ATK, 30), (BONUS_FLEE, 10), (BONUS_MAX_HP, -400)),
}
assert all(_np["KIND"][i] == KIND_CARD and len(v) <= CARD_EXTRA_N for i, v in CARD_EXTRA.items())
_ex_b = np.zeros((N_ITEMS, CARD_EXTRA_N), np.int32)
_ex_v = np.zeros((N_ITEMS, CARD_EXTRA_N), np.int32)
for _i, _pairs in CARD_EXTRA.items():
    for _j, (_b, _v) in enumerate(_pairs):
        _ex_b[_i, _j], _ex_v[_i, _j] = _b, _v
CARD_EXTRA_BONUS = jnp.asarray(_ex_b)  # int32 [N_ITEMS, CARD_EXTRA_N] bonus ids (0 with value 0 = unused)
CARD_EXTRA_VALUE = jnp.asarray(_ex_v)  # int32 [N_ITEMS, CARD_EXTRA_N]
MVP_CARDS = (ITEM_GILDSCARAB_CARD, ITEM_SCORCHSTRIPE_CARD)


def card_effects(item_id: int) -> list:
    """Every (bonus id, value) pair of a card, the row's own pair first; [] for a non-card or an effectless row."""
    out = [(int(_np["CARD_BONUS"][item_id]), int(_np["CARD_VALUE"][item_id]))]
    out += [(int(b), int(v)) for b, v in CARD_EXTRA.get(item_id, ())]
    return [(b, v) for b, v in out if v != 0]


ITEM_TABLE = {
    "KIND": KIND, "CLASS": CLASS, "ATK": ATK, "DEF": DEF, "MATK": MATK, "ELEMENT": ELEMENT, "SLOTS": SLOTS,
    "WEIGHT": WEIGHT, "BUY": BUY, "SELL": SELL, "HEAL_HP": HEAL_HP, "HEAL_SP": HEAL_SP,
    "WEAPON_LEVEL": WEAPON_LEVEL, "STACKABLE": STACKABLE, "BONUS": BONUS, "BONUS_VALUE": BONUS_VALUE,
    "CARD_BONUS": CARD_BONUS, "CARD_VALUE": CARD_VALUE,
}

assert int(HEAL_HP[ITEM_MEAT]) == config.MEAT_HEAL

# Refine ore by weapon level (0 = armor); ryn cost per attempt by weapon level.
REFINE_ORE = jnp.asarray([ITEM_MOONSTONE, ITEM_TIN_LODE, ITEM_BRONZE_LODE, ITEM_STARSTEEL, ITEM_STARSTEEL], jnp.int32)
REFINE_RYN = jnp.asarray([2000, 50, 200, 5000, 20000], jnp.int32)


# --- job equip restrictions (classic pre-renewal job lists, irowiki item pages) ------------------------------------
# Job ids are game/data/stats.py's (Novice 0, Swordsman 1, Mage 2, Archer 3, Acolyte 4, Thief 5, Merchant 6).
N_JOBS = 7
_NOV, _SWD, _MAG, _ARC, _ACO, _THF, _MER = range(N_JOBS)
_ALL = frozenset(range(N_JOBS))
_BY_WEAPON_CLASS = {
    WC_DAGGER: {_NOV, _SWD, _MAG, _ARC, _THF, _MER},  # every job but Acolyte
    WC_SWORD: {_NOV, _SWD, _THF, _MER},
    WC_BOW: {_ARC, _THF},
    WC_STAFF: {_NOV, _MAG, _ACO},
    WC_MACE: {_NOV, _SWD, _ACO, _MER},
    WC_AXE: {_NOV, _SWD, _MER},
}
_BY_ITEM = {  # per-item exceptions to the weapon-class / all-jobs default
    ITEM_BATTLE_AXE: {_SWD, _MER},  # two-handed
    ITEM_WAND: {_MAG},
    ITEM_HELM: {_SWD, _ACO, _MER},
    ITEM_SHIELD: {_SWD, _ACO, _MER},
    ITEM_BUCKLER: _ALL - {_NOV},
    ITEM_MANTLE: _ALL - {_NOV},
    ITEM_BOOTS: _ALL - {_NOV},
}


def _jobs_of(i, r):
    if i in _BY_ITEM:
        return _BY_ITEM[i]
    if r[1] == KIND_WEAPON:
        return _BY_WEAPON_CLASS.get(r[2], _ALL)
    return _ALL


_EQUIP_JOBS_NP = np.array([[j in _jobs_of(i, r) for j in range(N_JOBS)] for i, r in enumerate(ROWS)], np.bool_)
EQUIP_JOBS = jnp.asarray(_EQUIP_JOBS_NP)  # bool [N_ITEMS, N_JOBS]: the job may wear the item (True for non-gear)

# --- weapon attack range in cells (RO: melee 1, bows 9 before Vulture's Eye) --------------------------------------
WEAPON_RANGE_BY_CLASS = (1, 1, 1, 9, 1, 1, 1)  # by WC_*; bare hands are WC_NONE
WEAPON_RANGE = jnp.asarray(
    [WEAPON_RANGE_BY_CLASS[r[2]] if r[1] == KIND_WEAPON else 1 for r in ROWS], jnp.int32)

# --- icon slot keys (client/assets/slots.json "item.*"); "item.etc" has no art yet ---------------------------------
_WC_ICON = {WC_DAGGER: "dagger", WC_SWORD: "sword", WC_BOW: "bow", WC_STAFF: "staff", WC_MACE: "mace", WC_AXE: "axe"}
_SLOT_ICON = {SLOT_HEAD: "hat", SLOT_BODY: "armor", SLOT_SHIELD: "shield", SLOT_GARMENT: "garment",
              SLOT_SHOES: "boots", SLOT_ACCESSORY: "ring"}
_NAME_ICON = {"gold coin": "gold_coin", "meat": "meat", "blue potion": "potion_blue", "herb": "herb",
              "clover": "herb", "raw fish": "fish", "cooked fish": "fish", "fish stew": "fish", "apple": "meat"}
_ORE_WORDS = ("ore", "coal", "bar", "tin lode", "bronze lode", "starsteel", "moonstone")


def _icon(r):
    name, kind, cls = r[0], r[1], r[2]
    if kind == KIND_NONE:
        return ""
    if name in _NAME_ICON:
        return "item." + _NAME_ICON[name]
    if kind == KIND_WEAPON:
        return "item." + _WC_ICON[cls]
    if kind == KIND_ARMOR:
        return "item." + _SLOT_ICON[cls]
    if kind == KIND_CARD:
        return "item.card"
    if kind == KIND_CONSUMABLE:
        return "item.potion_red"
    words = set(name.replace("-", " ").split())
    if any(w in words or (" " in w and w in name) for w in _ORE_WORDS):  # whole words: "spore" is not "ore"
        return "item.ore"
    return "item.etc"


ICONS = tuple(_icon(r) for r in ROWS)

# --- display names (ro5-items): what every panel shows; the engine keeps the lowercase NAMES ---------------------
_LOWER_WORDS = ("of", "the", "and")


def display_name(name: str) -> str:
    """'gildscarab card' -> 'Gildscarab Card', 'bill of birds' -> 'Bill of Birds'; '' stays ''."""
    words = name.split()
    return " ".join(w if (j > 0 and w in _LOWER_WORDS) else w[:1].upper() + w[1:] for j, w in enumerate(words))


DISPLAY_NAMES = tuple(display_name(n) if r[1] != KIND_NONE else "" for n, r in zip(NAMES, ROWS))


def table_for_client() -> list:
    """The item table as plain Python (list of N_ITEMS dicts, index == item id) for the welcome message.

    Reserved ids have name "" and kind 0. Round 2 adds "icon" (slots.json key) and "jobs" (job ids that may wear it).
    """
    keys = ("id", "name", "kind", "cls", "atk", "def", "matk", "element", "slots", "weight", "buy", "sell",
            "heal_hp", "heal_sp", "weapon_level", "stackable", "bonus", "bonus_value", "card_bonus", "card_value")
    out = []
    for i, r in enumerate(ROWS):
        d = dict(zip(keys, (i,) + tuple(r)))
        d["stackable"] = bool(d["stackable"])
        d["icon"] = ICONS[i]
        d["display"] = DISPLAY_NAMES[i]  # ro5-items: capitalised name ("Gildscarab Card"); client adds +refine, [slots]
        d["jobs"] = [j for j in range(N_JOBS) if _EQUIP_JOBS_NP[i, j]]
        out.append(d)
    return out
