"""Monster drop tables: per monster kind, DROPS_PER_KIND independent per-mille rolls plus a zeny range.

One row per kind of game/data/monsters.py (20 kinds, index == kind). Entries are ro-fidelity's classic pre-renewal
rAthena rows (docs/ro_fidelity.md section 2) restricted to the items we have; chances are per mille (rAthena's
per-10000 rate // 10, floor 1), so the 0.01 % card rate becomes the 1 per mille floor (10x classic; cards are the
last entry of every row; per-10000 is ro-fidelity's open recommendation to the lead). Rough Oridecon / Rough
Elunium map to Oridecon / Elunium, Iron Ore to craft's iron ore (101). Classic drops have no potions. Classic RO
monsters drop no zeny; the zeny range is ours (a coin-per-kill income source for bots and new players), scaled by
monster level. Every drop is one unit (`count` stays 1 until the drops block carries counts).
"""
import jax.numpy as jnp

from game.data import craft as craft_data
from game.data import items as item_data
from game.data import monsters as monster_data

DROPS_PER_KIND = 8
N_DROP_KINDS = monster_data.N_KINDS

_I = item_data
# (item, chance per mille, count) per entry; item 0 = unused entry.
_C = craft_data
_ROWS = {
    "Poring": ((_I.ITEM_JELLOPY, 700, 1), (_I.ITEM_APPLE, 100, 1), (_I.ITEM_STICKY_MUCUS, 40, 1),
               (_I.ITEM_APPLE, 15, 1), (_I.ITEM_KNIFE, 10, 1), (_I.ITEM_PORING_CARD, 1, 1)),
    "Orc Warrior": ((_I.ITEM_ORCISH_VOUCHER, 900, 1), (_C.ITEM_IRON_ORE, 21, 1), (_I.ITEM_AXE, 10, 1),
                    (_I.ITEM_ORIDECON, 4, 1), (_I.ITEM_BATTLE_AXE, 1, 1), (_I.ITEM_ORC_WARRIOR_CARD, 1, 1)),
    "Fabre": ((_I.ITEM_FLUFF, 650, 1), (_I.ITEM_CLOVER, 100, 1), (_I.ITEM_FEATHER, 50, 1), (_I.ITEM_CLUB, 20, 1),
              (_I.ITEM_CLUB, 8, 1), (_I.ITEM_FABRE_CARD, 1, 1)),
    "Lunatic": ((_I.ITEM_CLOVER, 650, 1), (_I.ITEM_APPLE, 200, 1), (_I.ITEM_FEATHER, 100, 1),
                (_I.ITEM_LUNATIC_CARD, 1, 1)),
    "Drops": ((_I.ITEM_JELLOPY, 750, 1), (_I.ITEM_APPLE, 110, 1), (_I.ITEM_APPLE, 80, 1),
              (_I.ITEM_STICKY_MUCUS, 50, 1), (_I.ITEM_ROD, 8, 1), (_I.ITEM_DROPS_CARD, 1, 1)),
    "Poporing": ((_I.ITEM_STICKY_MUCUS, 550, 1), (_I.ITEM_APPLE, 25, 1), (_I.ITEM_APPLE, 1, 1),
                 (_I.ITEM_MAIN_GAUCHE, 1, 1), (_I.ITEM_POPORING_CARD, 1, 1)),
    "Willow": ((_I.ITEM_TRUNK, 900, 1), (_I.ITEM_WILLOW_CARD, 1, 1)),
    "Rocker": ((_I.ITEM_GRASSHOPPER_LEG, 900, 1), (_I.ITEM_ROCKER_CARD, 1, 1)),
    "Spore": ((_I.ITEM_MUSHROOM_SPORE, 900, 1), (_I.ITEM_HAT, 4, 1), (_I.ITEM_SPORE_CARD, 1, 1)),
    "Thief Bug": ((_I.ITEM_JELLOPY, 200, 1), (_C.ITEM_IRON_ORE, 25, 1), (_I.ITEM_JACKET, 12, 1),
                  (_I.ITEM_THIEF_BUG_CARD, 1, 1)),
    "Zombie": ((_I.ITEM_DECAYED_NAIL, 900, 1), (_I.ITEM_STICKY_MUCUS, 100, 1), (_I.ITEM_ZOMBIE_CARD, 1, 1)),
    "Wolf": ((_I.ITEM_WOLF_CLAW, 900, 1), (_I.ITEM_ANIMAL_SKIN, 550, 1), (_I.ITEM_MEAT, 65, 1),
             (_I.ITEM_EMVERETARCON, 2, 1), (_I.ITEM_MANTLE, 1, 1), (_I.ITEM_WOLF_CARD, 1, 1)),
    "Skeleton": ((_I.ITEM_JELLOPY, 300, 1), (_I.ITEM_SKEL_BONE, 80, 1), (_I.ITEM_PHRACON, 9, 1),
                 (_I.ITEM_MACE, 8, 1), (_I.ITEM_SKELETON_CARD, 1, 1)),
    "Peco Peco": ((_I.ITEM_BILL_OF_BIRDS, 900, 1), (_I.ITEM_WAND, 10, 1), (_I.ITEM_SANDALS, 2, 1),
                  (_I.ITEM_PECO_PECO_CARD, 1, 1)),
    "Hode": ((_I.ITEM_EARTHWORM_PEELING, 900, 1), (_I.ITEM_STICKY_MUCUS, 300, 1), (_I.ITEM_ELUNIUM, 8, 1),
             (_I.ITEM_HODE_CARD, 1, 1)),
    "Soldier Skeleton": ((_I.ITEM_SKEL_BONE, 550, 1), (_I.ITEM_KNIFE, 15, 1), (_I.ITEM_ORIDECON, 6, 1),
                         (_I.ITEM_SOLDIER_SKELETON_CARD, 1, 1)),
    "Bigfoot": ((_I.ITEM_BEAR_FOOT, 900, 1), (_I.ITEM_ANIMAL_SKIN, 500, 1), (_I.ITEM_ORIDECON, 4, 1),
                (_I.ITEM_BIGFOOT_CARD, 1, 1)),
    "Mummy": ((_I.ITEM_ROTTEN_BANDAGE, 900, 1), (_I.ITEM_ORIDECON, 10, 1), (_I.ITEM_GLOVE, 1, 1),
              (_I.ITEM_MUMMY_CARD, 1, 1)),
    "Golden Thief Bug": ((_I.ITEM_ELUNIUM, 200, 1), (_I.ITEM_ORIDECON, 150, 1),
                         (_I.ITEM_GOLDEN_THIEF_BUG_CARD, 1, 1)),
    "Eddga": ((_I.ITEM_TIGER_SKIN, 500, 1), (_I.ITEM_ELUNIUM, 230, 1), (_I.ITEM_EDDGA_CARD, 1, 1)),
}
assert tuple(_ROWS) == monster_data.NAMES, "one drop row per monster kind, in kind order"
assert all(len(r) <= DROPS_PER_KIND for r in _ROWS.values())
assert all(r[-1][1] == 1 and item_data.KIND[r[-1][0]] == item_data.KIND_CARD for r in _ROWS.values())
_PADDED = tuple(r + ((0, 0, 0),) * (DROPS_PER_KIND - len(r)) for r in _ROWS.values())
# (min, max) zeny per kind, uniform inclusive: 2*level .. 5*level (MVPs 100*level .. 200*level).
_ZENY = tuple(
    (100 * m.level, 200 * m.level) if m.mvp else (2 * m.level, 5 * m.level) for m in monster_data.MONSTERS
)

DROP_ITEM = jnp.array([[e[0] for e in r] for r in _PADDED], jnp.int32)
DROP_CHANCE = jnp.array([[e[1] for e in r] for r in _PADDED], jnp.int32)
DROP_COUNT = jnp.array([[e[2] for e in r] for r in _PADDED], jnp.int32)
ZENY_MIN = jnp.array([z[0] for z in _ZENY], jnp.int32)
ZENY_MAX = jnp.array([z[1] for z in _ZENY], jnp.int32)


def table_for_client():
    """The drop table as plain Python, one dict per monster kind (unused entries left out)."""
    return [
        {
            "kind": k,
            "drops": [{"item": e[0], "chance": e[1], "count": e[2]} for e in _PADDED[k] if e[0] != 0],
            "zeny_min": _ZENY[k][0],
            "zeny_max": _ZENY[k][1],
        }
        for k in range(N_DROP_KINDS)
    ]
