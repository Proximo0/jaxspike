"""Monster drop tables: per monster kind, DROPS_PER_KIND independent per-10000 rolls plus a ryn range.

One row per kind of game/data/monsters.py (20 kinds, index == kind). Entries are ro-fidelity's classic pre-renewal
rAthena rows (docs/ro_fidelity.md section 2) restricted to the items we have; chances are rAthena's real per-10000
rates (cards stay 1 = 0.01 %, classic; cards are the last entry of every row). Rough Starsteel / Rough
Moonstone map to Starsteel / Moonstone, Iron Ore to craft's iron ore (101). Classic drops have no potions. Classic RO
monsters drop no ryn; the ryn range is ours (a coin-per-kill income source for bots and new players), scaled by
monster level. Every entry drops one unit (the count column); the ground drop carries it in drops["count"].
"""
import jax.numpy as jnp

from game.data import craft as craft_data
from game.data import items as item_data
from game.data import monsters as monster_data

DROPS_PER_KIND = 8
N_DROP_KINDS = monster_data.N_KINDS

_I = item_data
# (item, chance per 10000, count) per entry; item 0 = unused entry.
_C = craft_data
_ROWS = {
    "Blobbin": ((_I.ITEM_JELLY_BEAD, 7000, 1), (_I.ITEM_APPLE, 1000, 1), (_I.ITEM_STICKY_MUCUS, 400, 1),
               (_I.ITEM_APPLE, 150, 1), (_I.ITEM_KNIFE, 100, 1), (_I.ITEM_BLOBBIN_CARD, 1, 1)),
    "Grukk Brute": ((_I.ITEM_GRUKK_TOKEN, 9000, 1), (_C.ITEM_IRON_ORE, 210, 1), (_I.ITEM_AXE, 100, 1),
                    (_I.ITEM_STARSTEEL, 40, 1), (_I.ITEM_BATTLE_AXE, 10, 1), (_I.ITEM_GRUKK_BRUTE_CARD, 1, 1)),
    "Wrigglet": ((_I.ITEM_FLUFF, 6500, 1), (_I.ITEM_CLOVER, 1000, 1), (_I.ITEM_FEATHER, 500, 1), (_I.ITEM_CLUB, 200, 1),
              (_I.ITEM_CLUB, 80, 1), (_I.ITEM_WRIGGLET_CARD, 1, 1)),
    "Moonbun": ((_I.ITEM_CLOVER, 6500, 1), (_I.ITEM_APPLE, 2000, 1), (_I.ITEM_FEATHER, 1000, 1),
                (_I.ITEM_MOONBUN_CARD, 1, 1)),
    "Emberblob": ((_I.ITEM_JELLY_BEAD, 7500, 1), (_I.ITEM_APPLE, 1100, 1), (_I.ITEM_APPLE, 800, 1),
              (_I.ITEM_STICKY_MUCUS, 500, 1), (_I.ITEM_ROD, 80, 1), (_I.ITEM_EMBERBLOB_CARD, 1, 1)),
    "Vexblob": ((_I.ITEM_STICKY_MUCUS, 5500, 1), (_I.ITEM_APPLE, 250, 1), (_I.ITEM_APPLE, 5, 1),
                 (_I.ITEM_MAIN_GAUCHE, 5, 1), (_I.ITEM_VEXBLOB_CARD, 1, 1)),
    "Bramblet": ((_I.ITEM_TRUNK, 9000, 1), (_I.ITEM_BRAMBLET_CARD, 1, 1)),
    "Fiddlehop": ((_I.ITEM_GRASSHOPPER_LEG, 9000, 1), (_I.ITEM_FIDDLEHOP_CARD, 1, 1)),
    "Capling": ((_I.ITEM_MUSHROOM_SPORE, 9000, 1), (_I.ITEM_HAT, 40, 1), (_I.ITEM_CAPLING_CARD, 1, 1)),
    "Pilferbug": ((_I.ITEM_JELLY_BEAD, 2000, 1), (_C.ITEM_IRON_ORE, 250, 1), (_I.ITEM_JACKET, 120, 1),
                  (_I.ITEM_PILFERBUG_CARD, 1, 1)),
    "Rotling": ((_I.ITEM_DECAYED_NAIL, 9000, 1), (_I.ITEM_STICKY_MUCUS, 1000, 1), (_I.ITEM_ROTLING_CARD, 1, 1)),
    "Snarlfang": ((_I.ITEM_SNARLFANG_CLAW, 9000, 1), (_I.ITEM_ANIMAL_SKIN, 5500, 1), (_I.ITEM_MEAT, 650, 1),
             (_I.ITEM_BRONZE_LODE, 20, 1), (_I.ITEM_MANTLE, 10, 1), (_I.ITEM_SNARLFANG_CARD, 1, 1)),
    "Clackbone": ((_I.ITEM_JELLY_BEAD, 3000, 1), (_I.ITEM_SKEL_BONE, 800, 1), (_I.ITEM_TIN_LODE, 90, 1),
                 (_I.ITEM_MACE, 80, 1), (_I.ITEM_CLACKBONE_CARD, 1, 1)),
    "Gallopeck": ((_I.ITEM_BILL_OF_BIRDS, 9000, 1), (_I.ITEM_WAND, 100, 1), (_I.ITEM_SANDALS, 20, 1),
                  (_I.ITEM_GALLOPECK_CARD, 1, 1)),
    "Sandgulper": ((_I.ITEM_EARTHWORM_PEELING, 9000, 1), (_I.ITEM_STICKY_MUCUS, 3000, 1), (_I.ITEM_MOONSTONE, 80, 1),
             (_I.ITEM_SANDGULPER_CARD, 1, 1)),
    "Clackbone Guard": ((_I.ITEM_SKEL_BONE, 5500, 1), (_I.ITEM_KNIFE, 150, 1), (_I.ITEM_STARSTEEL, 60, 1),
                         (_I.ITEM_CLACKBONE_GUARD_CARD, 1, 1)),
    "Grumblebear": ((_I.ITEM_BEAR_FOOT, 9000, 1), (_I.ITEM_ANIMAL_SKIN, 5000, 1), (_I.ITEM_STARSTEEL, 43, 1),
                (_I.ITEM_GRUMBLEBEAR_CARD, 1, 1)),
    "Tombwrap": ((_I.ITEM_ROTTEN_BANDAGE, 9000, 1), (_I.ITEM_STARSTEEL, 100, 1), (_I.ITEM_GLOVE, 1, 1),
              (_I.ITEM_TOMBWRAP_CARD, 1, 1)),
    "Gildscarab": ((_I.ITEM_MOONSTONE, 2000, 1), (_I.ITEM_STARSTEEL, 1500, 1),
                         (_I.ITEM_GILDSCARAB_CARD, 1, 1)),
    "Scorchstripe": ((_I.ITEM_TIGER_SKIN, 5000, 1), (_I.ITEM_MOONSTONE, 2300, 1), (_I.ITEM_SCORCHSTRIPE_CARD, 1, 1)),
}
assert tuple(_ROWS) == monster_data.NAMES, "one drop row per monster kind, in kind order"
assert all(len(r) <= DROPS_PER_KIND for r in _ROWS.values())
assert all(r[-1][1] == 1 and item_data.KIND[r[-1][0]] == item_data.KIND_CARD for r in _ROWS.values())
_PADDED = tuple(r + ((0, 0, 0),) * (DROPS_PER_KIND - len(r)) for r in _ROWS.values())
# (min, max) ryn per kind, uniform inclusive: 2*level .. 5*level (MVPs 100*level .. 200*level).
_RYN = tuple(
    (100 * m.level, 200 * m.level) if m.mvp else (2 * m.level, 5 * m.level) for m in monster_data.MONSTERS
)

DROP_ITEM = jnp.array([[e[0] for e in r] for r in _PADDED], jnp.int32)
DROP_CHANCE = jnp.array([[e[1] for e in r] for r in _PADDED], jnp.int32)
DROP_COUNT = jnp.array([[e[2] for e in r] for r in _PADDED], jnp.int32)
RYN_MIN = jnp.array([z[0] for z in _RYN], jnp.int32)
RYN_MAX = jnp.array([z[1] for z in _RYN], jnp.int32)


def table_for_client():
    """The drop table as plain Python, one dict per monster kind (unused entries left out); chance is per 10000."""
    return [
        {
            "kind": k,
            "drops": [{"item": e[0], "chance": e[1], "count": e[2]} for e in _PADDED[k] if e[0] != 0],
            "ryn_min": _RYN[k][0],
            "ryn_max": _RYN[k][1],
        }
        for k in range(N_DROP_KINDS)
    ]
