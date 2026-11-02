"""Monster drop tables: per monster kind, DROPS_PER_KIND independent per-mille rolls plus a zeny range."""
import jax.numpy as jnp

from game.data import items as item_data

DROPS_PER_KIND = 6
N_DROP_KINDS = 2  # rat = MONSTER_KIND_RAT (0), orc = MONSTER_KIND_ORC (1)

_I = item_data
# One row per kind: (item, chance per mille, count). Item 0 = unused entry.
_ROWS = (
    (  # rat
        (_I.ITEM_JELLOPY, 700, 1),
        (_I.ITEM_STICKY_MUCUS, 400, 1),
        (_I.ITEM_APPLE, 200, 1),
        (_I.ITEM_RED_POTION, 100, 1),
        (_I.ITEM_KNIFE, 10, 1),
        (_I.ITEM_PORING_CARD, 1, 1),
    ),
    (  # orc
        (_I.ITEM_ORCISH_VOUCHER, 550, 1),
        (_I.ITEM_ANIMAL_SKIN, 300, 1),
        (_I.ITEM_ORANGE_POTION, 100, 1),
        (_I.ITEM_PHRACON, 100, 1),
        (_I.ITEM_AXE, 20, 1),
        (_I.ITEM_ORC_WARRIOR_CARD, 1, 1),
    ),
)
_ZENY = ((10, 30), (30, 80))  # (min, max) per kind, uniform inclusive

assert len(_ROWS) == N_DROP_KINDS and all(len(r) == DROPS_PER_KIND for r in _ROWS)

DROP_ITEM = jnp.array([[e[0] for e in r] for r in _ROWS], jnp.int32)
DROP_CHANCE = jnp.array([[e[1] for e in r] for r in _ROWS], jnp.int32)
DROP_COUNT = jnp.array([[e[2] for e in r] for r in _ROWS], jnp.int32)
ZENY_MIN = jnp.array([z[0] for z in _ZENY], jnp.int32)
ZENY_MAX = jnp.array([z[1] for z in _ZENY], jnp.int32)


def table_for_client():
    """The drop table as plain Python, one dict per monster kind."""
    return [
        {
            "kind": k,
            "drops": [
                {"item": e[0], "chance": e[1], "count": e[2]} for e in _ROWS[k]
            ],
            "zeny_min": _ZENY[k][0],
            "zeny_max": _ZENY[k][1],
        }
        for k in range(N_DROP_KINDS)
    ]
