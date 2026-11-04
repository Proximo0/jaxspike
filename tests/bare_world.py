"""Test helper: strip the Novice starter kit (inventory slots 0 and 1, worn knife and shirt) from a world."""
from game.data import items as item_data


def bare(w):
    it = w["items"]
    return {**w, "players": {**w["players"], "inv": w["players"]["inv"].at[:, :2].set(0)},
            "items": {**it, "inv_count": it["inv_count"].at[:, :2].set(0),
                      "equip": it["equip"].at[:, item_data.EQ_WEAPON].set(-1).at[:, item_data.EQ_BODY].set(-1)}}
