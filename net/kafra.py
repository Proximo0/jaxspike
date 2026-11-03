"""Kafra storage: pure numpy moves between a character's inventory and its storage."""
import numpy as np

from game.config import ITEM_NONE
from game.data import items as item_data

STORAGE_SLOTS = 100
# Ids a storage slot may hold: ITEM_NONE or any real row of the item table (RO items and craft ids 100..299);
# reserved ids (kind 0) and anything outside [0, N_ITEMS) are invalid.
_VALID_ID = np.asarray(item_data.KIND) != item_data.KIND_NONE
_VALID_ID[ITEM_NONE] = True


def valid_ids(ids) -> bool:
    """True when every id in `ids` is ITEM_NONE or a real item id of game/data/items.py."""
    ids = np.asarray(ids)
    if ids.dtype.kind not in "iu":
        return False
    inside = (ids >= 0) & (ids < item_data.N_ITEMS)
    return bool(inside.all() and _VALID_ID[ids].all())


def new_storage() -> np.ndarray:
    return np.full(STORAGE_SLOTS, ITEM_NONE, np.int32)


def transfer(inv, storage, slot, direction):
    """Return (inv, storage, ok); new arrays, inputs never mutated."""
    inv, storage = np.array(inv, copy=True), np.array(storage, copy=True)
    if direction == "in":
        src, dst = inv, storage
    elif direction == "out":
        src, dst = storage, inv
    else:
        return inv, storage, False
    if isinstance(slot, bool) or not isinstance(slot, (int, np.integer)) or not 0 <= slot < len(src):
        return inv, storage, False
    if src[slot] == ITEM_NONE:
        return inv, storage, False
    free = np.flatnonzero(dst == ITEM_NONE)
    if free.size == 0:
        return inv, storage, False
    dst[free[0]] = src[slot]
    src[slot] = ITEM_NONE
    return inv, storage, True


def character_storage(char) -> np.ndarray:
    s = char.get("storage")
    try:
        s = np.asarray(s)
        if s.dtype == np.int32 and s.shape == (STORAGE_SLOTS,) and valid_ids(s):
            return s.copy()
    except (TypeError, ValueError):
        pass
    return new_storage()
