"""Kafra storage: pure numpy moves between a character's inventory and its storage."""
import numpy as np

from game.config import ITEM_NONE, NUM_ITEMS

STORAGE_SLOTS = 100


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
        if s.dtype == np.int32 and s.shape == (STORAGE_SLOTS,) and ((s >= 0) & (s < NUM_ITEMS)).all():
            return s.copy()
    except (TypeError, ValueError):
        pass
    return new_storage()
