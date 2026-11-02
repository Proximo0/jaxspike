"""Character files and the town roster for the networked server (numpy only)."""
import json
import os
import re
import threading
import zipfile
import zlib
from pathlib import Path

import numpy as np

from game.config import INV_SLOTS, MAP, PLAYER_MAX_HP
from game.data.items import N_ITEMS
from game.world_layout import TOWNS
from net import char_fields
from net.kafra import character_storage, new_storage

_NAME_RE = re.compile(r"[a-z0-9_]{3,16}")
_INT_FIELDS = ("town", "hp", "kills", "deaths")


def valid_name(name: str) -> str | None:
    """Return the lower-cased name if it is 3-16 of [a-z0-9_], else None."""
    low = name.lower()
    return low if _NAME_RE.fullmatch(low) else None


def new_character(town: int, tile: tuple[int, int]) -> dict:
    """Build a fresh character standing on its spawn tile."""
    xy = np.array(tile, dtype=np.int32)
    return {"town": int(town), "pos": xy.copy(), "spawn": xy.copy(),
            "hp": PLAYER_MAX_HP, "inv": np.zeros(INV_SLOTS, np.int32),
            "kills": 0, "deaths": 0, **char_fields.fresh_ro_fields(), "storage": new_storage()}


def save(chars_dir: Path, name: str, fields: dict, guard=None) -> None:
    """Write a character to <chars_dir>/<name>.npz atomically (temp file, then replace).

    `guard`, if given, is called under the file lock; a falsy result skips the write."""
    final = Path(chars_dir) / f"{name}.npz"
    tmp = Path(chars_dir) / f"{name}.npz.tmp"
    with _save_lock:
        if guard is not None and not guard():
            return
        with open(tmp, "wb") as f:
            np.savez(f, **fields)
        os.replace(tmp, final)


_save_lock = threading.Lock()  # a periodic thread write and a logout write must not share the temp file
_KEYS = ("town", "pos", "spawn", "hp", "inv", "kills", "deaths")


def load(chars_dir: Path, name: str) -> dict | None:
    """Read a character; None if the file is missing, corrupt or fails validation."""
    try:
        with np.load(Path(chars_dir) / f"{name}.npz") as z:
            if not all(k in z.files for k in _KEYS):
                return None
            out = {k: z[k] for k in z.files}
        for k in _INT_FIELDS:
            out[k] = int(out[k])
        for k in ("pos", "spawn"):
            v = np.asarray(out[k])
            if v.shape != (2,) or not ((0 <= v) & (v < MAP)).all():
                return None
        if np.asarray(out["inv"]).shape != (INV_SLOTS,):
            return None
        if not 0 <= out["town"] < len(TOWNS):
            return None
        inv = np.asarray(out["inv"])
        if inv.dtype.kind not in "iu" or not ((0 <= inv) & (inv < N_ITEMS)).all():
            return None
        out.update(char_fields.validate_ro(out, inv))  # every RO key is optional: clamp or fall back to fresh
        out["storage"] = character_storage(out)
    except (OSError, ValueError, TypeError, EOFError, zlib.error, zipfile.BadZipFile):
        return None
    return out


def load_roster(chars_dir) -> dict[str, int]:
    """Read name -> town from roster.json; empty if absent or unreadable."""
    try:
        data = json.loads((Path(chars_dir) / "roster.json").read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): int(v) for k, v in data.items()}


def save_roster(chars_dir, roster) -> None:
    """Write name -> town to roster.json."""
    final = Path(chars_dir) / "roster.json"
    tmp = Path(chars_dir) / "roster.json.tmp"
    tmp.write_text(json.dumps(roster))
    os.replace(tmp, final)


def next_town(roster: dict[str, int]) -> int:
    """Pick the town with the fewest names, lowest index on ties."""
    counts = [0] * len(TOWNS)
    for t in roster.values():
        counts[t] += 1
    return counts.index(min(counts))
