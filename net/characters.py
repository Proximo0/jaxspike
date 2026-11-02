"""Character files and the town roster for the networked server (numpy only)."""
import json
import re
import zipfile
import zlib
from pathlib import Path

import numpy as np

from game.config import INV_SLOTS, PLAYER_MAX_HP
from game.world_layout import TOWNS

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
            "kills": 0, "deaths": 0}


def save(chars_dir: Path, name: str, fields: dict) -> None:
    """Write a character to <chars_dir>/<name>.npz."""
    np.savez(Path(chars_dir) / f"{name}.npz", **fields)


def load(chars_dir: Path, name: str) -> dict | None:
    """Read a character; None if the file is missing or corrupt."""
    try:
        with np.load(Path(chars_dir) / f"{name}.npz") as z:
            out = {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError, zlib.error, zipfile.BadZipFile):
        return None
    for k in _INT_FIELDS:
        if k in out:
            out[k] = int(out[k])
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
    (Path(chars_dir) / "roster.json").write_text(json.dumps(roster))


def next_town(roster: dict[str, int]) -> int:
    """Pick the town with the fewest names, lowest index on ties."""
    counts = [0] * len(TOWNS)
    for t in roster.values():
        counts[t] += 1
    return counts.index(min(counts))
