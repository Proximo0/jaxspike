"""Versioned, validated world save/load (.npz). Plain Python: not part of the jitted tick.

Builds on `game.persistence._flatten`. Adds `__version__` and `__layout__` leaves so a file written for a different
map size or entity capacity is refused instead of silently loaded.
"""
import os
import zipfile
import zlib

import jax
import jax.numpy as jnp
import numpy as np

from game.config import INV_SLOTS, MAP, MAX_DROPS, MAX_MONSTERS, MAX_PLAYERS
from game.persistence import _flatten
from game.state import make_empty_world
from game.world_layout import TOWNS

WORLD_FORMAT_VERSION = 1

_LAYOUT_NAMES = ("MAP", "MAX_PLAYERS", "MAX_MONSTERS", "MAX_DROPS", "INV_SLOTS", "len(TOWNS)")
_VERSION_KEY = "__version__"
_LAYOUT_KEY = "__layout__"
_READ_ERRORS = (OSError, ValueError, zipfile.BadZipFile, zlib.error, EOFError, KeyError)


class WorldFileError(ValueError):
    """A world file is corrupt, from another format version/layout, or does not match the world schema."""


def _layout():
    return np.array([MAP, MAX_PLAYERS, MAX_MONSTERS, MAX_DROPS, INV_SLOTS, len(TOWNS)], np.int32)


def save_world(world, path):
    """Atomically write `world` to `path` (.npz): write `<path>.tmp`, then os.replace."""
    path = os.fspath(path)
    tmp = path + ".tmp"
    flat = _flatten(world)
    flat[_VERSION_KEY] = np.asarray(WORLD_FORMAT_VERSION, np.int32)
    flat[_LAYOUT_KEY] = _layout()
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **flat)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _key_name(path):
    return "/".join(str(getattr(k, "key", k)) for k in path)


def _leaf_map(tree):
    return {_key_name(p): v for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]}


def world_matches(world, template):
    """None if `world` has every key of `template` with the same shape and dtype, else the first mismatch message.
    Extra keys in `world` are allowed."""
    have, want = _leaf_map(world), _leaf_map(template)
    for key, t in want.items():
        if key not in have:
            return f"missing key {key} (expected shape {tuple(t.shape)} dtype {t.dtype})"
        v = have[key]
        if tuple(v.shape) != tuple(t.shape) or np.dtype(v.dtype) != np.dtype(t.dtype):
            return (f"{key}: shape {tuple(v.shape)} dtype {np.dtype(v.dtype)} != expected "
                    f"shape {tuple(t.shape)} dtype {np.dtype(t.dtype)}")
    return None


def load_world(path):
    """Read a world written by `save_world`; raises WorldFileError on any problem."""
    name = os.path.basename(os.fspath(path))
    flat = {}
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                flat[key] = data[key]
    except _READ_ERRORS as e:
        raise WorldFileError(f"{name}: unreadable or corrupt world file ({type(e).__name__}: {e})") from e

    if _VERSION_KEY not in flat or _LAYOUT_KEY not in flat:
        raise WorldFileError(f"{name}: missing {_VERSION_KEY}/{_LAYOUT_KEY} (not a versioned world file)")
    version = int(flat.pop(_VERSION_KEY))
    if version != WORLD_FORMAT_VERSION:
        raise WorldFileError(f"{name}: format version {version} != {WORLD_FORMAT_VERSION} (rebuild the world)")
    layout = flat.pop(_LAYOUT_KEY)
    want = _layout()
    if layout.shape != want.shape:
        raise WorldFileError(f"{name}: layout shape {layout.shape} != {want.shape} (rebuild the world)")
    for label, got, exp in zip(_LAYOUT_NAMES, layout, want):
        if int(got) != int(exp):
            raise WorldFileError(f"{name}: {label} {int(got)} != {int(exp)} (rebuild the world)")

    world = {}
    for key, arr in flat.items():
        *parents, leaf = key.split("/")
        node = world
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = jnp.asarray(arr)

    problem = world_matches(world, make_empty_world(None))
    if problem is not None:
        raise WorldFileError(f"{name}: {problem} (rebuild the world)")
    return world
