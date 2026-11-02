"""Save and load a world (nested dict of arrays) as a flat .npz file."""
import numpy as np
import jax.numpy as jnp


def _flatten(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, name + "/"))
        else:
            out[name] = np.asarray(v)
    return out


def save(world, path):
    """Write the world to `path` with nested keys flattened to 'a/b'."""
    with open(path, "wb") as f:
        np.savez(f, **_flatten(world))


def load(path):
    """Read a world saved by `save`, rebuilding the nested dict of jnp arrays."""
    world = {}
    with np.load(path) as data:
        for name in data.files:
            *parents, leaf = name.split("/")
            node = world
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = jnp.asarray(data[name])
    return world
