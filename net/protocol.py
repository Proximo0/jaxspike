"""Wire protocol, message validation, host view and viewport snapshots (numpy and json only)."""
import json

import numpy as np

from game.config import (INV_SLOTS, MAP, MONSTER_HP, PLAYER_MAX_HP, PROTOCOL_VERSION, T_ROCK, VIEW_H,
                         VIEW_W)

_MONSTER_HP = np.asarray(MONSTER_HP)


def encode(msg: dict) -> str:
    """Serialise a message with the protocol version added."""
    return json.dumps({**msg, "v": PROTOCOL_VERSION})


def decode(text: str) -> dict | None:
    """Parse a message; None unless a JSON object with the right version and a string type."""
    try:
        msg = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(msg, dict) or msg.get("v") != PROTOCOL_VERSION or not isinstance(msg.get("t"), str):
        return None
    return msg


def _is_int(v) -> bool:
    """True for real ints (not bools)."""
    return isinstance(v, int) and not isinstance(v, bool)


def validate_client(msg: dict) -> dict | None:
    """Return the message reduced to its allowed keys, or None if invalid."""
    t = msg.get("t")
    if t == "login" and isinstance(msg.get("name"), str):
        return {"t": t, "name": msg["name"]}
    if t == "input" and _is_int(msg.get("move")) and 0 <= msg["move"] <= 4 \
            and isinstance(msg.get("attack"), bool) and isinstance(msg.get("eat"), bool):
        return {"t": t, "move": msg["move"], "attack": msg["attack"], "eat": msg["eat"]}
    if t == "walk_to" and all(_is_int(msg.get(k)) and 0 <= msg[k] < MAP for k in ("x", "y")):
        return {"t": t, "x": msg["x"], "y": msg["y"]}
    return None


def view_window(pos) -> tuple[int, int]:
    """Top-left tile of the view window centred on pos."""
    return int(pos[0]) - VIEW_W // 2, int(pos[1]) - VIEW_H // 2


def terrain_window(terrain: np.ndarray, x0: int, y0: int) -> list[list[int]]:
    """VIEW_W rows (one per x) of VIEW_H terrain ids, T_ROCK outside the map."""
    out = np.full((VIEW_W, VIEW_H), T_ROCK, np.int32)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + VIEW_W, MAP), min(y0 + VIEW_H, MAP)
    if sx0 < sx1 and sy0 < sy1:
        out[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = terrain[sx0:sx1, sy0:sy1]
    return out.tolist()


def host_view(world, static: dict | None = None) -> dict:
    """Numpy copies of the arrays a snapshot needs. `static` ({"terrain","pz"} numpy) skips re-copying the map."""
    p, m, d = world["players"], world["monsters"], world["drops"]
    if static is None:
        static = {"terrain": np.asarray(world["terrain"]), "pz": np.asarray(world["pz"])}
    return {
        "terrain": static["terrain"], "pz": static["pz"],
        "players": {k: np.asarray(p[k]) for k in ("pos", "spawn", "hp", "active", "dead", "inv", "kills", "deaths")},
        "monsters": {k: np.asarray(m[k]) for k in ("pos", "hp", "kind", "active")},
        "drops": {k: np.asarray(d[k]) for k in ("pos", "item", "active")},
    }


def _in_window(pos: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Mask of rows of pos inside the window box."""
    return (pos[:, 0] >= x0) & (pos[:, 0] < x0 + VIEW_W) & (pos[:, 1] >= y0) & (pos[:, 1] < y0 + VIEW_H)


def _player_ents(p: dict, x0: int, y0: int, names: dict) -> list[dict]:
    """Living players in the window, by name."""
    idx = np.nonzero(p["active"] & ~p["dead"] & _in_window(p["pos"], x0, y0))[0]
    return [{"kind": "p", "name": names.get(int(i), "bot"), "pos": p["pos"][i].tolist(),
             "hp": int(p["hp"][i]), "max_hp": PLAYER_MAX_HP} for i in idx]


def _monster_ents(m: dict, x0: int, y0: int) -> list[dict]:
    """Active monsters in the window."""
    idx = np.nonzero(m["active"] & _in_window(m["pos"], x0, y0))[0]
    return [{"kind": "m", "mkind": int(m["kind"][i]), "pos": m["pos"][i].tolist(), "hp": int(m["hp"][i]),
             "max_hp": int(_MONSTER_HP[m["kind"][i]])} for i in idx]


def _drop_ents(d: dict, x0: int, y0: int) -> list[dict]:
    """Active drops in the window."""
    idx = np.nonzero(d["active"] & _in_window(d["pos"], x0, y0))[0]
    return [{"kind": "d", "item": int(d["item"][i]), "pos": d["pos"][i].tolist()} for i in idx]


def snapshot(hv: dict, slot: int, names: dict, send_terrain: bool, notice: str | None, tick: int,
             town: int) -> dict:
    """Build the snap message for one player's view window; `town` is the player's home town."""
    p = hv["players"]
    pos = p["pos"][slot]
    x0, y0 = view_window(pos)
    you = {"pos": pos.tolist(), "hp": int(p["hp"][slot]), "max_hp": PLAYER_MAX_HP,
           "kills": int(p["kills"][slot]), "deaths": int(p["deaths"][slot]),
           "inv": p["inv"][slot][:INV_SLOTS].tolist(), "town": int(town),
           "safe": bool(hv["pz"][pos[0], pos[1]])}
    ents = _player_ents(p, x0, y0, names) + _monster_ents(hv["monsters"], x0, y0) + _drop_ents(hv["drops"], x0, y0)
    terrain = {"x0": x0, "y0": y0, "rows": terrain_window(hv["terrain"], x0, y0)} if send_terrain else None
    return {"t": "snap", "tick": tick, "you": you, "ents": ents, "terrain": terrain, "notice": notice}
