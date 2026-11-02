"""Wire protocol v2: codec, message validation, map blobs and per-viewer delta snapshots (numpy, json, zlib)."""
import base64
import json
import zlib

import numpy as np

from game.config import INV_SLOTS, MAP, PLAYER_MAX_HP, PROTOCOL_VERSION, VIEW_H, VIEW_W


def encode(msg: dict) -> str:
    """Serialise a message (compact JSON) with the protocol version added."""
    return json.dumps({**msg, "v": PROTOCOL_VERSION}, separators=(",", ":"))


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


def encode_map(arr: np.ndarray, dtype) -> str:
    """base64(zlib level 6) of an [x, y]-indexed MAP x MAP array, row-major with index y * MAP + x."""
    raw = np.ascontiguousarray(np.asarray(arr, dtype).T).tobytes()
    return base64.b64encode(zlib.compress(raw, 6)).decode("ascii")


def decode_map(b64: str, dtype) -> np.ndarray:
    """Inverse of encode_map: an [x, y]-indexed MAP x MAP array (raises ValueError on a bad blob)."""
    try:
        raw = zlib.decompress(base64.b64decode(b64))
    except (zlib.error, ValueError) as e:
        raise ValueError("bad map blob") from e
    a = np.frombuffer(raw, dtype)
    if a.size != MAP * MAP:
        raise ValueError("bad map size")
    return np.ascontiguousarray(a.reshape(MAP, MAP).T)


def welcome_message(slot: int, tick_ms: int, maps: tuple[str, str]) -> dict:
    """The welcome message; `maps` = (terrain_b64, pz_b64) built once per server with encode_map."""
    return {"t": "welcome", "slot": slot, "map": MAP, "tick_ms": tick_ms, "view_w": VIEW_W, "view_h": VIEW_H,
            "terrain": maps[0], "pz": maps[1]}


def entity_rows(ids: np.ndarray, tab: dict, names: dict) -> list[dict]:
    """JSON entities for the global ids (numpy int array, no padding) from the pulled entity table."""
    pos, hp = tab["pos"][ids].tolist(), tab["hp"][ids].tolist()
    max_hp, kind = tab["max_hp"][ids].tolist(), tab["kind"][ids].tolist()
    out = []
    for i, e in enumerate(ids.tolist()):
        k = kind[i]
        ent = {"id": e, "kind": k, "pos": pos[i], "hp": hp[i], "max_hp": max_hp[i]}
        if k == 0:
            ent["name"] = names.get(e, "bot")
        elif k >= 100:
            ent["item"] = k - 100
        else:
            ent["mkind"] = k - 1
        out.append(ent)
    return out


def snapshot(pull: dict, slot: int, names: dict, notice: str | None, tick: int, town: int, full: bool) -> dict:
    """Build the snap message for one viewer (row `slot`) from the tick's single host pull.

    pull: {"send", "gone": int [MAX_VIEWERS, VIEW_K] with -1 pads, "tab": entity table, "players": arrays, "safe"}."""
    p = pull["players"]
    send, gone = pull["send"][slot], pull["gone"][slot]
    you = {"pos": p["pos"][slot].tolist(), "hp": int(p["hp"][slot]), "max_hp": PLAYER_MAX_HP,
           "kills": int(p["kills"][slot]), "deaths": int(p["deaths"][slot]),
           "inv": p["inv"][slot][:INV_SLOTS].tolist(), "town": int(town), "safe": bool(pull["safe"][slot])}
    return {"t": "snap", "tick": tick, "you": you, "ents": entity_rows(send[send >= 0], pull["tab"], names),
            "gone": gone[gone >= 0].tolist(), "notice": notice, "full": bool(full)}
