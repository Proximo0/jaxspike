"""Wire protocol v2: codec, message validation, map blobs and per-viewer delta snapshots (numpy, json, zlib)."""
import base64
import json
import zlib

import numpy as np

from game.config import INV_SLOTS, MAP, N_ENTITIES, PLAYER_MAX_HP, PROTOCOL_VERSION, VIEW_H, VIEW_W
from game.data import craft as craft_data
from game.data import drops as drops_data
from game.data import elements as elem_data
from game.data import items as item_data
from game.data import monsters as monster_data
from game.data import skills as skill_data
from game.data import stats as stats_data
from game.data import terrain as terrain_data
from game import world_layout
from net.chat import CHAT_KINDS, validate_chat

PARTY_OPS = ("create", "leave", "invite", "accept")
STORAGE_OPS = ("in", "out")
BUFF_NAMES = ["", "Blessing", "Increase AGI", "Angelus", "Endure", "Provoke", "Hiding", "Play Dead"]


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
        out = {"t": t, "name": msg["name"]}
        if "password" in msg:  # accounts_wire.md: optional; absent keeps name-only characters working
            if not isinstance(msg["password"], str):
                return None
            out["password"] = msg["password"]
        if "create" in msg:
            if not isinstance(msg["create"], bool):
                return None
            out["create"] = msg["create"]
        return out
    if t == "input" and _is_int(msg.get("move")) and 0 <= msg["move"] <= 4 \
            and isinstance(msg.get("attack"), bool) and isinstance(msg.get("eat"), bool):
        return {"t": t, "move": msg["move"], "attack": msg["attack"], "eat": msg["eat"]}
    if t == "walk_to" and all(_is_int(msg.get(k)) and 0 <= msg[k] < MAP for k in ("x", "y")):
        return {"t": t, "x": msg["x"], "y": msg["y"]}
    return _validate_ro(t, msg)


def _validate_ro(t, msg: dict) -> dict | None:
    """The RO messages (docs/*_wire.md); None when a field is missing or has the wrong type or range."""
    if t == "allot":
        return {"t": t, "stat": msg["stat"]} if _is_int(msg.get("stat")) and 0 <= msg["stat"] <= 5 else None
    if t == "job":
        return {"t": t, "job": msg["job"]} if _is_int(msg.get("job")) and 1 <= msg["job"] <= 6 else None
    if t == "skill":
        ok = _is_int(msg.get("id")) and 1 <= msg["id"] < skill_data.N_SKILLS             and _is_int(msg.get("target")) and -1 <= msg["target"] < N_ENTITIES
        return {"t": t, "id": msg["id"], "target": msg["target"]} if ok else None
    if t == "learn":
        ok = _is_int(msg.get("id")) and 1 <= msg["id"] < skill_data.N_SKILLS
        return {"t": t, "id": msg["id"]} if ok else None
    if t in ("use_item", "equip", "refine", "drop"):  # any int: the engine ignores slots outside 0..INV_SLOTS-1
        return {"t": t, "slot": msg["slot"]} if _is_int(msg.get("slot")) else None
    if t == "gather":
        return {"t": t}
    if t == "craft":
        ok = _is_int(msg.get("recipe")) and 1 <= msg["recipe"] < craft_data.N_RECIPES
        return {"t": t, "recipe": msg["recipe"]} if ok else None
    if t == "party":
        target = msg.get("target")
        ok = msg.get("op") in PARTY_OPS and (target is None or isinstance(target, str))
        return {"t": t, "op": msg["op"], "target": target} if ok else None
    if t == "chat":
        return validate_chat(msg)
    if t == "storage":
        ok = msg.get("op") in STORAGE_OPS and _is_int(msg.get("slot"))
        return {"t": t, "op": msg["op"], "slot": msg["slot"]} if ok else None
    if t == "save":
        return {"t": t}
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


def build_welcome_static() -> dict:
    """Static tables for `welcome` (docs/*_wire.md). Built once at server start; everything is plain JSON data."""
    sk = skill_data
    arr = {n: np.asarray(getattr(sk, n)) for n in (
        "SKILL_JOB", "SKILL_MAX_LEVEL", "SKILL_EFFECT", "SKILL_ELEMENT", "SKILL_TARGET", "SKILL_AOE", "SKILL_BUFF",
        "SKILL_HITS", "SKILL_SP", "SKILL_CAST", "SKILL_COOLDOWN", "SKILL_RANGE", "SKILL_POWER", "SKILL_BUFF_TICKS")}
    skill_list = []
    for sid in range(1, sk.N_SKILLS):
        maxl = int(arr["SKILL_MAX_LEVEL"][sid])
        levels = [{"sp": int(arr["SKILL_SP"][sid, n]), "cast": int(arr["SKILL_CAST"][sid, n]),
                   "cooldown": int(arr["SKILL_COOLDOWN"][sid, n]), "range": int(arr["SKILL_RANGE"][sid, n]),
                   "power": int(arr["SKILL_POWER"][sid, n]), "buff_ticks": int(arr["SKILL_BUFF_TICKS"][sid, n])}
                  for n in range(1, maxl + 1)]
        skill_list.append({"id": sid, "name": sk.SKILL_NAME[sid], "job": int(arr["SKILL_JOB"][sid]),
                           "max_level": maxl, "effect": int(arr["SKILL_EFFECT"][sid]),
                           "target": int(arr["SKILL_TARGET"][sid]), "element": int(arr["SKILL_ELEMENT"][sid]),
                           "aoe": int(arr["SKILL_AOE"][sid]), "hits": int(arr["SKILL_HITS"][sid]),
                           "buff": int(arr["SKILL_BUFF"][sid]), "levels": levels})
    centers, kafra, station = (world_layout.town_centers().tolist(), world_layout.kafra_tiles().tolist(),
                               world_layout.town_station_tiles().tolist())
    towns = [{"idx": i, "name": world_layout.TOWN_NAMES[i], "role": world_layout.TOWN_ROLES[i],
              "center": centers[i], "kafra": kafra[i], "station": station[i]} for i in range(len(centers))]
    boxes = world_layout.dungeon_boxes().tolist()
    dungeons = [{"idx": i, "name": d[3], "box": boxes[i]} for i, d in enumerate(world_layout.DUNGEONS)]
    src, dst = world_layout.warp_tiles()
    warps = [{"from": a, "to": b} for a, b in zip(src.tolist(), dst.tolist())]
    monsters = [{"kind": k, "name": m.name, "hp": m.hp, "level": m.level, "mvp": bool(m.mvp), "size": int(m.size)}
                for k, m in enumerate(monster_data.MONSTERS)]
    cd = craft_data
    recipes = [{"id": r, "inputs": [int(x) for x in np.asarray(cd.RECIPE_INPUTS[r]) if x > 0],
                "output": int(cd.RECIPE_OUTPUT[r]), "skill": int(cd.RECIPE_SKILL[r]),
                "level": int(cd.RECIPE_LEVEL[r]), "exp": int(cd.RECIPE_EXP[r]), "station": int(cd.RECIPE_STATION[r])}
               for r in range(1, cd.N_RECIPES)]
    nodes = [{"kind": k, "name": cd.NODE_NAMES[k], "skill": int(cd.NODE_SKILL[k]), "level": int(cd.NODE_LEVEL_REQ[k]),
              "product": int(cd.NODE_PRODUCT[k])} for k in range(cd.N_NODE_KINDS)]
    return {
        "world": {"towns": towns, "dungeons": dungeons, "warps": warps,
                  "terrain_names": list(terrain_data.TERRAIN_NAMES), "monsters": monsters},
        "items": item_data.table_for_client(), "drops": drops_data.table_for_client(),
        "skills": {"names": list(sk.SKILL_NAME), "jobs": list(stats_data.JOB_NAMES), "list": skill_list},
        "elements": list(elem_data.ELEM_NAME), "elem_mult": np.asarray(elem_data.ELEM_MULT).tolist(),
        "buffs": list(BUFF_NAMES),
        "craft_items": {str(k): v for k, v in cd.ITEM_NAMES.items()}, "recipes": recipes, "nodes": nodes,
        "craft_skills": {"names": list(cd.SKILL_NAMES), "level_exp": [int(x) for x in cd.LEVEL_EXP]},
        "stat_names": list(stats_data.STAT_NAMES), "job_names": list(stats_data.JOB_NAMES),
        "stat_cost": list(stats_data.STAT_COST), "max_base_level": stats_data.MAX_BASE_LEVEL,
        "max_job_level": list(stats_data.MAX_JOB_LEVEL), "base_exp_len": len(stats_data.BASE_EXP_TABLE),
        "job_exp_len": len(stats_data.JOB_EXP_TABLE[1]),
        "mvp_kinds": [m.name if m.mvp else None for m in monster_data.MONSTERS],
    }


def welcome_message(slot: int, tick_ms: int, maps: tuple[str, str], static: dict | None = None) -> dict:
    """The welcome message; `maps` = (terrain_b64, pz_b64) built once per server with encode_map, `static` =
    build_welcome_static() (the RO tables, built once per server)."""
    msg = {"t": "welcome", "slot": slot, "map": MAP, "tick_ms": tick_ms, "view_w": VIEW_W, "view_h": VIEW_H,
           "terrain": maps[0], "pz": maps[1]}
    if static:
        msg.update(static)
    return msg


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


_BASE_EXP = np.asarray(stats_data.BASE_EXP_TABLE, np.int64)
_JOB_EXP = np.asarray(stats_data.JOB_EXP_TABLE, np.int64)
_MAX_JOB_LEVEL = np.asarray(stats_data.MAX_JOB_LEVEL, np.int64)
_STAT6 = stats_data.STAT_NAMES


def _inv_rows(ids, count, refine, cards) -> list[dict]:
    ids, count, refine, cards = ids.tolist(), count.tolist(), refine.tolist(), cards.tolist()
    return [{"item": ids[i], "count": count[i], "refine": refine[i], "cards": cards[i]} for i in range(INV_SLOTS)]


def _pairs(row: np.ndarray, *more: np.ndarray) -> list[list[int]]:
    """[[index, row[index], more[0][index], ...], ...] for the indices where row > 0, ascending."""
    idx = np.flatnonzero(row > 0)
    cols = [idx.tolist(), row[idx].tolist()] + [m[idx].tolist() for m in more]
    return [list(t) for t in zip(*cols)]


def ro_you(pull: dict, slot: int) -> dict:
    """The RO fields of `snap.you` for one viewer, from the tick's single host pull (pull["ro"], pull["players"])."""
    ro, p = pull["ro"], pull["players"]
    st, sk, it, cr, pa, dv = ro["stats"], ro["skills"], ro["items"], ro["craft"], ro["party"], ro["derived"]
    lvl, job, jlv = int(st["base_level"][slot]), int(st["job"][slot]), int(st["job_level"][slot])
    job = min(max(job, 0), stats_data.NUM_JOBS - 1)
    lvl = min(max(lvl, 1), stats_data.MAX_BASE_LEVEL)
    base_next = int(_BASE_EXP[lvl]) if lvl < stats_data.MAX_BASE_LEVEL else 0
    job_next = int(_JOB_EXP[job][jlv]) if 0 <= jlv < _MAX_JOB_LEVEL[job] else 0
    return {
        "level": lvl, "job_level": jlv, "job": job,
        "stats": [int(st[k][slot]) for k in _STAT6],
        "stat_points": int(st["stat_points"][slot]), "skill_points": int(st["skill_points"][slot]),
        "sp": int(st["sp"][slot]), "max_sp": int(dv["max_sp"][slot]),  # stats.sp is the one SP pool
        "base_exp": int(st["base_exp"][slot]), "base_exp_next": base_next,
        "job_exp": int(st["job_exp"][slot]), "job_exp_next": job_next,
        "max_hp": int(dv["max_hp"][slot]),
        "buffs": _pairs(sk["buffs"][slot], sk["buff_level"][slot]),
        "cooldowns": _pairs(sk["cooldown"][slot]),
        "learned": _pairs(sk["learned"][slot]),
        "casting": int(sk["casting"][slot]), "cast_left": int(sk["cast_left"][slot]),
        "cast_target": int(sk["cast_target"][slot]),
        "zeny": int(it["zeny"][slot]),
        "inv": _inv_rows(p["inv"][slot][:INV_SLOTS], it["inv_count"][slot], it["inv_refine"][slot],
                         it["inv_cards"][slot]),
        "equip": it["equip"][slot].tolist(),
        "craft": {"level": cr["level"][slot].tolist(), "exp": cr["exp"][slot].tolist()},
        "party": int(pa["party_id"][slot]), "leader": bool(pa["leader"][slot]),
    }


def snapshot(pull: dict, slot: int, names: dict, notice: str | None, tick: int, town: int, full: bool,
             you_extra: dict | None = None) -> dict:
    """Build the snap message for one viewer (row `slot`) from the tick's single host pull.

    pull: {"send", "gone": int [MAX_VIEWERS, VIEW_K] with -1 pads, "tab": entity table, "players": arrays, "safe",
    "ro": RO block rows (optional: without it the v1 `you` fields are sent)}. `you_extra` is merged into `you`
    (the server's change-detected `members` / `storage`)."""
    p = pull["players"]
    send, gone = pull["send"][slot], pull["gone"][slot]
    you = {"pos": p["pos"][slot].tolist(), "hp": int(p["hp"][slot]), "max_hp": PLAYER_MAX_HP,
           "kills": int(p["kills"][slot]), "deaths": int(p["deaths"][slot]),
           "inv": p["inv"][slot][:INV_SLOTS].tolist(), "town": int(town), "safe": bool(pull["safe"][slot])}
    ents = entity_rows(send[send >= 0], pull["tab"], names)
    if "ro" in pull:
        you.update(ro_you(pull, slot))
        max_hp = pull["ro"]["derived"]["max_hp"]
        for e in ents:
            if e["kind"] == 0:  # the view table carries the constant PLAYER_MAX_HP for players
                e["max_hp"] = int(max_hp[e["id"]])
    if you_extra:
        you.update(you_extra)
    return {"t": "snap", "tick": tick, "you": you, "ents": ents,
            "gone": gone[gone >= 0].tolist(), "notice": notice, "full": bool(full)}
