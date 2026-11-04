"""Wire protocol v2: codec, message validation, map blobs and per-viewer delta snapshots (numpy, json, zlib)."""
import base64
import json
import struct
import zlib

import jax.numpy as jnp
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
from game import ro_shop, world_layout
from game.view import N_VIEW_ENTITIES, NODE_KIND_BASE
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
        if "format" in msg:  # snap encoding for this connection; absent = "json"
            if msg["format"] not in SNAP_FORMATS:
                return None
            out["format"] = msg["format"]
        return out
    if t == "input" and _is_int(msg.get("move")) and 0 <= msg["move"] <= 8 \
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
    if t in ("use_item", "equip", "refine", "drop", "sell"):  # any int: the engine ignores slots outside 0..INV_SLOTS-1
        return {"t": t, "slot": msg["slot"]} if _is_int(msg.get("slot")) else None
    if t == "buy":  # docs/items_wire.md: item id; the engine ignores ids the town does not stock
        return {"t": t, "item": msg["item"]} if _is_int(msg.get("item")) else None
    if t == "teleport":  # docs/world_wire.md: waybell teleport destination town
        ok = _is_int(msg.get("town")) and 0 <= msg["town"] < len(world_layout.TOWNS)
        return {"t": t, "town": msg["town"]} if ok else None
    if t in ("gather", "sit"):
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
    centers, waybell, station = (world_layout.town_centers().tolist(), world_layout.waybell_tiles().tolist(),
                               world_layout.town_station_tiles().tolist())
    towns = [{"idx": i, "name": world_layout.TOWN_NAMES[i], "role": world_layout.TOWN_ROLES[i],
              "center": centers[i], "waybell": waybell[i], "station": station[i]} for i in range(len(centers))]
    boxes = world_layout.dungeon_boxes().tolist()
    dungeons = [{"idx": i, "name": d[3], "box": boxes[i]} for i, d in enumerate(world_layout.DUNGEONS)]
    src, dst = world_layout.warp_tiles()
    warps = [{"from": a, "to": b} for a, b in zip(src.tolist(), dst.tolist())]
    # ro-world2 (docs/world_wire.md): zones, portals, shop tiles, waybell teleport, save points
    zb = world_layout.zone_boxes().tolist()
    zones = [{"id": i, "name": z[0], "kind": world_layout.ZONE_KINDS[z[1]], "box": zb[i], "levels": list(z[3]),
              "music": z[4]} for i, z in enumerate(world_layout.ZONES)]
    psrc, pdst = world_layout.portal_warp_tiles()
    pzone = np.asarray(world_layout.zone_of(jnp.asarray(np.concatenate([psrc, pdst])))).reshape(2, -1)
    portals = [{"from": a, "to": b, "zones": [int(pzone[0, k]), int(pzone[1, k])]}
               for k, (a, b) in enumerate(zip(psrc.tolist(), pdst.tolist()))]
    st, stown = world_layout.shop_tiles()
    shop_tiles = [{"tile": t, "town": int(k)} for t, k in zip(st.tolist(), stown.tolist())]
    arrive = world_layout.waybell_teleport_dest().tolist()
    waybell_teleport = [{"from": a, "to": b, "ryn": z, "arrive": arrive[b]} for a, b, z in world_layout.WAYBELL_TELEPORT]
    save_points = world_layout.save_point_tiles().tolist()
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
                  "terrain_names": list(terrain_data.TERRAIN_NAMES), "monsters": monsters,
                  "zones": zones, "portals": portals, "shops": shop_tiles, "waybell_teleport": waybell_teleport,
                  "save_points": save_points},
        "shops": ro_shop.stock_for_client(),  # docs/items_wire.md: per-town NPC stock
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


_ANIM_COLS = ("action", "face", "start_tick", "dur_ms", "speed_ms", "hit")
_ANIM_KEYS = ("a", "d", "t", "m", "s")


def entity_rows(ids: np.ndarray, tab: dict, names: dict) -> list[dict]:
    """JSON entities for the global ids (numpy int array, no padding) from the pulled entity table."""
    pos, hp = tab["pos"][ids].tolist(), tab["hp"][ids].tolist()
    max_hp, kind = tab["max_hp"][ids].tolist(), tab["kind"][ids].tolist()
    anim = [tab[k][ids].tolist() for k in _ANIM_COLS] if "action" in tab else None
    out = []
    for i, e in enumerate(ids.tolist()):
        k = kind[i]
        if k >= NODE_KIND_BASE:  # netperf2: resource node rows are {id, kind, pos, hp} (docs/craft_wire.md)
            out.append({"id": e, "kind": k, "pos": pos[i], "hp": hp[i]})
            continue
        ent = {"id": e, "kind": k, "pos": pos[i], "hp": hp[i], "max_hp": max_hp[i]}
        if anim is not None and k < 100:  # docs/anim_wire.md: a, d, t, m, s and h only on a hit this tick
            ent.update(zip(_ANIM_KEYS, (col[i] for col in anim[:5])))
            if anim[5][i]:  # ro3-wire: hit column -1 = a swing missed this tick, sent as h: 0
                ent["h"] = max(anim[5][i], 0)
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
        "ryn": int(it["ryn"][slot]),
        "inv": _inv_rows(p["inv"][slot][:INV_SLOTS], it["inv_count"][slot], it["inv_refine"][slot],
                         it["inv_cards"][slot]),
        "equip": it["equip"][slot].tolist(),
        "craft": {"level": cr["level"][slot].tolist(), "exp": cr["exp"][slot].tolist()},
        "party": int(pa["party_id"][slot]), "leader": bool(pa["leader"][slot]),
        **_anim_you(pull, slot),
    }


def _anim_you(pull: dict, slot: int) -> dict:
    """docs/anim_wire.md: speed_ms from the entity table, amotion_ms from pull["anim"], after_delay (ticks) from the
    skills rows; 0 while the server does not pull them yet."""
    tab, anim, sk = pull.get("tab", {}), pull.get("anim", {}), pull["ro"]["skills"]
    it = pull["ro"]["items"]
    return {"speed_ms": int(tab["speed_ms"][slot]) if "speed_ms" in tab else 0,
            "range": int(it["range"][slot]) if "range" in it else 1,  # docs/items_wire.md: weapon range in cells
            "amotion_ms": int(anim["amotion_ms"][slot]) if "amotion_ms" in anim else 0,
            "after_delay": int(sk["after_delay"][slot]) if "after_delay" in sk else 0}


def you_full(pull: dict, slot: int, town: int) -> dict:
    """Every `snap.you` field for one viewer (v1 fields, then the RO fields when the pull has them)."""
    p = pull["players"]
    you = {"pos": p["pos"][slot].tolist(), "hp": int(p["hp"][slot]), "max_hp": PLAYER_MAX_HP,
           "kills": int(p["kills"][slot]), "deaths": int(p["deaths"][slot]),
           "inv": p["inv"][slot][:INV_SLOTS].tolist(), "town": int(town), "safe": bool(pull["safe"][slot])}
    if "ro" in pull:
        you.update(ro_you(pull, slot))
    return you


def snapshot(pull: dict, slot: int, names: dict, notice: str | None, tick: int, town: int, full: bool,
             you_extra: dict | None = None) -> dict:
    """Build the snap message for one viewer (row `slot`) from the tick's single host pull.

    pull: {"send", "gone": int [MAX_VIEWERS, VIEW_K] with -1 pads, "tab": entity table, "players": arrays, "safe",
    "ro": RO block rows (optional: without it the v1 `you` fields are sent)}. `you_extra` is merged into `you`
    (the server's change-detected `members` / `storage`)."""
    send, gone = pull["send"][slot], pull["gone"][slot]
    you = you_full(pull, slot, town)
    ents = entity_rows(send[send >= 0], pull["tab"], names)
    if "ro" in pull:
        max_hp = pull["ro"]["derived"]["max_hp"]
        for e in ents:
            if e["kind"] == 0:  # the view table carries the constant PLAYER_MAX_HP for players
                e["max_hp"] = int(max_hp[e["id"]])
    if you_extra:
        you.update(you_extra)
    return {"t": "snap", "tick": tick, "you": you, "ents": ents,
            "gone": gone[gone >= 0].tolist(), "notice": notice, "full": bool(full)}


# ----- netperf2: change-only `you`, one batched snap pass, binary snaps (docs/proto_v2_notes.md) -----
SNAP_FORMATS = ("json", "bin")
YOU_ALWAYS = ("pos", "hp")  # in every `you`: the Godot store rebuilds pos from each `you` it merges
_MISSING = object()
_JSON = json.JSONEncoder(separators=(",", ":"))


def you_delta(you: dict, last: dict, full: bool) -> dict:
    """The `you` keys to send: all on `full`, else pos, hp and every key whose value differs from `last` (this
    slot's last-sent values, updated in place). The client merges `you` over its previous copy."""
    if full:
        last.clear()
        last.update(you)
        return you
    out = {}
    for k, v in you.items():
        if k in YOU_ALWAYS or last.get(k, _MISSING) != v:
            out[k] = last[k] = v
    return out


# Binary snap (login "format": "bin"), little-endian, no padding. Header (16 bytes):
#   0 "JS" | 2 u8 layout (1) | 3 u8 flags: bit0 full, bit1 anim columns valid | 4 u32 tick | 8 u16 n_ents |
#   10 u16 n_gone | 12 u32 json_len
# then json_len bytes of UTF-8 JSON {"you": {...}, "notice": str or null, "names": {"<id>": name}} (names: the
# human players among the ents; any other player is "bot"), then n_ents BIN_ROW records, then n_gone u16 ids.
# Decoded like JSON: name for kind 0, mkind = kind - 1 for 1..99, item = kind - 100 for 100..NODE_KIND_BASE-1,
# a/d/t/m/s for kind < 100 when flag bit1 is set, h only when a hit or a miss happened this tick (miss = 0).
BIN_MAGIC = b"JS"
BIN_LAYOUT = 1
BIN_HEADER = np.dtype([("magic", "S2"), ("layout", "u1"), ("flags", "u1"), ("tick", "<u4"), ("n_ents", "<u2"),
                       ("n_gone", "<u2"), ("json_len", "<u4")])
BIN_ROW = np.dtype([("id", "<u2"), ("kind", "<u2"), ("x", "<u2"), ("y", "<u2"), ("hp", "<i4"), ("max_hp", "<i4"),
                    ("t", "<i4"), ("h", "<i4"), ("s", "<i2"), ("m", "<i2"), ("a", "i1"), ("d", "i1")])
_BIN_ANIM = (("a", "action"), ("d", "face"), ("t", "start_tick"), ("m", "dur_ms"), ("s", "speed_ms"), ("h", "hit"))
assert N_VIEW_ENTITIES < 2 ** 16 and BIN_HEADER.itemsize == 16 and BIN_ROW.itemsize == 30
_BIN_PACK = struct.Struct("<2sBBIHHI")  # BIN_HEADER for one snap, packed without numpy


def _bin_table(tab: dict) -> np.ndarray:
    """The whole pulled entity table as BIN_ROW records: one numpy pass per tick, shared by every viewer."""
    rows = np.zeros(tab["kind"].shape[0], BIN_ROW)
    rows["id"] = np.arange(rows.shape[0])
    rows["kind"], rows["hp"], rows["max_hp"] = tab["kind"], tab["hp"], tab["max_hp"]
    rows["x"], rows["y"] = tab["pos"][:, 0], tab["pos"][:, 1]
    if "action" in tab:
        for col, key in _BIN_ANIM:
            rows[col] = tab[key]
    return rows


def build_snaps(pull: dict, views: list, tick: int, names: dict) -> list:
    """Encode one tick's snaps for every viewer in one pass. `views` = [(slot, you, notice, full, fmt)] with `you`
    already reduced by you_delta; returns the messages in the same order (str for "json", bytes for "bin"). Each
    entity row is encoded once per tick and shared by every viewer it is sent to; the per-viewer numpy work (row
    counts, gathers) runs once over all viewers."""
    send_all, gone_all, tab = pull["send"], pull["gone"], pull["tab"]
    n_send = (send_all >= 0).sum(axis=1).tolist()  # rows are compacted: valid ids first, -1 pads after
    n_gone = (gone_all >= 0).sum(axis=1).tolist()
    out = [None] * len(views)
    js = [i for i, v in enumerate(views) if v[4] != "bin"]
    bn = [i for i, v in enumerate(views) if v[4] == "bin"]
    if js:
        rows = send_all[[views[i][0] for i in js]]
        uniq = np.unique(rows[rows >= 0])
        frag = [_JSON.encode(e) for e in entity_rows(uniq, tab, names)]
        where = np.searchsorted(uniq, rows).tolist()  # -1 pads map to junk positions, cut by n_send below
        for j, i in enumerate(js):
            slot, you, notice, full, _ = views[i]
            ents = ",".join([frag[k] for k in where[j][:n_send[slot]]])
            out[i] = (f'{{"t":"snap","tick":{tick},"you":{_JSON.encode(you)},"ents":[{ents}],'
                      f'"gone":{_JSON.encode(gone_all[slot, :n_gone[slot]].tolist())},"notice":{_JSON.encode(notice)},'
                      f'"full":{"true" if full else "false"},"v":{PROTOCOL_VERSION}}}')
    if bn:
        slots = [views[i][0] for i in bn]
        rows = _bin_table(tab)[send_all[slots]]  # [B, VIEW_K] records; -1 pads gather the last row, cut below
        gone_u2 = gone_all[slots].astype("<u2")
        human = np.zeros(tab["kind"].shape[0], bool)
        human[[e for e in names if 0 <= e < human.shape[0]]] = True
        named = [send_all[sl, :n_send[sl]] for sl in slots]
        anim_flag = 2 if "action" in tab else 0
        for j, i in enumerate(bn):
            slot, you, notice, full, _ = views[i]
            ids, n, ng = named[j], n_send[slot], n_gone[slot]
            body = _JSON.encode({"you": you, "notice": notice,
                                 "names": {str(e): names[e] for e in ids[human[ids]].tolist()}}).encode()
            hdr = _BIN_PACK.pack(BIN_MAGIC, BIN_LAYOUT, int(bool(full)) | anim_flag, tick, n, ng, len(body))
            out[i] = b"".join((hdr, body, rows[j, :n].tobytes(), gone_u2[j, :ng].tobytes()))
    return out


def decode_snap_bin(data: bytes) -> dict | None:
    """Inverse of the "bin" encoding: the same dict a JSON snap decodes to; None on a bad or truncated buffer."""
    if len(data) < BIN_HEADER.itemsize:
        return None
    h = np.frombuffer(data, BIN_HEADER, 1)[0]
    if bytes(h["magic"]) != BIN_MAGIC or int(h["layout"]) != BIN_LAYOUT:
        return None
    n, ng, jl, off = int(h["n_ents"]), int(h["n_gone"]), int(h["json_len"]), BIN_HEADER.itemsize
    if len(data) != off + jl + n * BIN_ROW.itemsize + 2 * ng:
        return None
    try:
        head = json.loads(data[off:off + jl].decode("utf-8"))
    except ValueError:
        return None
    if not isinstance(head, dict):
        return None
    rows = np.frombuffer(data, BIN_ROW, n, off + jl)
    gone = np.frombuffer(data, "<u2", ng, off + jl + n * BIN_ROW.itemsize)
    anim, names, ents = bool(h["flags"] & 2), head.get("names", {}), []
    for e_id, k, x, y, hp, max_hp, t, hit, s, m, a, d in rows.tolist():
        if k >= NODE_KIND_BASE:
            ents.append({"id": e_id, "kind": k, "pos": [x, y], "hp": hp})
            continue
        ent = {"id": e_id, "kind": k, "pos": [x, y], "hp": hp, "max_hp": max_hp}
        if anim and k < 100:
            ent.update(a=a, d=d, t=t, m=m, s=s)
            if hit:
                ent["h"] = max(hit, 0)  # -1 (a miss) decodes as h: 0
        if k == 0:
            ent["name"] = names.get(str(e_id), "bot")
        elif k < 100:
            ent["mkind"] = k - 1
        else:
            ent["item"] = k - 100
        ents.append(ent)
    return {"t": "snap", "v": PROTOCOL_VERSION, "tick": int(h["tick"]), "you": head.get("you", {}), "ents": ents,
            "gone": gone.tolist(), "notice": head.get("notice"), "full": bool(h["flags"] & 1)}
