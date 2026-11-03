class_name EntityStore
extends RefCounted
## Client copy of the entities the server says we can see. Upsert `ents`, delete `gone`, wipe on `full`.
## Entity dictionary keys: id, kind, pos (Vector2i), prev_pos (Vector2i, position before the last change, for
## interpolation), hp, max_hp, and name (players, kind 0), item (drops, kind >= 100), mkind (monsters).
## `you` keeps the own-player fields of the snap: pos (Vector2i), hp, max_hp, kills, deaths, inv, town, safe.

const KIND_PLAYER: int = 0
const KIND_DROP_BASE: int = 100

var you: Dictionary = {}
var tick: int = 0
var notice: String = ""
var snaps: int = 0
var version: int = 0  ## Bumped by every apply_snap (and clear/restore).
var _changed: PackedInt32Array = PackedInt32Array()  ## Ids upserted by the last apply_snap.
var _removed: PackedInt32Array = PackedInt32Array()  ## Ids deleted by the last apply_snap (full wipe included).
var _ents: Dictionary = {}  ## int id -> Dictionary


## Applies one decoded `snap` message.
func apply_snap(snap: Dictionary) -> void:
	tick = int(snap.get("tick", tick))
	snaps += 1
	version += 1
	_changed = PackedInt32Array()
	_removed = PackedInt32Array()
	if bool(snap.get("full", false)):
		_removed = ids()
		_ents.clear()
	var you_in: Variant = snap.get("you")
	if typeof(you_in) == TYPE_DICTIONARY:
		# Merge over the previous `you`: the server sends change-only fields (members, storage) once.
		var merged: Dictionary = you.duplicate(true)
		merged.merge(_entity_store_norm_you(you_in), true)
		you = merged
	var n: Variant = snap.get("notice")
	notice = String(n) if typeof(n) == TYPE_STRING else ""
	var gone: Variant = snap.get("gone")
	if typeof(gone) == TYPE_ARRAY:
		for g: Variant in gone:
			if _ents.erase(int(g)):
				_removed.append(int(g))
	var ents: Variant = snap.get("ents")
	if typeof(ents) == TYPE_ARRAY:
		for e: Variant in ents:
			if typeof(e) == TYPE_DICTIONARY:
				_upsert(e)


## Ids upserted by the last apply_snap (an id listed twice appears twice).
func changed_ids() -> PackedInt32Array:
	return _changed


## Ids removed by the last apply_snap: existing `gone` entries, plus every old id on a `full` snap. After a full
## snap an id can be in both lists; has_ent() says whether it exists now.
func removed_ids() -> PackedInt32Array:
	return _removed


func get_ent(id: int) -> Dictionary:
	return _ents.get(id, {})


func has_ent(id: int) -> bool:
	return _ents.has(id)


func count() -> int:
	return _ents.size()


## Known ids, ascending.
func ids() -> PackedInt32Array:
	var keys: Array = _ents.keys()
	keys.sort()
	var out := PackedInt32Array()
	for k: Variant in keys:
		out.append(int(k))
	return out


func clear() -> void:
	version += 1
	_removed = ids()
	_changed = PackedInt32Array()
	_ents.clear()
	you = {}
	tick = 0
	notice = ""
	snaps = 0


## Deep copy of the whole store, safe to keep and to feed to restore().
func snapshot() -> Dictionary:
	return {"tick": tick, "snaps": snaps, "notice": notice, "you": you.duplicate(true), "ents": _ents.duplicate(true)}


func restore(d: Dictionary) -> void:
	version += 1
	_removed = ids()
	tick = int(d.get("tick", 0))
	snaps = int(d.get("snaps", 0))
	notice = String(d.get("notice", ""))
	var y: Variant = d.get("you")
	you = (y as Dictionary).duplicate(true) if typeof(y) == TYPE_DICTIONARY else {}
	var e: Variant = d.get("ents")
	_ents = (e as Dictionary).duplicate(true) if typeof(e) == TYPE_DICTIONARY else {}
	_changed = ids()


func _upsert(src: Dictionary) -> void:
	if not src.has("id"):
		return
	var id: int = int(src["id"])
	var pos: Vector2i = _entity_store_pos(src.get("pos"))
	var e: Dictionary = {
		"id": id,
		"kind": int(src.get("kind", 0)),
		"pos": pos,
		"prev_pos": pos,
		"hp": int(src.get("hp", 0)),
		"max_hp": int(src.get("max_hp", 0)),
	}
	if _ents.has(id):
		var old: Dictionary = _ents[id]
		e["prev_pos"] = old["pos"]
	if src.has("name"):
		e["name"] = String(src["name"])
	if src.has("item"):
		e["item"] = int(src["item"])
	if src.has("mkind"):
		e["mkind"] = int(src["mkind"])
	# Anim fields (docs/anim_wire.md): a missing one keeps its old value, except `h` (damage on this tick only).
	var prev: Dictionary = _ents.get(id, {})
	for k: String in ["a", "d", "t", "m", "s", "h", "job"]:
		if src.has(k):
			e[k] = int(src[k])
		elif k != "h" and prev.has(k):
			e[k] = prev[k]
	_ents[id] = e
	_changed.append(id)


func _entity_store_norm_you(src: Dictionary) -> Dictionary:
	var out: Dictionary = src.duplicate(true)
	out["pos"] = _entity_store_pos(src.get("pos"))
	for k: String in ["hp", "max_hp", "kills", "deaths", "town"]:
		if out.has(k) and (typeof(out[k]) == TYPE_FLOAT or typeof(out[k]) == TYPE_INT):
			out[k] = int(out[k])
	return out


static func _entity_store_pos(v: Variant) -> Vector2i:
	if typeof(v) == TYPE_ARRAY and (v as Array).size() >= 2:
		var a: Array = v
		return Vector2i(int(a[0]), int(a[1]))
	if typeof(v) == TYPE_VECTOR2I:
		return v
	return Vector2i.ZERO
