class_name Codec
extends RefCounted
## JSON wire codec for protocol v2 plus the one-shot map blob decoder.

const PROTOCOL_VERSION: int = 2


## Encodes a message, stamping "v" = PROTOCOL_VERSION.
static func encode(msg: Dictionary) -> String:
	var m: Dictionary = msg.duplicate()
	m["v"] = PROTOCOL_VERSION
	return JSON.stringify(m)


## Parses one message. Returns {} on bad JSON, a non-object, a wrong/missing "v" or a missing string "t".
static func decode(text: String) -> Dictionary:
	var json := JSON.new()
	if json.parse(text) != OK:
		return {}
	var data: Variant = json.data
	if typeof(data) != TYPE_DICTIONARY:
		return {}
	var d: Dictionary = data
	var v: Variant = d.get("v")
	if typeof(v) != TYPE_FLOAT and typeof(v) != TYPE_INT:
		return {}
	if int(v) != PROTOCOL_VERSION or float(v) != float(PROTOCOL_VERSION):
		return {}
	if typeof(d.get("t")) != TYPE_STRING:
		return {}
	return d


## base64 -> zlib stream (RFC 1950, what Python's zlib.compress emits) -> `size` raw bytes. Empty on any failure
## or when the inflated length differs from `size`.
static func decode_map(b64: String, size: int) -> PackedByteArray:
	if b64.is_empty() or size <= 0:
		return PackedByteArray()
	var packed: PackedByteArray = Marshalls.base64_to_raw(b64)
	if packed.is_empty():
		return PackedByteArray()
	var raw: PackedByteArray = packed.decompress(size, FileAccess.COMPRESSION_DEFLATE)
	if raw.size() != size:
		return PackedByteArray()
	return raw


# Binary snap ("format": "bin" at login; net/protocol.py BIN_HEADER / BIN_ROW), all little-endian:
# header 16 bytes ("JS", layout u8, flags u8 [bit0 full, bit1 anim], tick u32, n_ents u16, n_gone u16, json_len u32),
# then UTF-8 JSON {you, notice, names}, then n_ents rows of 30 bytes, then n_gone u16 ids.
const BIN_LAYOUT: int = 1
const BIN_HEADER_BYTES: int = 16
const BIN_ROW_BYTES: int = 30
const KIND_DROP_BASE: int = 100
const KIND_NODE_BASE: int = 400  ## game/view.py NODE_KIND_BASE (= 100 + ITEM_ID_LIMIT)


## Decodes one binary snap frame into the same Dictionary Codec.decode gives for the JSON snap (keys t, v, tick,
## full, you, notice, ents, gone; entity keys id, kind, pos, hp, max_hp, name/mkind/item, a/d/t/m/s, h).
## Returns {} on a bad magic or layout, a size mismatch or bad JSON.
static func decode_snap_bin(data: PackedByteArray) -> Dictionary:
	if data.size() < BIN_HEADER_BYTES or data[0] != 0x4A or data[1] != 0x53 or data[2] != BIN_LAYOUT:
		return {}
	var flags: int = data[3]
	var n: int = data.decode_u16(8)
	var n_gone: int = data.decode_u16(10)
	var json_len: int = data.decode_u32(12)
	if data.size() != BIN_HEADER_BYTES + json_len + n * BIN_ROW_BYTES + 2 * n_gone:
		return {}
	var json := JSON.new()
	if json.parse(data.slice(BIN_HEADER_BYTES, BIN_HEADER_BYTES + json_len).get_string_from_utf8()) != OK:
		return {}
	if typeof(json.data) != TYPE_DICTIONARY:
		return {}
	var head: Dictionary = json.data
	var names: Dictionary = head.get("names", {}) if typeof(head.get("names")) == TYPE_DICTIONARY else {}
	var anim: bool = (flags & 2) != 0
	var ents: Array = []
	var off: int = BIN_HEADER_BYTES + json_len
	for i: int in n:
		ents.append(_bin_row(data, off, anim, names, head.get("pk", [])))
		off += BIN_ROW_BYTES
	var gone: Array = []
	for i: int in n_gone:
		gone.append(data.decode_u16(off + 2 * i))
	return {
		"t": "snap", "v": PROTOCOL_VERSION, "tick": data.decode_u32(4), "full": (flags & 1) != 0,
		"you": head.get("you", {}), "notice": head.get("notice"), "ents": ents, "gone": gone,
	}


static func _bin_row(data: PackedByteArray, off: int, anim: bool, names: Dictionary, pk: Array = []) -> Dictionary:
	var id: int = data.decode_u16(off)
	var kind: int = data.decode_u16(off + 2)
	var e: Dictionary = {"id": id, "kind": kind, "pos": [data.decode_u16(off + 4), data.decode_u16(off + 6)],
		"hp": data.decode_s32(off + 8)}
	if kind >= KIND_NODE_BASE:
		return e
	e["max_hp"] = data.decode_s32(off + 12)
	if anim and kind < KIND_DROP_BASE:
		e["t"] = data.decode_s32(off + 16)
		var hit: int = data.decode_s32(off + 20)
		if hit != 0:
			e["h"] = hit if hit < -1 else maxi(hit, 0)  # -1 is the miss marker (h = 0); h <= -2 is a crit of |h|
		e["s"] = data.decode_s16(off + 24)
		e["m"] = data.decode_s16(off + 26)
		e["a"] = data.decode_s8(off + 28)
		e["d"] = data.decode_s8(off + 29)
	if kind == 0:
		e["name"] = String(names.get(str(id), "bot"))
		if pk.has(id) or pk.has(float(id)):  # ro5-pvp2: PK flag travels in the JSON head (numbers arrive as floats)
			e["k"] = 1
	elif kind < KIND_DROP_BASE:
		e["mkind"] = kind - 1
	else:
		e["item"] = kind - KIND_DROP_BASE
	return e
