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
