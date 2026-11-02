class_name FrameCode
extends RefCounted
## Frame stamp code. 14 bytes = 112 bits:
## frame uint32 big endian (4) + state hash int64 big endian (8) + CRC16-CCITT big endian (2) over the first 12 bytes.
## Pure logic: the debug stamp draws these bits, tools/frame_code.py reads them back from a PNG.

const PAYLOAD_BYTES: int = 12
const BYTE_COUNT: int = 14


static func encode(frame: int, state_hash: int) -> PackedByteArray:
	var out: PackedByteArray = PackedByteArray()
	out.resize(PAYLOAD_BYTES)
	var f: int = frame & 0xFFFFFFFF
	for i: int in range(4):
		out[i] = (f >> (24 - 8 * i)) & 0xFF
	for i: int in range(8):
		out[4 + i] = (state_hash >> (56 - 8 * i)) & 0xFF
	var crc: int = crc16(out)
	out.append((crc >> 8) & 0xFF)
	out.append(crc & 0xFF)
	return out


## Returns {ok: bool, frame: int, state_hash: int}. ok is false on a wrong length or a CRC mismatch;
## frame and state_hash are still filled in when the length is right, so a bad read can be inspected.
static func decode(bytes: PackedByteArray) -> Dictionary:
	if bytes.size() != BYTE_COUNT:
		return {"ok": false, "frame": 0, "state_hash": 0}
	var f: int = 0
	for i: int in range(4):
		f = (f << 8) | bytes[i]
	var h: int = 0
	for i: int in range(8):
		h = (h << 8) | bytes[4 + i]
	var payload: PackedByteArray = bytes.slice(0, PAYLOAD_BYTES)
	var stored: int = (bytes[12] << 8) | bytes[13]
	return {"ok": crc16(payload) == stored, "frame": f, "state_hash": h}


## CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor.
static func crc16(bytes: PackedByteArray) -> int:
	var crc: int = 0xFFFF
	for b: int in bytes:
		crc = crc ^ (b << 8)
		for _bit: int in range(8):
			if (crc & 0x8000) != 0:
				crc = ((crc << 1) ^ 0x1021) & 0xFFFF
			else:
				crc = (crc << 1) & 0xFFFF
	return crc
