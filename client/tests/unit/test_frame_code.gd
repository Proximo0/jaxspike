extends TestCase


func test_crc16_check_value() -> void:
	# CRC-16/CCITT-FALSE check value for "123456789"
	assert_eq(FrameCode.crc16("123456789".to_ascii_buffer()), 0x29B1)


func test_crc16_empty_is_init() -> void:
	assert_eq(FrameCode.crc16(PackedByteArray()), 0xFFFF)


func test_encode_length_is_14() -> void:
	assert_eq(FrameCode.encode(1, 2).size(), 14)


func test_encode_layout_is_big_endian() -> void:
	var b: PackedByteArray = FrameCode.encode(0x01020304, 0x0102030405060708)
	for i: int in range(4):
		assert_eq(b[i], i + 1, "frame byte %d" % i)
	for i: int in range(8):
		assert_eq(b[4 + i], i + 1, "hash byte %d" % i)
	var payload: PackedByteArray = b.slice(0, 12)
	var crc: int = FrameCode.crc16(payload)
	assert_eq(b[12], (crc >> 8) & 0xFF, "crc high")
	assert_eq(b[13], crc & 0xFF, "crc low")


func test_round_trip() -> void:
	var cases: Array = [
		[0, 0],
		[1, 1],
		[123456, 987654321],
		[0xFFFFFFFF, -1],
		[7, 0x7FFFFFFFFFFFFFFF],
		[42, -0x7FFFFFFFFFFFFFFF - 1],
	]
	for c: Array in cases:
		var d: Dictionary = FrameCode.decode(FrameCode.encode(c[0], c[1]))
		assert_true(d["ok"], "ok for %s" % str(c))
		assert_eq(d["frame"], c[0], "frame for %s" % str(c))
		assert_eq(d["state_hash"], c[1], "hash for %s" % str(c))


func test_frame_is_truncated_to_32_bits() -> void:
	var d: Dictionary = FrameCode.decode(FrameCode.encode(0x1FFFFFFFF, 5))
	assert_true(d["ok"])
	assert_eq(d["frame"], 0xFFFFFFFF)


func test_decode_detects_single_bit_flip() -> void:
	var b: PackedByteArray = FrameCode.encode(900, 12345)
	for bit: int in range(112):
		var c: PackedByteArray = b.duplicate()
		c[bit >> 3] = c[bit >> 3] ^ (1 << (7 - (bit & 7)))
		assert_false(FrameCode.decode(c)["ok"], "flip bit %d" % bit)


func test_decode_rejects_wrong_length() -> void:
	var d: Dictionary = FrameCode.decode(PackedByteArray([1, 2, 3]))
	assert_false(d["ok"])
	assert_eq(d["frame"], 0)
	assert_eq(d["state_hash"], 0)
