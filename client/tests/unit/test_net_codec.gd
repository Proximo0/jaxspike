extends TestCase
## Blob made with Python zlib.compress(bytes([0,1,2,3,255,254,7,7,7] * 4), 6): a standard zlib stream.

const BLOB: String = "eJxjYGRi/v+PnZ2dATcDAJZICGE="


func test_encode_stamps_version_and_round_trips() -> void:
	var m: Dictionary = Codec.decode(Codec.encode({"t": "input", "move": 3, "attack": true, "eat": false}))
	assert_eq(int(m["v"]), 2)
	assert_eq(m["t"], "input")
	assert_eq(int(m["move"]), 3)
	assert_eq(m["attack"], true)


func test_decode_bad_input_is_empty() -> void:
	assert_true(Codec.decode("").is_empty())
	assert_true(Codec.decode("not json").is_empty())
	assert_true(Codec.decode("[1,2]").is_empty())
	assert_true(Codec.decode('{"t":"snap"}').is_empty())
	assert_true(Codec.decode('{"v":1,"t":"snap"}').is_empty())
	assert_true(Codec.decode('{"v":2.5,"t":"snap"}').is_empty())
	assert_true(Codec.decode('{"v":2}').is_empty())
	assert_false(Codec.decode('{"v":2,"t":"snap"}').is_empty())


func test_decode_map_zlib_blob() -> void:
	var raw: PackedByteArray = Codec.decode_map(BLOB, 36)
	assert_eq(raw.size(), 36)
	assert_eq(raw[4], 255)
	assert_eq(raw[5], 254)
	assert_eq(raw[35], 7)


func test_decode_map_failures_are_empty() -> void:
	assert_true(Codec.decode_map("", 36).is_empty())
	assert_true(Codec.decode_map(BLOB, 0).is_empty())
