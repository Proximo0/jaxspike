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


## Made by net.protocol.build_snaps (format "bin"): player 0 "me" walking, bot player 1, monster 2 (kind 3) hit for
## 12, drop 3 (item 5), node 4 (kind 404); gone [7]; tick 99, full, notice "hi", you {pos [1, 2], hp 5}.
const SNAP_BIN: String = "SlMBA2MAAAAFAAEAPQAAAHsieW91Ijp7InBvcyI6WzEsMl0sImhwIjo1fSwibm90aWNlIjoiaGkiLCJuYW1lcyI6eyIwIjoibWUifX0AAAAAAQACAAUAAAAyAAAAWgAAAAAAAACWAAAAAQIBAAAAAwAEAAAAAAAAAAAAAAAAAAAAAACWAAAAAAACAAMABQAGACgAAAA8AAAAXwAAAAwAAADIALwCBQcDAGkABwAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAJQBCQAKAAEAAAABAAAAAAAAAAAAAAAAAAAAAAAHAA=="


func test_decode_snap_bin_matches_the_json_shape() -> void:
	var s: Dictionary = Codec.decode_snap_bin(Marshalls.base64_to_raw(SNAP_BIN))
	assert_eq(s["t"], "snap")
	assert_eq(int(s["tick"]), 99)
	assert_eq(s["full"], true)
	assert_eq(s["notice"], "hi")
	assert_eq(s["gone"], [7])
	assert_eq(int(s["you"]["hp"]), 5)
	var ents: Array = s["ents"]
	assert_eq(ents.size(), 5)
	assert_eq(ents[0], {"id": 0, "kind": 0, "pos": [1, 2], "hp": 5, "max_hp": 50, "a": 1, "d": 2, "t": 90, "m": 0,
		"s": 150, "name": "me"})
	assert_eq(ents[1]["name"], "bot")
	assert_eq(ents[2]["mkind"], 2)
	assert_eq(ents[2]["h"], 12)
	assert_eq(ents[2]["m"], 700)
	assert_false(ents[0].has("h"))
	assert_eq(ents[3], {"id": 3, "kind": 105, "pos": [7, 8], "hp": 0, "max_hp": 0, "item": 5})
	assert_eq(ents[4], {"id": 4, "kind": 404, "pos": [9, 10], "hp": 1})


func test_decode_snap_bin_miss_marker_is_zero() -> void:
	var raw: PackedByteArray = Marshalls.base64_to_raw(SNAP_BIN)
	var row2: int = 16 + raw.decode_u32(12) + 60
	assert_eq(raw.decode_s32(row2 + 20), 12)
	raw.encode_s32(row2 + 20, -1)
	var s: Dictionary = Codec.decode_snap_bin(raw)
	assert_eq(int(s["ents"][2]["h"]), 0)


func test_decode_snap_bin_rejects_bad_frames() -> void:
	var raw: PackedByteArray = Marshalls.base64_to_raw(SNAP_BIN)
	assert_true(Codec.decode_snap_bin(PackedByteArray()).is_empty())
	assert_true(Codec.decode_snap_bin(raw.slice(0, raw.size() - 1)).is_empty())
	var bad: PackedByteArray = raw.duplicate()
	bad[0] = 0x58
	assert_true(Codec.decode_snap_bin(bad).is_empty())


func test_store_merges_you_over_change_only_snaps() -> void:
	var store := EntityStore.new()
	store.apply_snap({"t": "snap", "tick": 1, "full": true, "ents": [], "gone": [],
		"you": {"pos": [1, 2], "hp": 5, "inv": [1, 2], "zeny": 3}})
	store.apply_snap({"t": "snap", "tick": 2, "full": false, "ents": [], "gone": [],
		"you": {"pos": [1, 3], "hp": 4}})
	assert_eq(store.you["pos"], Vector2i(1, 3))
	assert_eq(int(store.you["hp"]), 4)
	assert_eq(store.you["inv"], [1, 2])
	assert_eq(int(store.you["zeny"]), 3)
	store.apply_snap({"t": "snap", "tick": 3, "full": false, "ents": [], "gone": [],
		"you": {"pos": [1, 3], "hp": 4, "zeny": 9}})
	assert_eq(int(store.you["zeny"]), 9)
	assert_eq(store.you["inv"], [1, 2])
