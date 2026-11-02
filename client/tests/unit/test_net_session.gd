extends TestCase

const BLOB: String = "eJxjYGRi/v+PnZ2dATcDAJZICGE="


func _welcome() -> String:
	return Codec.encode({"t": "welcome", "slot": 2, "map": 6, "tick_ms": 120, "view_w": 21, "view_h": 15,
		"terrain": BLOB, "pz": BLOB})


func test_map_data_from_welcome() -> void:
	var m := MapData.new()
	assert_true(m.from_welcome(Codec.decode(_welcome())))
	assert_eq(m.size, 6)
	assert_eq(m.at(0, 0), 0)
	assert_eq(m.at(1, 0), 1)
	assert_eq(m.at(4, 0), -1)
	assert_eq(m.at(5, 0), -2)
	assert_eq(m.at(-1, 0), -1)
	assert_eq(m.at(6, 0), -1)
	assert_eq(m.at(0, 6), -1)
	assert_true(m.pz_at(4, 0))
	assert_false(m.pz_at(0, 0))
	assert_false(MapData.new().from_welcome({"map": 6, "terrain": "", "pz": ""}))


func test_session_login_flow_over_fake_link() -> void:
	var link := FakeLink.new()
	var s := NetSession.new(link)
	s.connect_to("ws://x")
	s.login("ann")
	s.send_input(4, true, false)
	s.walk_to(7, 9)
	assert_eq(link.sent.size(), 3)
	assert_eq(Codec.decode(link.sent[0])["name"], "ann")
	assert_eq(int(Codec.decode(link.sent[1])["move"]), 4)
	assert_eq(int(Codec.decode(link.sent[2])["y"]), 9)
	link.push_inbound(_welcome())
	link.push_inbound("garbage")
	link.push_inbound(Codec.encode({"t": "snap", "tick": 5, "full": true, "gone": [],
		"you": {"pos": [1, 2], "hp": 3, "max_hp": 4},
		"ents": [{"id": 0, "kind": 0, "pos": [1, 2], "hp": 3, "max_hp": 4, "name": "ann"}]}))
	var got: Array[Dictionary] = s.poll()
	assert_eq(got.size(), 2)
	assert_true(s.welcomed)
	assert_eq(s.slot, 2)
	assert_eq(s.tick_ms, 120)
	assert_eq(s.map.size, 6)
	assert_eq(s.store.tick, 5)
	assert_eq(s.store.ids(), PackedInt32Array([0]))
	assert_eq(s.bad_messages, 1)
	link.push_inbound(Codec.encode({"t": "error", "msg": "full"}))
	s.poll()
	assert_eq(s.last_error, "full")


func test_session_records_traffic() -> void:
	var path: String = "user://test_net_session.jsonl"
	var link := FakeLink.new()
	var s := NetSession.new(link)
	s.recorder = FlightRecorder.new()
	assert_true(s.recorder.start(path))
	s.connect_to("x")
	s.login("bo")
	link.push_inbound(_welcome())
	s.poll()
	s.recorder.stop()
	var lines: Array[Dictionary] = FlightRecorder.read_file(path)
	assert_eq(lines.size(), 2)
	assert_eq(lines[0]["d"], "out")
	assert_eq(lines[1]["d"], "in")
	DirAccess.remove_absolute(ProjectSettings.globalize_path(path))
