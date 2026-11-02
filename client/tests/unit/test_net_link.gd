extends TestCase


func test_fake_link_lag_ordering() -> void:
	var l := FakeLink.new()
	l.connect_to("x")
	l.push_inbound("late", 2)
	l.push_inbound("a", 0)
	l.push_inbound("b", 0)
	l.push_inbound("mid", 1)
	assert_eq(l.poll(), ["a", "b"] as Array[String])
	l.tick()
	assert_eq(l.poll(), ["mid"] as Array[String])
	assert_true(l.poll().is_empty())
	l.tick()
	assert_eq(l.poll(), ["late"] as Array[String])
	assert_eq(l.pending(), 0)


func test_fake_link_records_outbound_only_when_open() -> void:
	var l := FakeLink.new()
	l.send("dropped")
	assert_eq(l.state(), NetLink.STATE_CLOSED)
	l.connect_to("x")
	l.send("kept")
	assert_eq(l.sent, ["kept"] as Array[String])
	l.close()
	assert_eq(l.state(), NetLink.STATE_CLOSED)


func test_ws_link_unconnected_is_closed() -> void:
	var l := WsLink.new()
	assert_eq(l.state(), NetLink.STATE_CLOSED)
	assert_true(l.poll().is_empty())
	l.send("x")
	l.close()


func test_recorder_round_trip_and_replay() -> void:
	var path: String = "user://test_net_recorder.jsonl"
	var r := FlightRecorder.new()
	assert_true(r.start(path))
	assert_true(r.is_recording())
	r.record_in(1, '{"v":2,"t":"snap"}')
	r.record_out(1, "out1")
	r.record_in(3, "later")
	r.stop()
	assert_false(r.is_recording())
	assert_eq(r.lines_written, 3)
	var lines: Array[Dictionary] = FlightRecorder.read_file(path)
	assert_eq(lines.size(), 3)
	assert_eq(lines[0], {"f": 1, "d": "in", "m": '{"v":2,"t":"snap"}'})
	assert_eq(lines[1]["d"], "out")
	var rp := ReplayLink.new()
	assert_eq(rp.load_file(path), 2)
	rp.connect_to("")
	rp.tick()
	assert_eq(rp.poll(), ['{"v":2,"t":"snap"}'] as Array[String])
	assert_true(rp.poll().is_empty())
	assert_false(rp.finished())
	rp.tick()
	rp.tick()
	assert_eq(rp.poll(), ["later"] as Array[String])
	assert_true(rp.finished())
	DirAccess.remove_absolute(ProjectSettings.globalize_path(path))


func test_recorder_bad_path_fails() -> void:
	assert_false(FlightRecorder.new().start("user://no_such_dir_zz/x.jsonl"))
	assert_true(FlightRecorder.read_file("user://missing_zz.jsonl").is_empty())
