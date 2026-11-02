extends TestCase


func test_move_codes_and_change_flag() -> void:
	var s := InputState.new()
	assert_false(s.update(false, false, false, false, false, false))
	assert_true(s.update(true, false, false, false, false, false))
	assert_eq(s.move, InputState.MOVE_NORTH)
	assert_false(s.update(true, false, false, false, false, false))
	assert_true(s.update(false, true, false, false, false, false))
	assert_eq(s.move, InputState.MOVE_SOUTH)
	assert_true(s.update(false, false, true, false, false, false))
	assert_eq(s.move, InputState.MOVE_WEST)
	assert_true(s.update(false, false, false, true, false, false))
	assert_eq(s.move, InputState.MOVE_EAST)
	assert_true(s.update(false, false, false, false, false, false))
	assert_eq(s.move, InputState.MOVE_NONE)


func test_newest_direction_wins_and_falls_back() -> void:
	var s := InputState.new()
	s.update(true, false, false, false, false, false)
	s.update(true, false, false, true, false, false)
	assert_eq(s.move, InputState.MOVE_EAST)
	s.update(true, false, false, false, false, false)
	assert_eq(s.move, InputState.MOVE_NORTH)


func test_opposite_keys_cancel() -> void:
	var s := InputState.new()
	s.update(true, true, false, false, false, false)
	assert_eq(s.move, InputState.MOVE_NONE)
	s.update(false, false, true, true, false, false)
	assert_eq(s.move, InputState.MOVE_NONE)


func test_eat_tap_sends_press_then_release_once() -> void:
	var s := InputState.new()
	var sends: int = 0
	var frames: Array[bool] = [false, true, true, true, false, false]
	for held: bool in frames:
		if s.update(false, false, false, false, false, held):
			sends += 1
	assert_eq(sends, 2)
	assert_false(s.eat)


func test_attack_flag() -> void:
	var s := InputState.new()
	assert_true(s.update(false, false, false, false, true, false))
	assert_true(s.attack)


func test_click_walk_sends_open_tiles_and_flashes_blocked() -> void:
	var link := FakeLink.new()
	var session := NetSession.new(link)
	session.map.size = 3
	session.map.terrain = PackedByteArray([0, 1, 4, 0, 5, 0, 6, 7, 0])
	session.connect_to("ws://x")
	var pi := PlayerInput.new()
	pi.session = session
	var sent: Array = []
	var flashed: Array = []
	pi.walk_target.connect(func(t: Vector2i) -> void: sent.append(t))
	pi.walk_blocked.connect(func(t: Vector2i) -> void: flashed.append(t))
	assert_true(pi.click_walk(Vector2i(1, 0)))
	assert_false(pi.click_walk(Vector2i(2, 0)))
	assert_false(pi.click_walk(Vector2i(1, 1)))
	assert_false(pi.click_walk(Vector2i(0, 2)))
	assert_false(pi.click_walk(Vector2i(1, 2)))
	assert_false(pi.click_walk(Vector2i(9, 9)))
	assert_eq(link.sent.size(), 1)
	assert_eq(sent, [Vector2i(1, 0)])
	assert_eq(flashed.size(), 4)
	pi.free()
