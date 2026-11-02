extends TestCase


func test_first_target_places_at_once() -> void:
	var t := EntityTrack.new()
	t.set_target(Vector2(5, 5), 1000.0, 120.0)
	assert_eq(t.sample(1000.0), Vector2(5, 5))


func test_interpolates_over_duration() -> void:
	var t := EntityTrack.new()
	t.set_target(Vector2(5, 5), 0.0, 120.0)
	t.set_target(Vector2(6, 5), 1000.0, 120.0)
	assert_eq(t.sample(1000.0), Vector2(5, 5))
	assert_near(t.sample(1060.0).x, 5.5)
	assert_eq(t.sample(1120.0), Vector2(6, 5))
	assert_eq(t.sample(5000.0), Vector2(6, 5))
	assert_true(t.settled(1120.0))
	assert_false(t.settled(1050.0))


func test_retarget_starts_from_drawn_position() -> void:
	var t := EntityTrack.new()
	t.set_target(Vector2(0, 0), 0.0, 100.0)
	t.set_target(Vector2(1, 0), 0.0, 100.0)
	t.set_target(Vector2(2, 0), 50.0, 100.0)
	assert_near(t.sample(50.0).x, 0.5)
	assert_near(t.sample(150.0).x, 2.0)


func test_same_target_keeps_motion() -> void:
	var t := EntityTrack.new()
	t.set_target(Vector2(0, 0), 0.0, 100.0)
	t.set_target(Vector2(1, 0), 0.0, 100.0)
	t.set_target(Vector2(1, 0), 50.0, 100.0)
	assert_near(t.sample(50.0).x, 0.5)


func test_zero_delay_and_teleport_snap() -> void:
	var t := EntityTrack.new()
	t.set_target(Vector2(0, 0), 0.0, 0.0)
	t.set_target(Vector2(1, 0), 10.0, 0.0)
	assert_eq(t.sample(10.0), Vector2(1, 0))
	t.set_target(Vector2(50, 50), 20.0, 120.0)
	assert_eq(t.sample(20.0), Vector2(50, 50))
