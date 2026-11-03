extends TestCase


func test_step_ms_straight_and_diagonal() -> void:
	assert_near(CellLerp.step_ms(Vector2(0, 0), Vector2(1, 0), 150.0), 150.0)
	assert_near(CellLerp.step_ms(Vector2(0, 0), Vector2(1, 1), 150.0), 210.0)
	assert_near(CellLerp.step_ms(Vector2(0, 0), Vector2(2, 1), 150.0), 360.0)


func test_first_push_places() -> void:
	var l := CellLerp.new()
	l.push(Vector2(5, 5), 1, 150.0, 100.0, 0.0)
	assert_eq(l.sample(0.0), Vector2(5, 5))
	assert_false(l.moving(0.0))


## RO accumulator at a 100 ms tick and 150 ms/cell steps on ticks 2, 3, 5, 6, 8, 9: the drawn walk is uniform.
func test_speed_150_at_tick_100_is_uniform() -> void:
	var l := CellLerp.new()
	l.push(Vector2(0, 0), 0, 150.0, 100.0, -100.0)
	var steps: Array[int] = [2, 3, 5, 6, 8, 9]
	var x: int = 0
	var ms: float = 100.0
	while ms <= 1000.0:
		# a snap of tick T is drawn from render time (T - 1) * tick_ms on (1 tick of delay)
		for t: int in steps:
			if is_equal_approx(float(t - 1) * 100.0, ms):
				x += 1
				l.push(Vector2(x, 0), t, 150.0, 100.0, ms)
		assert_near(l.sample(ms).x, (ms - 100.0) / 150.0, 0.001, "at %f" % ms)
		ms += 25.0


func test_diagonal_takes_1_4() -> void:
	var l := CellLerp.new()
	l.push(Vector2(0, 0), 0, 100.0, 100.0, 0.0)
	l.push(Vector2(1, 1), 1, 100.0, 100.0, 0.0)
	assert_near(l.sample(70.0).x, 0.5, 0.001)
	assert_true(l.moving(139.0))
	assert_false(l.moving(141.0))


func test_late_step_starts_now_without_jump() -> void:
	var l := CellLerp.new()
	l.push(Vector2(0, 0), 0, 150.0, 100.0, 0.0)
	l.push(Vector2(1, 0), 2, 150.0, 100.0, 180.0)  # scheduled at 100, seen at 180
	assert_near(l.sample(180.0).x, 0.0)
	assert_near(l.sample(255.0).x, 0.5, 0.001)


func test_teleport_snaps() -> void:
	var l := CellLerp.new()
	l.push(Vector2(0, 0), 0, 150.0, 100.0, 0.0)
	l.push(Vector2(40, 40), 1, 150.0, 100.0, 0.0)
	assert_eq(l.sample(0.0), Vector2(40, 40))


func test_step_delta_and_queue_bound() -> void:
	var l := CellLerp.new()
	l.push(Vector2(0, 0), 0, 150.0, 100.0, 0.0)
	l.push(Vector2(-1, 1), 1, 150.0, 100.0, 0.0)
	assert_eq(l.step_delta(10.0), Vector2(-1, 1))
	for t: int in range(2, 30):
		l.push(Vector2(-1, 1 + t), t, 150.0, 100.0, 0.0)
	assert_true(l.pending() <= CellLerp.MAX_SEGMENTS)
