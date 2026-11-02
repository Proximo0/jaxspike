extends TestCase


func _arr(v: Array) -> PackedFloat64Array:
	return PackedFloat64Array(v)


func test_percentile_nearest_rank() -> void:
	var a: PackedFloat64Array = _arr([5.0, 1.0, 3.0, 2.0, 4.0])
	assert_near(DebugFrame.percentile(a, 50.0), 3.0)
	assert_near(DebugFrame.percentile(a, 95.0), 5.0)
	assert_near(DebugFrame.percentile(a, 0.0), 1.0)
	assert_near(DebugFrame.percentile(a, 100.0), 5.0)
	assert_eq(a[0], 5.0, "input untouched")


func test_percentile_hundred_values() -> void:
	var a := PackedFloat64Array()
	for i: int in range(1, 101):
		a.append(float(i))
	assert_near(DebugFrame.percentile(a, 95.0), 95.0)
	assert_near(DebugFrame.percentile(a, 50.0), 50.0)


func test_percentile_empty_is_zero() -> void:
	assert_near(DebugFrame.percentile(PackedFloat64Array(), 95.0), 0.0)


func test_frame_stats() -> void:
	var s: Dictionary = DebugFrame.frame_stats(_arr([16.0, 17.0, 16.0, 40.0]))
	assert_near(float(s["max"]), 40.0)
	assert_near(float(s["p50"]), 16.0)
	assert_near(float(s["p95"]), 40.0)
	assert_eq(int(s["n"]), 4)


func test_fps_from_ms() -> void:
	assert_near(DebugFrame.fps_from_ms(16.0), 62.5)
	assert_near(DebugFrame.fps_from_ms(0.0), 0.0)
