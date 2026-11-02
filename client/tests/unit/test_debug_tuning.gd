extends TestCase


func test_part_d_rows_present() -> void:
	var want: Dictionary = {
		"tile_px": [32.0, 16.0, 64.0, 8.0],
		"interp_delay_ticks": [1.0, 0.0, 3.0],
		"camera_zoom": [1.0, 0.5, 3.0],
		"overlay_window_s": [2.0, 0.5, 10.0],
		"name_label_px": [12.0, 8.0, 24.0],
	}
	for field: String in want:
		var info: Dictionary = Tuning.field_info(field)
		assert_false(info.is_empty(), field)
		var w: Array = want[field]
		assert_near(float(info["default"]), float(w[0]), 0.0001, field)
		assert_near(float(info["min"]), float(w[1]), 0.0001, field)
		assert_near(float(info["max"]), float(w[2]), 0.0001, field)
		if w.size() > 3:
			assert_near(float(info["step"]), float(w[3]), 0.0001, field)


func test_rows_clamp_and_reset() -> void:
	var t := Tuning.new()
	t.set_value("tile_px", 500.0)
	assert_near(t.tile_px, 64.0)
	t.set_value("overlay_window_s", 0.0)
	assert_near(t.overlay_window_s, 0.5)
	t.reset()
	assert_near(t.tile_px, 32.0)
