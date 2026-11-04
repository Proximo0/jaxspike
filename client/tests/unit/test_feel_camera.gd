extends TestCase
## CameraFollow: ordinary steps never snap; a warp holds the camera until the sprite arrives or HOLD_S passes.


func test_one_tile_steps_do_not_snap() -> void:
	var c := CameraFollow.new()
	for i: int in range(10):
		assert_false(c.observe(Vector2i(i, 0), i + 1, false, Vector2(i - 1, 0), 0.1))
	assert_eq(c.snaps, 0)


func test_jump_snaps_and_holds_until_arrival() -> void:
	var c := CameraFollow.new()
	c.observe(Vector2i(1, 0), 1, false, Vector2(1, 0), 0.1)
	assert_true(c.observe(Vector2i(6, 0), 2, false, Vector2(1, 0), 0.1), "5 tile jump")
	assert_true(c.observe(Vector2i(6, 0), 2, false, Vector2(3, 0), 0.1), "still travelling")
	assert_false(c.observe(Vector2i(6, 0), 2, false, Vector2(5.8, 0), 0.1), "arrived")
	assert_false(c.observe(Vector2i(6, 0), 3, false, Vector2(6, 0), 0.1))
	assert_eq(c.snaps, 1)


func test_warped_flag_snaps_one_tile() -> void:
	var c := CameraFollow.new()
	c.observe(Vector2i(1, 0), 1, false, Vector2(1, 0), 0.1)
	assert_true(c.observe(Vector2i(2, 0), 2, true, Vector2(0, 0), 0.1))
	assert_eq(c.snaps, 1)


func test_hold_expires() -> void:
	var c := CameraFollow.new()
	c.observe(Vector2i(1, 0), 1, false, Vector2(1, 0), 0.1)
	assert_true(c.observe(Vector2i(9, 0), 2, false, Vector2(1, 0), 0.1))
	var still: bool = true
	for i: int in range(20):
		still = c.observe(Vector2i(9, 0), 2, false, Vector2(1, 0), 0.1)
	assert_false(still, "released after HOLD_S")


func test_skipped_ticks_compare_per_observed_tick() -> void:
	var c := CameraFollow.new()
	c.observe(Vector2i(0, 0), 1, false, Vector2(0, 0), 0.1)
	assert_false(c.observe(Vector2i(2, 0), 5, false, Vector2(0, 0), 0.1), "2 tiles is not over WARP_TILES")
	assert_eq(c.snaps, 0)
