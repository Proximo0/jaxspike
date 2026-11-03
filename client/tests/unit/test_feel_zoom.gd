extends TestCase
## ZoomControl: wheel steps clamp to the bounds, the ease converges, snap jumps.


func test_wheel_clamps() -> void:
	var z := ZoomControl.new(1.0, 0.5, 2.0)
	z.wheel(1)
	assert_near(z.target, ZoomControl.STEP)
	for i: int in range(50):
		z.wheel(1)
	assert_near(z.target, 2.0)
	for i: int in range(100):
		z.wheel(-1)
	assert_near(z.target, 0.5)
	assert_eq(ZoomControl.new(9.0, 0.5, 2.0).current, 2.0, "start clamps")


func test_smooth_converges() -> void:
	var z := ZoomControl.new(1.0, 0.5, 2.0)
	z.wheel(3)
	assert_true(z.update(1.0 / 60.0), "still moving after one frame")
	assert_true(z.current > 1.0 and z.current < z.target, "moves part of the way")
	for i: int in range(120):
		z.update(1.0 / 60.0)
	assert_eq(z.current, z.target, "arrives within 2 s")
	assert_false(z.update(1.0 / 60.0))
	z.snap(0.7)
	assert_eq(z.vec(), Vector2(0.7, 0.7))
