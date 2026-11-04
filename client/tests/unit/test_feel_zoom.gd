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


func test_retarget_eases_and_clamps() -> void:
	var z := ZoomControl.new(1.0)
	z.retarget(1.5)
	assert_eq(z.target, 1.5)
	assert_eq(z.current, 1.0, "current is left to ease")
	z.update(1.0 / 60.0)
	assert_true(z.current > 1.0 and z.current < 1.5)
	z.retarget(9.0)
	assert_eq(z.target, ZoomControl.DEFAULT_MAX)
	z.retarget(0.01)
	assert_eq(z.target, ZoomControl.DEFAULT_MIN)


func test_tuning_slider_moves_player_input_zoom() -> void:
	var pi := PlayerInput.new()
	var cam := Camera2D.new()
	pi.camera = cam
	pi._ready()
	pi._ready()  # double _ready must not double-connect
	Tuning.shared().set_value("camera_zoom", 1.5)
	for i: int in range(200):
		pi._process(1.0 / 60.0)
	assert_near(cam.zoom.x, 1.5)
	assert_eq(Tuning.shared().changed.get_connections().size(), 1, "single connection")
	Tuning.shared().reset()
	Tuning.shared().changed.disconnect(pi._on_tuning_changed)
	pi.free()
	cam.free()
