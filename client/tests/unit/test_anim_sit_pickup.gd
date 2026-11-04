extends TestCase
## Sit (a = 2) and pickup (a = 3) through EntityTrack.clip_at and ClipTiming, and flinch position continuity.

const TICK: float = 100.0


func _track(a: int, m: int) -> EntityTrack:
	var t := EntityTrack.new()
	t.feed({"pos": [4, 4], "a": 0, "d": 0, "t": 0, "m": 0, "s": 150}, 0, TICK, 0.0, TICK)
	t.feed({"pos": [4, 4], "a": a, "d": 2, "t": 5, "m": m, "s": 150}, 5, TICK, 500.0, TICK)
	return t


func test_sit_loops_and_is_not_replaced_by_idle_or_walk() -> void:
	var t: EntityTrack = _track(ClipSelect.SIT, 0)
	for ms: float in [500.0, 800.0, 5000.0, 60000.0]:
		var c: Dictionary = t.clip_at(ms, TICK, TICK)
		assert_eq(int(c["action"]), ClipSelect.SIT, "sit at %f" % ms)
		assert_near(float(c["elapsed_ms"]), ms - 500.0)
	var info: Dictionary = PlaceholderClips.clip(ClipSelect.SIT, true)
	var frames: int = int(info["frames"])
	var delay: float = float(info["delay_ms"])
	assert_eq(ClipTiming.frame_at(ClipSelect.SIT, 99999.0, frames, delay, 0.0, 150.0), int(floorf(99999.0 / delay)) % frames)
	assert_true(ClipSelect.loops(ClipSelect.SIT))
	assert_eq(ClipSelect.clip_name(ClipSelect.SIT, true), "sit")


func test_sit_with_a_step_in_flight_keeps_sit() -> void:
	var t: EntityTrack = _track(ClipSelect.SIT, 0)
	t.feed({"pos": [5, 4]}, 6, TICK, 600.0, TICK)
	assert_true(t.cells.moving(650.0))
	assert_eq(int(t.clip_at(650.0, TICK, TICK)["action"]), ClipSelect.SIT)


func test_monster_sit_falls_back_to_idle_clip() -> void:
	assert_eq(ClipSelect.clip_name(ClipSelect.SIT, false), "idle")
	assert_eq(ClipSelect.clip_action(ClipSelect.SIT, false), ClipSelect.IDLE)


func test_pickup_with_dur_is_fitted() -> void:
	var t: EntityTrack = _track(ClipSelect.PICKUP, 400)
	var c: Dictionary = t.clip_at(700.0, TICK, TICK)
	assert_eq(int(c["action"]), ClipSelect.PICKUP)
	assert_near(float(c["dur_ms"]), 400.0)
	assert_eq(ClipTiming.frame_at(ClipSelect.PICKUP, 0.0, 4, 100.0, 400.0, 150.0), 0)
	assert_eq(ClipTiming.frame_at(ClipSelect.PICKUP, 399.0, 4, 100.0, 400.0, 150.0), 3)
	assert_eq(ClipTiming.frame_at(ClipSelect.PICKUP, 400.0, 4, 100.0, 400.0, 150.0), -1)
	assert_eq(ClipTiming.follow_up(ClipSelect.PICKUP), ClipSelect.IDLE)


func test_pickup_without_dur_uses_frames_times_delay() -> void:
	var t: EntityTrack = _track(ClipSelect.PICKUP, 0)
	assert_eq(int(t.clip_at(900.0, TICK, TICK)["action"]), ClipSelect.PICKUP)
	assert_near(ClipTiming.one_shot_ms(ClipSelect.PICKUP, 4, 120.0, 0.0), 480.0)
	assert_eq(ClipTiming.frame_at(ClipSelect.PICKUP, 119.0, 4, 120.0, 0.0, 150.0), 0)
	assert_eq(ClipTiming.frame_at(ClipSelect.PICKUP, 120.0, 4, 120.0, 0.0, 150.0), 1)
	assert_eq(ClipTiming.frame_at(ClipSelect.PICKUP, 479.0, 4, 120.0, 0.0, 150.0), 3)
	assert_eq(ClipTiming.frame_at(ClipSelect.PICKUP, 480.0, 4, 120.0, 0.0, 150.0), -1)


func test_clip_at_dict_is_pooled_per_track() -> void:
	var t: EntityTrack = _track(ClipSelect.SIT, 0)
	var a: Dictionary = t.clip_at(600.0, TICK, TICK)
	var b: Dictionary = t.clip_at(700.0, TICK, TICK)
	assert_true(a == b and is_same(a, b), "same pooled dictionary")
	assert_near(float(b["elapsed_ms"]), 200.0)


func test_hurt_during_walk_keeps_position_continuous() -> void:
	var t := EntityTrack.new()
	var speed: float = 150.0
	t.feed({"pos": [0, 0], "a": 0, "d": 6, "t": 0, "m": 0, "s": speed}, 0, TICK, 0.0, TICK)
	t.feed({"pos": [1, 0], "a": 1, "d": 6, "t": 1, "m": 0, "s": speed}, 1, TICK, 100.0, TICK)
	t.feed({"pos": [2, 0], "a": 1, "d": 6, "t": 1, "m": 0, "s": speed}, 3, TICK, 300.0, TICK)
	t.feed({"pos": [3, 0], "a": 6, "d": 6, "t": 5, "m": 300, "s": speed, "h": 5}, 5, TICK, 500.0, TICK)
	t.flinch_ms = 560.0  # the layer delays the flinch to the attacker's atk frame
	var dt: float = 1000.0 / 60.0
	var last: Vector2 = t.pos_at(500.0)
	var now: float = 500.0 + dt
	var limit: float = 1.2 * dt / speed
	var saw_hurt: bool = false
	while now <= 900.0:
		var p: Vector2 = t.pos_at(now)
		assert_true(p.distance_to(last) <= limit, "position jump %f at %f" % [p.distance_to(last), now])
		var c: Dictionary = t.clip_at(now, TICK, TICK)
		if int(c["action"]) == ClipSelect.HURT:
			saw_hurt = true
		last = p
		now += dt
	assert_true(saw_hurt, "the hurt clip played")
	assert_true(t.pos_at(900.0).x > 2.0, "the walk interpolation kept running")
