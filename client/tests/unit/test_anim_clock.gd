extends TestCase


func test_render_tick_is_newest_minus_delay() -> void:
	var c := RenderClock.new(100.0, 1.0)
	c.observe(10, 5000.0)
	assert_near(c.render_tick(5000.0), 9.0)
	assert_near(c.render_tick(5050.0), 9.5)
	assert_near(c.drift_ticks(5000.0), 0.0)


func test_never_runs_past_newest_tick() -> void:
	var c := RenderClock.new(100.0, 1.0)
	c.observe(10, 0.0)
	assert_near(c.render_tick(1000.0), 10.0)


func test_late_snap_is_jitter_early_snap_pulls() -> void:
	var c := RenderClock.new(100.0, 1.0)
	c.observe(10, 1000.0)
	c.observe(11, 1130.0)  # 30 ms late: the base stays, the jitter margin grows slowly
	assert_near(c.peak_late_ms, 30.0, 0.5)
	var r: float = c.render_tick(1130.0)
	assert_true(r > 10.2 and r < 10.3, "render %f" % r)
	c.observe(12, 1180.0)  # 20 ms early: the base moves up at once
	assert_true(c.render_tick(1180.0) > 10.7)


func test_margin_slews_without_jumps() -> void:
	var c := RenderClock.new(100.0, 1.0)
	var last: float = -INF
	for t: int in range(1, 100):
		c.observe(t, float(t) * 100.0 + (60.0 if t % 5 == 0 else 0.0))
		for k: int in range(6):
			var now: float = float(t) * 100.0 + float(k) * 16.0
			var r: float = c.render_ms(now)
			if last > -INF:
				assert_true(r >= last, "never backwards at %f" % now)
			last = r
	assert_true(c.margin_ms > 20.0, "margin grew to cover the late snaps")


func test_steady_stream_has_no_drift() -> void:
	var c := RenderClock.new(100.0, 1.0)
	for t: int in range(1, 200):
		var jitter: float = float((t * 37) % 23)
		c.observe(t, float(t) * 100.0 + jitter)
		assert_true(absf(c.drift_ticks(float(t) * 100.0 + jitter)) < 1.0, "tick %d" % t)


func test_long_pause_reanchors() -> void:
	var c := RenderClock.new(100.0, 1.0)
	c.observe(10, 1000.0)
	c.observe(11, 3000.0)  # server stalled 2 s
	assert_near(c.render_tick(3000.0), 10.0, 0.01)
