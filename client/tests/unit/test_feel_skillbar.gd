extends TestCase
## Skill bar grey-out truth table and the cast bar's seconds and between-tick glide. The SkillBar/CastBar nodes are
## covered by sim_ui_stats; these test the pure functions behind them.


func test_can_act_truth_table() -> void:
	assert_true(UiStatsCalc.can_act({}))
	assert_true(UiStatsCalc.can_act({"after_delay": 0, "casting": 0}))
	assert_false(UiStatsCalc.can_act({"after_delay": 3, "casting": 0}))
	assert_true(UiStatsCalc.can_act({"after_delay": 0, "casting": 2}), "casting alone does not grey the bar")
	assert_false(UiStatsCalc.can_act({"after_delay": 3, "casting": 2}))


func test_after_delay_fraction() -> void:
	assert_near(UiStatsCalc.after_delay_fraction({"after_delay": 5}, 10), 0.5)
	assert_near(UiStatsCalc.after_delay_fraction({"after_delay": 0}, 10), 0.0)
	assert_near(UiStatsCalc.after_delay_fraction({"after_delay": 20}, 10), 1.0)
	assert_near(UiStatsCalc.after_delay_fraction({"after_delay": 5}, 0), 0.0)


func test_seconds_left() -> void:
	var t := UiCastTrack.new()
	t.update(4, 4)
	assert_near(t.seconds_left(100), 0.4)
	assert_near(t.seconds_left(50), 0.2)
	t.update(0, 0)
	assert_near(t.seconds_left(100), 0.0)


func test_cast_glides_between_ticks() -> void:
	var t := UiCastTrack.new()
	t.update(4, 10, 1000.0)
	assert_near(t.fraction(), 0.0)
	t.update(4, 10, 1050.0)
	assert_near(t.seconds_left(100), 0.95, 0.0001, "half a tick elapsed")
	assert_near(t.fraction(), 0.05)
	t.update(4, 10, 1500.0)
	assert_near(t.seconds_left(100), 0.9, 0.0001, "capped at one tick")
	t.update(4, 9, 1500.0)
	assert_near(t.fraction(), 0.1, 0.0001, "new tick restarts the glide")
	assert_near(t.seconds_left(100), 0.9)
