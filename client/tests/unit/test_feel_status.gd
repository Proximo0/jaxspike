extends TestCase
## StatSection: job buttons from a welcome, stat values and gates from a `you`.


func _welcome() -> Dictionary:
	return {"job_names": ["Novice", "Swordsman", "Mage", "Archer", "Acolyte", "Thief", "Merchant"],
		"stat_cost": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]}


func test_rebuild_jobs_count() -> void:
	var s := StatSection.new()
	s.set_welcome(_welcome())
	assert_eq(s.job_buttons.size(), 6)
	assert_eq(s.stat_labels.size(), 6)
	assert_eq(s.stat_buttons.size(), 6)
	assert_eq(s.job_buttons[0].text, "Swordsman")
	s.free()


func test_refresh_values_and_gates() -> void:
	var s := StatSection.new()
	s.set_welcome(_welcome())
	var you: Dictionary = {"stats": [5, 4, 3, 2, 1, 1], "stat_points": 3, "skill_points": 2, "job": 0, "job_level": 10}
	assert_true(UiStatsCalc.can_allot(you, _welcome(), 0))
	var store := EntityStore.new()
	store.you = you
	var sess := NetSession.new(FakeLink.new())
	sess.store = store
	s.bind(sess)
	s.refresh()
	assert_eq(s.stat_labels[0].text, "5")
	assert_eq(s.stat_labels[3].text, "2")
	assert_false(s.stat_buttons[0].disabled)
	assert_eq(s.stat_buttons[0].focus_mode, Control.FOCUS_NONE)
	assert_true(s.points_label.text.contains("3"))
	assert_false(s.job_buttons[0].disabled)
	store.you = {"stats": [5, 4, 3, 2, 1, 1], "stat_points": 0, "job": 0, "job_level": 3}
	s.refresh()
	assert_true(s.stat_buttons[0].disabled)
	assert_true(s.job_buttons[0].disabled)
	s.free()
