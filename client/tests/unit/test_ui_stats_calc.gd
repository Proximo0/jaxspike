extends TestCase


func _defs() -> Dictionary:
	var w: Dictionary = {"skills": {"list": [null,
		{"id": 1, "name": "Sword Mastery", "job": 1, "max_level": 10, "effect": 0, "target": 0, "levels": [{"sp": 0}]},
		{"id": 4, "name": "Bash", "job": 1, "max_level": 10, "effect": 1, "target": 2,
			"levels": [{"sp": 8, "cooldown": 4, "cast": 0}, {"sp": 10, "cooldown": 5, "cast": 0}]},
		{"id": 9, "name": "Bolt", "job": 2, "max_level": 3, "effect": 2, "target": 2, "levels": [{"sp": 12}]},
		{"id": 2, "name": "Basic", "job": 0, "max_level": 3, "effect": 4, "target": 0, "levels": [{"sp": 1}]},
	]}}
	return UiStatsCalc.skill_index(w)


func test_exp_fraction_and_text() -> void:
	assert_near(UiStatsCalc.exp_fraction(50, 200), 0.25)
	assert_near(UiStatsCalc.exp_fraction(500, 200), 1.0)
	assert_near(UiStatsCalc.exp_fraction(5, 0), 1.0)
	assert_eq(UiStatsCalc.exp_text(50, 200), "50 / 200")
	assert_eq(UiStatsCalc.exp_text(50, 0), "max")


func test_can_allot() -> void:
	var w: Dictionary = {"stat_cost": [0, 1, 1, 2, 2]}
	var you: Dictionary = {"stat_points": 1, "stats": [1, 3, 4, 99, 1, 1]}
	assert_true(UiStatsCalc.can_allot(you, w, 0))
	assert_false(UiStatsCalc.can_allot(you, w, 1), "cost 2 > 1 point")
	assert_false(UiStatsCalc.can_allot(you, w, 3), "capped")
	assert_false(UiStatsCalc.can_allot({"stat_points": 0, "stats": [1]}, w, 0))
	assert_true(UiStatsCalc.can_allot({"stat_points": 1, "stats": [5]}, {}, 0), "no table -> cost 1")


func test_job_change_rules() -> void:
	assert_true(UiStatsCalc.can_change_job({"job": 0, "job_level": 10}))
	assert_false(UiStatsCalc.can_change_job({"job": 0, "job_level": 9}))
	assert_false(UiStatsCalc.can_change_job({"job": 2, "job_level": 40}))
	var c: Array[Dictionary] = UiStatsCalc.job_choices({"job_names": ["Novice", "Swordsman", "Mage"]})
	assert_eq(c.size(), 2)
	assert_eq(c[0]["job"], 1)
	assert_eq(c[1]["name"], "Mage")


func test_cooldown_fraction() -> void:
	assert_near(UiStatsCalc.cooldown_fraction(2, 4), 0.5)
	assert_near(UiStatsCalc.cooldown_fraction(0, 4), 0.0)
	assert_near(UiStatsCalc.cooldown_fraction(9, 4), 1.0)
	assert_near(UiStatsCalc.cooldown_fraction(3, 0), 0.0)


func test_bar_skills_learned_and_castable() -> void:
	var you: Dictionary = {"learned": [[1, 5], [4, 2], [2, 1], [9, 0]]}
	var bar: Array[Dictionary] = UiStatsCalc.bar_skills(_defs(), you)
	assert_eq(bar.size(), 2, "passive 1 and unlearned 9 excluded")
	assert_eq(bar[0]["id"], 2)
	assert_eq(bar[1]["id"], 4)
	assert_eq(bar[1]["sp"], 10, "level 2 entry")
	assert_eq(bar[1]["cooldown"], 5)
	assert_true(UiStatsCalc.can_afford({"sp": 10}, 10))
	assert_false(UiStatsCalc.can_afford({"sp": 9}, 10))


func test_bar_capped_at_nine() -> void:
	var list: Array = []
	var learned: Array = []
	for i: int in range(1, 13):
		list.append({"id": i, "name": "S%d" % i, "job": 0, "max_level": 1, "effect": 1, "levels": [{"sp": 1}]})
		learned.append([i, 1])
	var bar: Array[Dictionary] = UiStatsCalc.bar_skills(UiStatsCalc.skill_index({"skills": {"list": list}}), {"learned": learned})
	assert_eq(bar.size(), 9)


func test_slot_for_key() -> void:
	assert_eq(UiStatsCalc.slot_for_key(KEY_1), 0)
	assert_eq(UiStatsCalc.slot_for_key(KEY_9), 8)
	assert_eq(UiStatsCalc.slot_for_key(KEY_0), -1)


func test_panel_skills_job_filter_and_learn() -> void:
	var p: Array[Dictionary] = UiStatsCalc.panel_skills(_defs(), {"job": 1, "skill_points": 2, "learned": [[4, 10]]})
	assert_eq(p.size(), 3, "novice + swordsman only")
	assert_eq(p[0]["id"], 2, "novice first")
	assert_true(p[0]["can_learn"])
	assert_false(p[2]["can_learn"], "Bash at max")
	assert_eq(UiStatsCalc.panel_skills(_defs(), {"job": 0}).size(), 1)
	assert_false(UiStatsCalc.can_learn({"skill_points": 0}, 0, 5))
	assert_true(UiStatsCalc.can_learn({}, 0, 5), "no stats block: free")


func test_buff_rows() -> void:
	var w: Dictionary = {"buffs": ["", "Blessing", "Increase AGI"]}
	var r: Array[Dictionary] = UiStatsCalc.buff_rows(w, {"buffs": [[2, 50, 3], [1, 0, 1], [7, 10, 1]]}, 120)
	assert_eq(r.size(), 2)
	assert_eq(r[0]["name"], "Increase AGI")
	assert_near(float(r[0]["seconds"]), 6.0)
	assert_eq(r[0]["level"], 3)
	assert_eq(r[1]["name"], "Buff 7")


func test_cast_track_fills() -> void:
	var t := UiCastTrack.new()
	t.update(4, 10)
	assert_near(t.fraction(), 0.0)
	t.update(4, 5)
	assert_near(t.fraction(), 0.5)
	assert_true(t.active())
	t.update(0, 0)
	assert_false(t.active())
	t.update(4, 3)
	assert_near(t.fraction(), 0.0, 0.0001, "new cast restarts total")
