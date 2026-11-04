extends TestCase
## Hud.info_rows and its formatters: the RO basic info window strings.


func test_info_rows() -> void:
	var you: Dictionary = {"level": 12, "job_level": 7, "job": 1, "hp": 150, "max_hp": 200, "sp": 6, "max_sp": 30,
		"base_exp": 50, "base_exp_next": 200, "job_exp": 0, "job_exp_next": 0, "ryn": 1234567}
	var w: Dictionary = {"job_names": ["Novice", "Swordsman"]}
	var r: Dictionary = Hud.info_rows(you, {"name": "Odin"}, w)
	assert_eq(r["name"], "Odin")
	assert_eq(r["job"], "Swordsman")
	assert_eq(r["levels"], "Base Lv. 12   Job Lv. 7")
	assert_eq(r["hp_text"], "150 / 200")
	assert_near(r["hp"], 0.75)
	assert_eq(r["sp_text"], "6 / 30")
	assert_near(r["sp"], 0.2)
	assert_eq(r["base_text"], "25.0 %")
	assert_near(r["base"], 0.25)
	assert_eq(r["job_text"], "MAX")
	assert_eq(r["ryn"], "Ryn 1,234,567")
	assert_eq(r["weight"], "Weight - / -")
	assert_eq(Hud.info_rows({"name": "Me"}, {"name": "Odin"}, {})["name"], "Me", "you.name wins")
	assert_eq(Hud.info_rows({}, {}, {})["name"], "-")


func test_formatters() -> void:
	assert_eq(Hud.ryn_text(0), "0")
	assert_eq(Hud.ryn_text(999), "999")
	assert_eq(Hud.ryn_text(1000), "1,000")
	assert_eq(Hud.ryn_text(-12345), "-12,345")
	assert_eq(Hud.exp_percent(1, 3), "33.3 %")
	assert_eq(Hud.exp_percent(500, 100), "100.0 %")
	assert_eq(Hud.weight_text({"weight": 120, "max_weight": 2000}), "Weight 120 / 2000")
