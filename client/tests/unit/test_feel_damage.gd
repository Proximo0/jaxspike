extends TestCase
## DamageStyle: kinds, colours, text, sizes, rise/alpha curves, drift.


func test_kind_of() -> void:
	assert_eq(DamageStyle.kind_of(50, false, false), DamageStyle.NORMAL)
	assert_eq(DamageStyle.kind_of(50, false, true), DamageStyle.CRIT)
	assert_eq(DamageStyle.kind_of(50, true, false), DamageStyle.OWN)
	assert_eq(DamageStyle.kind_of(50, true, true), DamageStyle.OWN, "own wins over crit")
	assert_eq(DamageStyle.kind_of(0, false, false), DamageStyle.MISS)
	assert_eq(DamageStyle.kind_of(0, true, true), DamageStyle.MISS)


func test_colors_text_sizes() -> void:
	assert_eq(DamageStyle.color_of(DamageStyle.NORMAL), Color(1, 1, 1))
	assert_eq(DamageStyle.color_of(DamageStyle.CRIT), Color(1.0, 0.9, 0.2))
	assert_eq(DamageStyle.color_of(DamageStyle.OWN), Color(1.0, 0.35, 0.3))
	assert_true(DamageStyle.color_of(DamageStyle.MISS).r < 1.0, "miss is grey")
	assert_eq(DamageStyle.text_of(0, DamageStyle.MISS), "Miss")
	assert_eq(DamageStyle.text_of(123, DamageStyle.NORMAL), "123")
	assert_eq(DamageStyle.text_of(-77, DamageStyle.CRIT), "77")
	assert_near(DamageStyle.font_px(DamageStyle.NORMAL), 20.0, 0.001)
	assert_near(DamageStyle.font_px(DamageStyle.OWN), 20.0, 0.001)
	assert_near(DamageStyle.font_px(DamageStyle.MISS), 20.0, 0.001)
	assert_near(DamageStyle.font_px(DamageStyle.CRIT), 26.0, 0.001)
	assert_eq(DamageStyle.outline_px(), 5)


func test_rise() -> void:
	assert_near(DamageStyle.rise_tiles(0.0, 900.0), 0.0, 0.0001)
	assert_near(DamageStyle.rise_tiles(900.0, 900.0), 1.2, 0.0001)
	assert_near(DamageStyle.rise_tiles(450.0, 900.0), 1.2 * 0.75, 0.0001, "matches DamagePops curve")
	var prev: float = -1.0
	for i: int in 10:
		var r: float = DamageStyle.rise_tiles(float(i) * 100.0, 900.0)
		assert_true(r > prev, "monotonic")
		prev = r


func test_alpha() -> void:
	assert_near(DamageStyle.alpha(0.0, 900.0, 300.0), 1.0, 0.0001)
	assert_near(DamageStyle.alpha(600.0, 900.0, 300.0), 1.0, 0.0001, "full until the fade starts")
	assert_near(DamageStyle.alpha(750.0, 900.0, 300.0), 0.5, 0.0001)
	assert_near(DamageStyle.alpha(900.0, 900.0, 300.0), 0.0, 0.0001)
	assert_near(DamageStyle.alpha(1000.0, 900.0, 300.0), 0.0, 0.0001)


func test_drift() -> void:
	assert_near(DamageStyle.drift_px(7, DamageStyle.MISS), 0.0, 0.0001)
	var seen: Dictionary = {}
	for s: int in 20:
		var d: float = DamageStyle.drift_px(s, DamageStyle.NORMAL)
		assert_true(absf(d) <= 6.0 + 0.0001, "within 6 px")
		assert_near(d, DamageStyle.drift_px(s, DamageStyle.CRIT), 0.0001, "deterministic")
		seen[d] = true
	assert_true(seen.size() >= 4, "offsets vary")
	assert_true(DamageStyle.drift_px(1, DamageStyle.NORMAL) != DamageStyle.drift_px(2, DamageStyle.NORMAL), "neighbours differ")
