extends TestCase


func test_defaults_match_fields() -> void:
	var t := Tuning.new()
	for f: Array in Tuning.FIELDS:
		assert_near(float(t.get(f[0])), float(f[1]), 0.0001, str(f[0]))


func test_set_value_clamps() -> void:
	var t := Tuning.new()
	assert_true(t.set_value("gravity", 1000.0))
	assert_near(t.gravity, 40.0)


func test_unknown_field_rejected() -> void:
	assert_false(Tuning.new().set_value("nope", 1.0))


func test_diff_and_dict_round_trip() -> void:
	var t := Tuning.new()
	t.set_value("move_speed", 10.0)
	assert_eq(t.diff_from_defaults().keys(), ["move_speed"])
	var u := Tuning.new()
	assert_eq(u.apply_dict(t.to_dict()), Tuning.FIELDS.size())
	assert_near(u.move_speed, 10.0)
