extends TestCase

const SAVED_PATH: String = "res://game/ui/theme/ro_theme.tres"


func test_default_font_size() -> void:
	var t: Theme = RoThemeBuilder.build()
	assert_eq(t.default_font_size, 12)
	assert_eq(t.get_font_size("font_size", "Button"), 12)


func test_button_normal_is_flat_with_palette() -> void:
	var t: Theme = RoThemeBuilder.build()
	var s := t.get_stylebox("normal", "Button") as StyleBoxFlat
	assert_true(s != null, "Button normal is StyleBoxFlat")
	if s != null:
		assert_eq(s.bg_color, Color(0.28, 0.33, 0.45))
		assert_eq(s.border_width_left, 1)


func test_bar_variations_fill_colours() -> void:
	var t: Theme = RoThemeBuilder.build()
	var expected: Dictionary = {"HpBar": RoThemeBuilder.HP, "SpBar": RoThemeBuilder.SP, "ExpBar": RoThemeBuilder.EXP}
	for v: String in expected:
		var s := t.get_stylebox("fill", v) as StyleBoxFlat
		assert_true(s != null, v + " fill")
		if s != null:
			assert_eq(s.bg_color, expected[v] as Color, v)
		assert_eq(t.get_type_variation_base(v), &"ProgressBar")
	assert_eq(t.get_type_variation_base("TitleBar"), &"Panel")


func test_saved_theme_loads() -> void:
	if not FileAccess.file_exists(SAVED_PATH):
		assert_true(true)
		return
	var t := load(SAVED_PATH) as Theme
	assert_true(t != null, "ro_theme.tres loads as Theme")
	if t != null:
		assert_eq(t.default_font_size, 12)
