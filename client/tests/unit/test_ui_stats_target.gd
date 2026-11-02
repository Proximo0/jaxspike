extends TestCase


func test_pick_nearest_within_radius() -> void:
	var t := TargetSelect.new()
	var pos: Dictionary = {1: Vector2(10, 0), 2: Vector2(4, 0), 3: Vector2(100, 0)}
	assert_eq(t.pick(Vector2.ZERO, pos, 20.0), 2)
	assert_eq(t.selected_id, 2)
	assert_eq(t.pick(Vector2(300, 0), pos, 20.0), -1)
	assert_eq(t.selected_id, -1)


func test_pick_skips_and_tie_goes_low() -> void:
	var t := TargetSelect.new()
	var pos: Dictionary = {5: Vector2(3, 0), 4: Vector2(-3, 0), 0: Vector2(0, 0)}
	assert_eq(t.pick(Vector2.ZERO, pos, 10.0, [0]), 4)


func test_resolve_target_types() -> void:
	var t := TargetSelect.new()
	assert_eq(t.resolve(UiStatsCalc.TARGET_SELF, 7), -1)
	assert_eq(t.resolve(UiStatsCalc.TARGET_ALLY, 7), 7)
	assert_eq(t.resolve(UiStatsCalc.TARGET_ENEMY, 7), TargetSelect.NEEDS_TARGET)
	t.select(600)
	assert_eq(t.resolve(UiStatsCalc.TARGET_ENEMY, 7), 600)
	assert_eq(t.resolve(UiStatsCalc.TARGET_ALLY, 7), 600)
	assert_eq(t.resolve(UiStatsCalc.TARGET_SELF, 7), -1)
	assert_eq(t.resolve(UiStatsCalc.TARGET_GROUND, 7), 600)


func test_prune_clears_missing() -> void:
	var t := TargetSelect.new()
	t.select(9)
	t.prune(func(id: int) -> bool: return id == 9)
	assert_eq(t.selected_id, 9)
	t.prune(func(_id: int) -> bool: return false)
	assert_eq(t.selected_id, -1)
