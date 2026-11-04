extends TestCase
## FxLogic with fake stores: cast loop, delayed hit, level-up, heal, cap of 64, and FxLayer posing.


func _ent(id: int, a: int, hp: int, h: int = -1) -> Dictionary:
	var d: Dictionary = {"id": id, "kind": 0, "name": "P", "pos": [1, 1], "hp": hp, "max_hp": 100, "a": a}
	if h >= 0:
		d["h"] = h
	return d


func _snap(s: EntityStore, tick: int, ents: Array, level: int = 5) -> void:
	s.apply_snap({"tick": tick, "ents": ents, "gone": [], "you": {"pos": [1, 1], "level": level}})


func test_cast_loops_while_action_8() -> void:
	var s := EntityStore.new()
	var fx := FxLogic.new()
	_snap(s, 1, [_ent(1, 0, 100)])
	fx.observe(s, 1, Callable())
	_snap(s, 2, [_ent(1, 8, 100)])
	assert_eq(fx.observe(s, 1, Callable()), 1)
	assert_eq(fx.live_of(FxLogic.CAST), 1)
	fx.advance(90.0 * 20.0)  # a loop never expires and wraps
	assert_eq(fx.live_of(FxLogic.CAST), 1)
	assert_true(fx.frame_of(0) >= 0 and fx.frame_of(0) < 8)
	_snap(s, 3, [_ent(1, 8, 100)])
	fx.observe(s, 1, Callable())
	assert_eq(fx.live_of(FxLogic.CAST), 1, "no second circle while casting")
	_snap(s, 4, [_ent(1, 0, 100)])
	fx.observe(s, 1, Callable())
	assert_eq(fx.live_of(FxLogic.CAST), 0)


func test_hit_spark_waits_for_swing_then_plays_once() -> void:
	var s := EntityStore.new()
	var fx := FxLogic.new()
	_snap(s, 1, [_ent(2, 0, 100)])
	fx.observe(s, 1, Callable())
	_snap(s, 2, [_ent(2, 0, 90, 10)])
	var delay := func(_id: int, _t: int) -> float: return 200.0
	assert_eq(fx.observe(s, 1, delay), 1)
	assert_eq(fx.frame_of(0), -1, "waiting for the atk frame")
	fx.advance(199.0)
	assert_eq(fx.frame_of(0), -1)
	fx.advance(2.0)
	assert_eq(fx.frame_of(0), 0)
	fx.advance(60.0 * 6.0)
	assert_eq(fx.live_of(FxLogic.HIT), 0, "one-shot freed after its last frame")


func test_miss_makes_no_spark_and_crit_does() -> void:
	var s := EntityStore.new()
	var fx := FxLogic.new()
	_snap(s, 1, [_ent(2, 0, 100)])
	fx.observe(s, 1, Callable())
	_snap(s, 2, [_ent(2, 0, 100, 0)])
	assert_eq(fx.observe(s, 1, Callable()), 0)
	s.apply_snap({"tick": 3, "ents": [{"id": 2, "kind": 0, "pos": [1, 1], "hp": 80, "max_hp": 100, "h": -20}], "gone": []})
	assert_eq(fx.observe(s, 1, Callable()), 1)


func test_level_up_only_on_increase() -> void:
	var s := EntityStore.new()
	var fx := FxLogic.new()
	_snap(s, 1, [_ent(1, 0, 100)], 5)
	fx.observe(s, 1, Callable())
	assert_eq(fx.live_count, 0, "first sight is a baseline")
	_snap(s, 2, [_ent(1, 0, 100)], 5)
	fx.observe(s, 1, Callable())
	assert_eq(fx.live_count, 0)
	_snap(s, 3, [_ent(1, 0, 100)], 6)
	fx.observe(s, 1, Callable())
	assert_eq(fx.live_of(FxLogic.LEVEL_UP), 1)
	assert_eq(fx.targets[0], 1)


func test_heal_needs_a_ten_percent_jump() -> void:
	var s := EntityStore.new()
	var fx := FxLogic.new()
	_snap(s, 1, [_ent(3, 0, 50)])
	fx.observe(s, 1, Callable())
	_snap(s, 2, [_ent(3, 0, 52)])  # regen
	fx.observe(s, 1, Callable())
	assert_eq(fx.live_count, 0)
	_snap(s, 3, [_ent(3, 0, 70)])  # +18
	fx.observe(s, 1, Callable())
	assert_eq(fx.live_of(FxLogic.HEAL), 1)
	_snap(s, 4, [_ent(3, 0, 40)])  # damage
	fx.observe(s, 1, Callable())
	assert_eq(fx.live_of(FxLogic.HEAL), 1)


func test_cap_is_64_live_effects() -> void:
	var s := EntityStore.new()
	var fx := FxLogic.new()
	var first: Array = []
	var second: Array = []
	for i: int in 80:
		first.append(_ent(10 + i, 0, 100))
		second.append(_ent(10 + i, 8, 100))
	_snap(s, 1, first)
	fx.observe(s, 1, Callable())
	_snap(s, 2, second)
	fx.observe(s, 1, Callable())
	assert_eq(fx.live_count, 64)
	assert_eq(fx.dropped, 16)


func test_removed_caster_stops_its_circle() -> void:
	var s := EntityStore.new()
	var fx := FxLogic.new()
	_snap(s, 1, [_ent(1, 0, 100)])
	fx.observe(s, 1, Callable())
	_snap(s, 2, [_ent(1, 8, 100)])
	fx.observe(s, 1, Callable())
	s.apply_snap({"tick": 3, "ents": [], "gone": [1]})
	fx.observe(s, 1, Callable())
	assert_eq(fx.live_count, 0)


func test_layer_poses_sprites_from_entity_layer() -> void:
	var s := EntityStore.new()
	_snap(s, 1, [_ent(1, 0, 100)])
	var el := EntityLayer.new()
	el.auto_refresh = false
	el.store = s
	el.own_id = 1
	el.refresh(0.0)
	var layer := FxLayer.new()
	layer.auto_refresh = false
	layer.bind_store(s, el)
	layer.refresh(16.0)
	_snap(s, 2, [_ent(1, 8, 100)])
	el.refresh(100.0)
	layer.refresh(16.0)
	assert_eq(layer.logic.live_of(FxLogic.CAST), 1)
	layer.free()
	el.free()
