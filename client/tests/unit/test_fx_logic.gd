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


func _fx_snap(s: EntityStore, tick: int, ents: Array, fx: Array) -> void:
	s.apply_snap({"tick": tick, "ents": ents, "gone": [], "you": {"pos": [1, 1], "level": 5}, "fx": fx})


func test_skill_kind_maps_every_family() -> void:
	var fx := FxLogic.new()
	assert_eq(fx.skill_kind(FxLogic.SK_BASH), FxLogic.HIT)
	assert_eq(fx.skill_kind(FxLogic.SK_MAMMONITE), FxLogic.HIT)
	for sid: int in FxLogic.BOLT_SKILLS:
		assert_eq(fx.skill_kind(sid), FxLogic.BOLT)
	assert_eq(fx.skill_kind(FxLogic.SK_ARROW_SHOWER), FxLogic.ARROW_SHOWER)
	assert_eq(fx.skill_kind(FxLogic.SK_DOUBLE_STRAFE), FxLogic.ARROW_SHOWER)
	assert_eq(fx.skill_kind(FxLogic.SK_MAGNUM_BREAK), FxLogic.MAGNUM_BREAK)
	assert_eq(fx.skill_kind(FxLogic.SK_HEAL), FxLogic.HEAL)
	assert_eq(fx.skill_kind(FxLogic.SK_FIRST_AID), FxLogic.HEAL)
	assert_eq(fx.skill_kind(21), FxLogic.CAST_BURST, "Blessing has no sprite of its own")
	fx.set_skills([{"id": 33, "effect": FxLogic.EFFECT_ATK, "element": 1}, {"id": 21, "effect": 4, "element": 0}])
	assert_eq(fx.skill_kind(33), FxLogic.HIT, "an ATK skill from welcome.skills sparks")
	assert_eq(fx.skill_kind(21), FxLogic.CAST_BURST)


func test_bolt_tint_follows_the_element() -> void:
	var fx := FxLogic.new()
	assert_eq(FxLogic.element_tint(fx.skill_element(FxLogic.SK_FIRE_BOLT)), Color(1.0, 0.55, 0.2))
	assert_eq(FxLogic.element_tint(fx.skill_element(FxLogic.SK_COLD_BOLT)), Color(0.45, 0.72, 1.0))
	assert_eq(FxLogic.element_tint(fx.skill_element(FxLogic.SK_LIGHTNING_BOLT)), Color(1.0, 0.95, 0.35))
	fx.set_skills([{"id": FxLogic.SK_FIRE_BOLT, "effect": 2, "element": FxLogic.ELEM_WATER}])
	assert_eq(fx.skill_element(FxLogic.SK_FIRE_BOLT), FxLogic.ELEM_WATER, "welcome.skills wins")


func test_fx_events_start_tile_effects_once() -> void:
	var s := EntityStore.new()
	var fx := FxLogic.new()
	_snap(s, 1, [_ent(1, 0, 100)])
	fx.observe(s, 1, Callable())
	_fx_snap(s, 2, [], [[1, FxLogic.SK_COLD_BOLT, 4, 5], [1, FxLogic.SK_MAGNUM_BREAK, 1, 1], [2, 21, 3, 3]])
	assert_eq(fx.observe(s, 1, Callable()), 3)
	assert_eq(fx.live_of(FxLogic.BOLT), 1)
	assert_eq(fx.live_of(FxLogic.MAGNUM_BREAK), 1)
	assert_eq(fx.live_of(FxLogic.CAST_BURST), 1)
	assert_eq(fx.targets[0], -1)
	assert_eq(fx.tiles[0], Vector2(4, 5))
	assert_eq(fx.tints[0], Color(0.45, 0.72, 1.0))
	assert_eq(fx.tints[1], Color.WHITE)
	assert_eq(fx.observe(s, 1, Callable()), 0, "same snap: nothing new")
	fx.advance(60.0 * 8.0 + 1.0)
	assert_eq(fx.live_count, 0, "all one-shots, the burst included")
	_snap(s, 3, [])
	assert_eq(fx.observe(s, 1, Callable()), 0, "events last one snap")


func test_bash_spark_not_doubled_where_the_hit_already_sparked() -> void:
	var s := EntityStore.new()
	var fx := FxLogic.new()
	_snap(s, 1, [_ent(2, 0, 100)])
	fx.observe(s, 1, Callable())
	_fx_snap(s, 2, [_ent(2, 0, 80, 20)], [[1, FxLogic.SK_BASH, 1, 1], [1, FxLogic.SK_BASH, 6, 6]])
	assert_eq(fx.observe(s, 1, Callable()), 2, "the entity spark plus the spark on the empty tile")
	assert_eq(fx.live_of(FxLogic.HIT), 2)


func test_layer_draws_a_tile_effect_tinted_at_the_tile_centre() -> void:
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
	_fx_snap(s, 2, [_ent(1, 0, 100)], [[1, FxLogic.SK_FIRE_BOLT, 3, 4]])
	el.refresh(100.0)
	layer.refresh(1.0)
	assert_eq(layer.logic.live_of(FxLogic.BOLT), 1)
	assert_eq(layer.logic.tints[0], Color(1.0, 0.55, 0.2))
	layer.free()
	el.free()


func test_layer_set_welcome_reads_the_skill_elements() -> void:
	var layer := FxLayer.new()
	layer.set_welcome({"t": "welcome", "skills": {"list": [{"id": FxLogic.SK_FIRE_BOLT, "effect": 2, "element": 4}]}})
	assert_eq(layer.logic.skill_element(FxLogic.SK_FIRE_BOLT), 4)
	layer.set_welcome({"t": "welcome"})
	assert_eq(layer.logic.skill_element(FxLogic.SK_FIRE_BOLT), 4, "a welcome without skills keeps the table")
	layer.free()
