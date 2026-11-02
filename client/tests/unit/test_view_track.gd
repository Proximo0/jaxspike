extends TestCase


func test_first_target_places_at_once() -> void:
	var t := EntityTrack.new()
	t.set_target(Vector2(5, 5), 1000.0, 120.0)
	assert_eq(t.sample(1000.0), Vector2(5, 5))


func test_interpolates_over_duration() -> void:
	var t := EntityTrack.new()
	t.set_target(Vector2(5, 5), 0.0, 120.0)
	t.set_target(Vector2(6, 5), 1000.0, 120.0)
	assert_eq(t.sample(1000.0), Vector2(5, 5))
	assert_near(t.sample(1060.0).x, 5.5)
	assert_eq(t.sample(1120.0), Vector2(6, 5))
	assert_eq(t.sample(5000.0), Vector2(6, 5))
	assert_true(t.settled(1120.0))
	assert_false(t.settled(1050.0))


func test_retarget_starts_from_drawn_position() -> void:
	var t := EntityTrack.new()
	t.set_target(Vector2(0, 0), 0.0, 100.0)
	t.set_target(Vector2(1, 0), 0.0, 100.0)
	t.set_target(Vector2(2, 0), 50.0, 100.0)
	assert_near(t.sample(50.0).x, 0.5)
	assert_near(t.sample(150.0).x, 2.0)


func test_same_target_keeps_motion() -> void:
	var t := EntityTrack.new()
	t.set_target(Vector2(0, 0), 0.0, 100.0)
	t.set_target(Vector2(1, 0), 0.0, 100.0)
	t.set_target(Vector2(1, 0), 50.0, 100.0)
	assert_near(t.sample(50.0).x, 0.5)


func test_zero_delay_and_teleport_snap() -> void:
	var t := EntityTrack.new()
	t.set_target(Vector2(0, 0), 0.0, 0.0)
	t.set_target(Vector2(1, 0), 10.0, 0.0)
	assert_eq(t.sample(10.0), Vector2(1, 0))
	t.set_target(Vector2(50, 50), 20.0, 120.0)
	assert_eq(t.sample(20.0), Vector2(50, 50))


func _layer_store(n: int) -> EntityStore:
	var s := EntityStore.new()
	var ents: Array = []
	for i: int in range(n):
		ents.append({"id": 600 + i, "kind": 1, "pos": [i, 0], "hp": 5, "max_hp": 5, "mkind": 0})
	s.apply_snap({"tick": 1, "full": true, "ents": ents, "gone": []})
	return s


func test_layer_updates_only_changed_ids() -> void:
	var s: EntityStore = _layer_store(50)
	var layer := EntityLayer.new()
	layer.store = s
	layer.refresh(0.0)
	assert_eq(layer.active_count(), 50)
	assert_eq(layer.configured_count, 50)
	layer.refresh(16.0)
	assert_eq(layer.configured_count, 0)
	s.apply_snap({"tick": 2, "ents": [{"id": 603, "kind": 1, "pos": [3, 5], "hp": 5, "max_hp": 5}], "gone": []})
	layer.refresh(32.0)
	assert_eq(layer.configured_count, 1)
	assert_eq(layer.active_count(), 50)
	layer.free()


func test_layer_interpolates_unchanged_nodes_and_removes() -> void:
	var s: EntityStore = _layer_store(3)
	var layer := EntityLayer.new()
	layer.store = s
	layer.refresh(0.0)
	s.apply_snap({"tick": 2, "ents": [{"id": 600, "kind": 1, "pos": [0, 2], "hp": 5, "max_hp": 5}], "gone": [601]})
	layer.refresh(10.0)
	var y0: float = layer.sprite_of(600).position.y
	layer.refresh(60.0)
	assert_true(layer.sprite_of(600).position.y > y0, "moving node keeps interpolating with no change this frame")
	assert_false(layer.has_sprite(601))
	assert_eq(layer.active_count(), 2)
	layer.free()


func test_layer_resyncs_after_missed_snap() -> void:
	var s: EntityStore = _layer_store(2)
	var layer := EntityLayer.new()
	layer.store = s
	layer.refresh(0.0)
	s.apply_snap({"tick": 2, "ents": [{"id": 700, "kind": 1, "pos": [9, 9]}], "gone": []})
	s.apply_snap({"tick": 3, "ents": [], "gone": [600]})
	layer.refresh(10.0)
	assert_true(layer.has_sprite(700))
	assert_false(layer.has_sprite(600))
	assert_true(layer.has_sprite(601))
	layer.free()
