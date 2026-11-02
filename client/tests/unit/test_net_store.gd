extends TestCase


func _snap(tick: int, ents: Array, gone: Array, full: bool) -> Dictionary:
	return {"v": 2, "t": "snap", "tick": tick, "you": {"pos": [3.0, 4.0], "hp": 10.0, "max_hp": 20.0, "kills": 1.0},
		"ents": ents, "gone": gone, "notice": "hi", "full": full}


func _ent(id: int, x: int, y: int) -> Dictionary:
	return {"id": id, "kind": 1, "pos": [x, y], "hp": 5, "max_hp": 5, "mkind": 0}


func test_upsert_moves_and_keeps_prev_pos() -> void:
	var s := EntityStore.new()
	s.apply_snap(_snap(1, [_ent(600, 1, 1)], [], true))
	s.apply_snap(_snap(2, [_ent(600, 2, 1)], [], false))
	var e: Dictionary = s.get_ent(600)
	assert_eq(e["pos"], Vector2i(2, 1))
	assert_eq(e["prev_pos"], Vector2i(1, 1))
	assert_eq(s.tick, 2)
	assert_eq(s.you["pos"], Vector2i(3, 4))
	assert_eq(s.you["hp"], 10)
	assert_eq(s.notice, "hi")


func test_gone_deletes() -> void:
	var s := EntityStore.new()
	s.apply_snap(_snap(1, [_ent(600, 1, 1), _ent(5, 2, 2)], [], true))
	assert_eq(s.ids(), PackedInt32Array([5, 600]))
	s.apply_snap(_snap(2, [], [600, 999], false))
	assert_eq(s.ids(), PackedInt32Array([5]))
	assert_true(s.get_ent(600).is_empty())


func test_full_wipes_previous() -> void:
	var s := EntityStore.new()
	s.apply_snap(_snap(1, [_ent(600, 1, 1)], [], true))
	s.apply_snap(_snap(2, [_ent(7, 1, 1)], [], true))
	assert_eq(s.ids(), PackedInt32Array([7]))


func test_snapshot_restore_equality() -> void:
	var s := EntityStore.new()
	s.apply_snap(_snap(1, [{"id": 0, "kind": 0, "pos": [1, 2], "hp": 1, "max_hp": 2, "name": "ann"}], [], true))
	var snap: Dictionary = s.snapshot()
	s.apply_snap(_snap(9, [_ent(600, 9, 9)], [0], false))
	assert_true(s.snapshot() != snap)
	s.restore(snap)
	assert_eq(s.snapshot(), snap)
	assert_eq(s.get_ent(0)["name"], "ann")
	var t := EntityStore.new()
	t.restore(snap)
	assert_eq(t.snapshot(), snap)
