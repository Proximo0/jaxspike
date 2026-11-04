extends TestCase
## EntityLayer animation behaviour: dead fade on the sprite, configure skipped when nothing changed.


func _player(a: int, t: int, m: int, hp: int) -> Dictionary:
	return {"id": 900, "kind": 0, "name": "Zed", "job": 0, "pos": [2, 2], "hp": hp, "max_hp": 10,
			"a": a, "d": 0, "t": t, "m": m, "s": 150}


func test_dead_row_holds_then_fades_out() -> void:
	var s := EntityStore.new()
	s.apply_snap({"tick": 1, "ents": [_player(ClipSelect.DEAD, 1, 0, 0)], "gone": []})
	var layer := EntityLayer.new()
	layer.auto_refresh = false
	layer.store = s
	layer.refresh(0.0)
	var t0: float = layer.last_render_ms
	var a_early: float = layer.sprite_of(900).modulate.a
	assert_near(a_early, 1.0, 0.0001, "alpha at start")
	# jump the wall clock well past clip + hold + fade
	s.apply_snap({"tick": 60, "ents": [], "gone": []})
	layer.refresh(6000.0)
	assert_true(layer.last_render_ms > t0)
	assert_near(layer.sprite_of(900).modulate.a, 0.0, 0.0001, "faded out after 3 s")
	layer.free()


func test_configure_skipped_when_row_unchanged() -> void:
	var s := EntityStore.new()
	s.apply_snap({"tick": 1, "ents": [_player(ClipSelect.IDLE, 0, 0, 10)], "gone": []})
	var layer := EntityLayer.new()
	layer.auto_refresh = false
	layer.store = s
	layer.refresh(0.0)
	var sp: EntitySprite = layer.sprite_of(900)
	assert_eq(sp.hp_frac, 1.0)
	s.apply_snap({"tick": 2, "ents": [_player(ClipSelect.IDLE, 0, 0, 5)], "gone": []})
	layer.refresh(100.0)
	assert_near(sp.hp_frac, 0.5, 0.0001, "hp change reconfigured")
	layer.free()
