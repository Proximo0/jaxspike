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


func test_feed_uses_speed_and_anim_fields() -> void:
	var t := EntityTrack.new()
	t.feed({"pos": [0, 0], "a": 0, "d": 0, "t": 0, "m": 0, "s": 150}, 0, 100.0, -100.0, 100.0)
	t.feed({"pos": [1, 0], "a": 1, "d": 6, "t": 2, "m": 0, "s": 150}, 2, 100.0, 100.0, 100.0)
	assert_near(t.pos_at(175.0).x, 0.5, 0.001)
	var c: Dictionary = t.clip_at(175.0, 100.0, 100.0)
	assert_eq(int(c["action"]), ClipSelect.WALK)
	assert_eq(int(c["face"]), 6)
	assert_near(float(c["speed_ms"]), 150.0)
	assert_eq(int(t.clip_at(400.0, 100.0, 100.0)["action"]), ClipSelect.IDLE, "walk with no step left shows idle")


func test_feed_without_anim_fields_walks_from_motion() -> void:
	var t := EntityTrack.new()
	t.feed({"pos": [0, 0]}, 1, 120.0, 0.0, 120.0)
	t.feed({"pos": [0, -1]}, 2, 120.0, 120.0, 120.0)
	var c: Dictionary = t.clip_at(150.0, 120.0, 120.0)
	assert_eq(int(c["action"]), ClipSelect.WALK)
	assert_eq(int(c["face"]), 4)
	assert_eq(int(t.clip_at(400.0, 120.0, 120.0)["action"]), ClipSelect.IDLE)


func test_feed_reports_hit_once_per_tick() -> void:
	var t := EntityTrack.new()
	assert_eq(t.feed({"pos": [0, 0], "h": 12}, 5, 100.0, 0.0, 100.0), 12)
	assert_eq(t.feed({"pos": [0, 0], "h": 12}, 5, 100.0, 0.0, 100.0), 0)
	assert_eq(t.feed({"pos": [0, 0]}, 6, 100.0, 0.0, 100.0), 0)
	assert_eq(t.feed({"pos": [0, 0], "h": 7}, 7, 100.0, 0.0, 100.0), 7)


func _anim_store(ents: Dictionary, tick: int) -> EntityStore:
	var s := EntityStore.new()
	s.restore({"tick": tick, "snaps": 1, "ents": ents})
	return s


func test_layer_pops_damage_at_attackers_atk_frame() -> void:
	var ents: Dictionary = {
		1: {"id": 1, "kind": 0, "pos": Vector2i(5, 5), "hp": 10, "max_hp": 10, "a": 5, "d": 6, "t": 10, "m": 500, "s": 150},
		600: {"id": 600, "kind": 1, "pos": Vector2i(6, 5), "hp": 5, "max_hp": 10, "mkind": 0, "a": 6, "d": 2, "t": 10,
				"m": 300, "s": 150, "h": 9},
	}
	var layer := EntityLayer.new()
	layer.tick_ms = 100.0
	layer.store = _anim_store(ents, 10)
	layer.refresh(1000.0)  # render = tick 9 = 900 ms; the swing starts at 1000, its atk frame at 1300
	assert_eq(layer.shown_pops.size(), 0)
	assert_eq(layer.pops.pending(), 1)
	assert_eq(layer.sprite_of(600).clip_name, "idle", "the flinch waits for the atk frame")
	var later: Dictionary = ents.duplicate(true)
	(later[600] as Dictionary).erase("h")
	layer.store.restore({"tick": 13, "snaps": 2, "ents": later})
	layer.refresh(1300.0)  # tick 13 arrives on time
	layer.refresh(1399.0)  # render 1299
	assert_eq(layer.shown_pops.size(), 0)
	assert_eq(layer.sprite_of(1).frame, 2)
	layer.refresh(1401.0)  # render 1300: atk frame and number together
	assert_eq(layer.sprite_of(1).frame, 3)
	assert_eq(layer.shown_pops.size(), 1)
	assert_eq(int(layer.shown_pops[0]["dmg"]), 9)
	assert_eq(layer.sprite_of(600).clip_name, "hurt")
	layer.free()


func test_layer_animates_monster_walk_and_hurt_flash() -> void:
	var ents: Dictionary = {
		600: {"id": 600, "kind": 1, "pos": Vector2i(6, 5), "hp": 5, "max_hp": 10, "mkind": 0, "a": 6, "d": 0, "t": 12,
				"m": 300, "s": 150},
	}
	var layer := EntityLayer.new()
	layer.tick_ms = 100.0
	layer.store = _anim_store(ents, 15)
	layer.refresh(1000.0)  # render = tick 14: 200 ms into the flinch
	var sp: EntitySprite = layer.sprite_of(600)
	assert_true(sp.animated)
	assert_eq(sp.clip_name, "hurt")
	assert_true(sp.flash > 0.0)
	layer.refresh(1101.0)  # render = tick 15: the 300 ms flinch is over
	assert_eq(sp.clip_name, "idle")
	layer.free()


func test_layer_plays_dead_clip_for_a_monster_gone_at_hp_0() -> void:
	var s := EntityStore.new()
	s.restore({"tick": 10, "snaps": 1, "ents": {600: {"id": 600, "kind": 1, "pos": Vector2i(3, 3), "hp": 0, "max_hp": 10,
			"mkind": 0, "a": 6, "d": 0, "t": 9, "m": 300, "s": 150},
			601: {"id": 601, "kind": 1, "pos": Vector2i(5, 3), "hp": 7, "max_hp": 10, "mkind": 0}}})
	var layer := EntityLayer.new()
	layer.tick_ms = 100.0
	layer.store = s
	layer.refresh(1000.0)
	s.apply_snap({"tick": 11, "ents": [], "gone": [600, 601]})
	layer.refresh(1100.0)
	assert_true(layer.has_sprite(600), "the dead monster stays as a corpse")
	assert_false(layer.has_sprite(601), "a monster leaving view alive goes at once")
	s.apply_snap({"tick": 16, "ents": [], "gone": []})
	layer.refresh(1600.0)  # render 1500, the corpse lasts until render 2200
	assert_eq(layer.sprite_of(600).clip_name, "dead")
	s.apply_snap({"tick": 40, "ents": [], "gone": []})
	layer.refresh(4000.0)
	assert_false(layer.has_sprite(600), "the corpse is removed after CORPSE_MS")
	layer.free()


func test_store_keeps_anim_fields_and_drops_stale_h() -> void:
	var s := EntityStore.new()
	s.apply_snap({"tick": 1, "full": true, "ents": [{"id": 5, "kind": 0, "job": 3, "pos": [1, 1], "a": 5, "d": 6,
			"t": 1, "m": 500, "s": 150, "h": 12}], "gone": []})
	var e: Dictionary = s.get_ent(5)
	assert_eq(int(e["a"]), 5)
	assert_eq(int(e["h"]), 12)
	assert_eq(int(e["job"]), 3)
	s.apply_snap({"tick": 2, "ents": [{"id": 5, "kind": 0, "pos": [2, 1], "a": 1}], "gone": []})
	e = s.get_ent(5)
	assert_eq(int(e["a"]), 1)
	assert_eq(int(e["d"]), 6, "missing fields keep their old value")
	assert_eq(int(e["s"]), 150)
	assert_eq(int(e["job"]), 3)
	assert_false(e.has("h"), "h is for its own tick only")


func test_layer_shows_miss_text_once_per_tick() -> void:
	var ents: Dictionary = {
		600: {"id": 600, "kind": 1, "pos": Vector2i(6, 5), "hp": 5, "max_hp": 10, "mkind": 0, "h": 0},
	}
	var layer := EntityLayer.new()
	layer.tick_ms = 100.0
	layer.store = _anim_store(ents, 10)
	layer.refresh(1000.0)
	assert_eq(layer.shown_pops.size(), 1)
	assert_eq(int(layer.shown_pops[0]["dmg"]), 0)
	assert_eq(layer.pops.pending(), 1)
	var dl: DamageLayer = null
	for c: Node in layer.get_children():
		if c is DamageLayer:
			dl = c
	assert_true(dl != null and dl.items.size() == 1)
	assert_eq(String(dl.items[0]["text"]), "Miss")
	layer.store.restore({"tick": 10, "snaps": 2, "ents": ents})  # the same row again in the same tick
	layer.refresh(1010.0)
	assert_eq(layer.pops.pending(), 1, "no second miss for the same tick")
	layer.free()


func test_layer_drops_a_monster_that_showed_dead_at_once() -> void:
	var s := EntityStore.new()
	s.restore({"tick": 10, "snaps": 1, "ents": {600: {"id": 600, "kind": 1, "pos": Vector2i(3, 3), "hp": 0,
			"max_hp": 10, "mkind": 0, "a": 7, "d": 0, "t": 9, "m": 300, "s": 150}}})
	var layer := EntityLayer.new()
	layer.tick_ms = 100.0
	layer.store = s
	layer.refresh(1000.0)
	assert_true(layer.has_sprite(600))
	s.apply_snap({"tick": 11, "ents": [], "gone": [600]})
	layer.refresh(1100.0)
	assert_false(layer.has_sprite(600), "the server's corpse already played: no CORPSE_MS wait")
	layer.free()
