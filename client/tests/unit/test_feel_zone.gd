extends TestCase
## ZoneBanner.zone_at (welcome zones, dungeon boxes, towns by pz, fields by nearest town) and the fade curve.


static func _feel_world() -> Dictionary:
	return {"towns": [{"idx": 0, "name": "Prontera", "center": [50, 50]}, {"idx": 1, "name": "Geffen", "center": [10, 10]}],
		"dungeons": [{"idx": 0, "name": "Prontera Culvert", "box": [40, 0, 60, 8]}]}


static func _feel_zone_map() -> MapData:
	var m := MapData.new()
	m.size = 64
	m.terrain = PackedByteArray()
	m.terrain.resize(64 * 64)
	m.pz = PackedByteArray()
	m.pz.resize(64 * 64)
	for y: int in range(45, 56):
		for x: int in range(45, 56):
			m.pz[y * 64 + x] = 1
	m.terrain[30 * 64 + 30] = 11
	return m


func test_fallback_regions() -> void:
	var w: Dictionary = _feel_world()
	var m: MapData = _feel_zone_map()
	assert_eq(ZoneBanner.zone_at(w, m, Vector2i(50, 50), 0)["name"], "Prontera")
	assert_eq(ZoneBanner.zone_at(w, m, Vector2i(50, 50), 0)["kind"], ZoneBanner.KIND_TOWN)
	assert_eq(ZoneBanner.zone_at(w, m, Vector2i(50, 4), 0)["name"], "Prontera Culvert")
	assert_eq(ZoneBanner.zone_at(w, m, Vector2i(50, 4), 0)["kind"], ZoneBanner.KIND_DUNGEON)
	assert_eq(ZoneBanner.zone_at(w, m, Vector2i(14, 14), 0)["name"], "Geffen Field")
	assert_eq(ZoneBanner.zone_at(w, m, Vector2i(30, 30), 0)["kind"], ZoneBanner.KIND_DUNGEON, "cave floor")
	assert_eq(ZoneBanner.zone_at({}, m, Vector2i(50, 50), 0)["name"], "Town", "no world block")
	assert_eq(ZoneBanner.zone_at({}, m, Vector2i(1, 1), 0)["name"], "Field")


func test_welcome_zones_win() -> void:
	var w: Dictionary = _feel_world()
	w["zones"] = [{"name": "prt_fild08", "title": "Prontera Field 08", "box": [0, 0, 63, 63], "kind": "field"}]
	var z: Dictionary = ZoneBanner.zone_at(w, _feel_zone_map(), Vector2i(50, 50), 0)
	assert_eq(z["name"], "Prontera Field 08")
	assert_eq(z["key"], "zone:prt_fild08")


func test_observe_and_alpha() -> void:
	var b := ZoneBanner.new()
	var w: Dictionary = _feel_world()
	var m: MapData = _feel_zone_map()
	assert_true(b.observe(w, m, Vector2i(50, 50), 0), "first zone shows")
	assert_false(b.observe(w, m, Vector2i(51, 50), 0), "same zone: no restart")
	assert_true(b.observe(w, m, Vector2i(50, 3), 0), "dungeon entry")
	assert_eq(b.shown, 2)
	assert_eq(ZoneBanner.alpha_at(-1.0), 0.0)
	assert_near(ZoneBanner.alpha_at(ZoneBanner.FADE_IN_S * 0.5), 0.5)
	assert_eq(ZoneBanner.alpha_at(1.0), 1.0)
	assert_eq(ZoneBanner.alpha_at(100.0), 0.0)
	b.free()
