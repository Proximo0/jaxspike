extends TestCase
## WalkMarker path preview: computed with a map, trimmed as the player walks, cleared on hide; TargetRing radii.


func test_path_preview_lifecycle() -> void:
	var m := MapData.new()
	m.size = 16
	m.terrain = PackedByteArray()
	m.terrain.resize(256)
	var mk := WalkMarker.new()
	mk.show_target(Vector2i(5, 1), Vector2i(1, 1), 32.0)
	assert_true(mk.path.is_empty(), "no map: no preview")
	mk.map = m
	mk.show_target(Vector2i(5, 1), Vector2i(1, 1), 32.0)
	assert_eq(mk.path.size(), 4)
	mk.observe(Vector2i(2, 1), 1)
	assert_eq(mk.path.size(), 3, "trimmed")
	mk.flash_blocked(Vector2i(7, 7), 32.0)
	assert_true(mk.path.is_empty(), "blocked flash has no path")
	mk.show_target(Vector2i(5, 1), Vector2i(1, 1), 32.0)
	mk.hide_marker()
	assert_true(mk.path.is_empty())
	mk.free()


func test_ring_radii() -> void:
	var r: Vector2 = TargetRing.radii(32.0, 0.0)
	assert_near(r.x, 32.0 * 0.48)
	assert_near(r.y, r.x * 0.5)
