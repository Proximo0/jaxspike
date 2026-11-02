extends TestCase


func test_world_to_tile_floors() -> void:
	assert_eq(ViewMath.world_to_tile(Vector2(0, 0), 32.0), Vector2i(0, 0))
	assert_eq(ViewMath.world_to_tile(Vector2(31.9, 32.0), 32.0), Vector2i(0, 1))
	assert_eq(ViewMath.world_to_tile(Vector2(-0.1, -32.1), 32.0), Vector2i(-1, -2))


func test_screen_to_tile_centre_is_camera_tile() -> void:
	var cam: Vector2 = ViewMath.tile_center(Vector2(10, 20), 32.0)
	var vp := Vector2(1920, 1080)
	assert_eq(ViewMath.screen_to_tile(vp * 0.5, cam, 1.0, vp, 32.0), Vector2i(10, 20))
	assert_eq(ViewMath.screen_to_tile(vp * 0.5 + Vector2(32, 0), cam, 1.0, vp, 32.0), Vector2i(11, 20))


func test_screen_to_tile_zoom() -> void:
	var cam: Vector2 = ViewMath.tile_center(Vector2(10, 20), 32.0)
	var vp := Vector2(1000, 800)
	# At zoom 2 one tile is 64 screen pixels.
	assert_eq(ViewMath.screen_to_tile(vp * 0.5 + Vector2(64, 64), cam, 2.0, vp, 32.0), Vector2i(11, 21))
	assert_eq(ViewMath.screen_to_tile(vp * 0.5 + Vector2(-33, 0), cam, 2.0, vp, 32.0), Vector2i(9, 20))


func test_in_map() -> void:
	assert_true(ViewMath.in_map(Vector2i(0, 511), 512))
	assert_false(ViewMath.in_map(Vector2i(512, 0), 512))
	assert_false(ViewMath.in_map(Vector2i(-1, 3), 512))


func test_kind_colors() -> void:
	assert_eq(ViewMath.kind_color(0, true), Color(1, 1, 1))
	assert_eq(ViewMath.kind_color(0, false), ViewMath.PLAYER_COLOR)
	assert_eq(ViewMath.kind_color(101, false), ViewMath.DROP_COLOR)
	assert_eq(ViewMath.kind_color(1, false), ViewMath.MONSTER_COLORS[0])
	assert_eq(ViewMath.kind_color(2, false), ViewMath.MONSTER_COLORS[1])
	assert_true(ViewMath.kind_color(1, false) != ViewMath.kind_color(2, false))
	assert_eq(ViewMath.category(0), ViewMath.CAT_PLAYER)
	assert_eq(ViewMath.category(5), ViewMath.CAT_MONSTER)
	assert_eq(ViewMath.category(100), ViewMath.CAT_DROP)
	assert_false(ViewMath.has_hp_bar(102))
	assert_true(ViewMath.has_hp_bar(3))


func test_terrain_colors_distinct() -> void:
	assert_eq(ViewMath.TERRAIN_COLORS.size(), 8)
	for i: int in range(8):
		for j: int in range(i + 1, 8):
			assert_true(ViewMath.terrain_color(i) != ViewMath.terrain_color(j), "%d vs %d" % [i, j])


func test_to_vec_and_hp() -> void:
	assert_eq(ViewMath.to_vec([3, 4]), Vector2(3, 4))
	assert_eq(ViewMath.to_vec(Vector2i(5, 6)), Vector2(5, 6))
	assert_eq(ViewMath.to_vec("x"), Vector2.ZERO)
	assert_near(ViewMath.hp_fraction(50, 200), 0.25)
	assert_near(ViewMath.hp_fraction(5, 0), 0.0)
	assert_near(ViewMath.hp_fraction(500, 200), 1.0)


func test_tuned_falls_back_for_unknown_field() -> void:
	assert_near(ViewMath.tuned("no_such_row", 7.5), 7.5)


func test_blocked_terrain_ids() -> void:
	for t: int in [4, 5, 6, 7]:
		assert_true(ViewMath.is_blocked_terrain(t), str(t))
	for t: int in [-2, -1, 0, 1, 2, 3, 8]:
		assert_false(ViewMath.is_blocked_terrain(t), str(t))


func test_is_blocked_tile_reads_map() -> void:
	var m := MapData.new()
	m.size = 2
	m.terrain = PackedByteArray([0, 4, 6, 3])
	assert_false(ViewMath.is_blocked_tile(m, Vector2i(0, 0)))
	assert_true(ViewMath.is_blocked_tile(m, Vector2i(1, 0)))
	assert_true(ViewMath.is_blocked_tile(m, Vector2i(0, 1)))
	assert_false(ViewMath.is_blocked_tile(m, Vector2i(1, 1)))
	assert_false(ViewMath.is_blocked_tile(m, Vector2i(5, 5)))
	assert_false(ViewMath.is_blocked_tile(MapData.new(), Vector2i(0, 0)))
	assert_false(ViewMath.is_blocked_tile(null, Vector2i(0, 0)))


func test_marker_done() -> void:
	assert_true(ViewMath.marker_done(Vector2i(3, 3), Vector2i(3, 3), 0))
	assert_false(ViewMath.marker_done(Vector2i(3, 3), Vector2i(1, 1), 9))
	assert_true(ViewMath.marker_done(Vector2i(3, 3), Vector2i(1, 1), 10))
	assert_true(ViewMath.marker_done(Vector2i(3, 3), Vector2i(1, 1), 2, 2))


func test_walk_marker_lifecycle() -> void:
	var mk := WalkMarker.new()
	mk.show_target(Vector2i(5, 5), Vector2i(1, 1), 32.0)
	assert_true(mk.active)
	assert_eq(mk.position, Vector2(160, 160))
	mk.observe(Vector2i(2, 1), 1)
	mk.observe(Vector2i(2, 1), 2)
	assert_true(mk.active)
	for t: int in range(3, 3 + ViewMath.MARKER_STALL_TICKS):
		mk.observe(Vector2i(2, 1), t)
	assert_false(mk.active, "hidden after the stall limit")
	mk.show_target(Vector2i(5, 5), Vector2i(2, 1), 32.0)
	mk.observe(Vector2i(5, 5), 20)
	assert_false(mk.active, "hidden on arrival")
	mk.flash_blocked(Vector2i(7, 7), 32.0)
	assert_true(mk.active and mk.blocked_flash)
	mk._process(1.0)
	assert_false(mk.active)
	mk.free()
