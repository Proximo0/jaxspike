extends TestCase
## PathPreview: 8-connected A* with 10/14 costs, no corner cutting, the 64-cell cap and trimming.


static func _feel_map(rows: Array[String]) -> MapData:
	var m := MapData.new()
	m.size = rows.size()
	m.terrain = PackedByteArray()
	m.terrain.resize(m.size * m.size)
	for y: int in range(m.size):
		for x: int in range(m.size):
			m.terrain[y * m.size + x] = 6 if rows[y][x] == "#" else 0
	return m


func test_straight_and_diagonal() -> void:
	var m: MapData = _feel_map([".....", ".....", ".....", ".....", "....."])
	var p: Array[Vector2i] = PathPreview.find(m, Vector2i(0, 0), Vector2i(4, 4))
	assert_eq(p.size(), 4, "pure diagonal takes 4 steps")
	assert_eq(p[p.size() - 1], Vector2i(4, 4))
	assert_eq(PathPreview.find(m, Vector2i(0, 0), Vector2i(4, 0)).size(), 4)
	assert_true(PathPreview.find(m, Vector2i(2, 2), Vector2i(2, 2)).is_empty(), "start == goal")


func test_no_corner_cutting() -> void:
	var m: MapData = _feel_map([".#.", "...", "..."])
	# (0,0) -> (1,1) diagonally would cut past the wall at (1,0): must go down then right.
	var p: Array[Vector2i] = PathPreview.find(m, Vector2i(0, 0), Vector2i(1, 1))
	assert_eq(p.size(), 2)
	assert_eq(p[0], Vector2i(0, 1))
	assert_eq(p[1], Vector2i(1, 1))
	assert_false(PathPreview.can_step(m, Vector2i(0, 0), Vector2i(1, 1)))


func test_blocked_goal_and_walls() -> void:
	var m: MapData = _feel_map(["..#..", "..#..", "..#..", "..#..", "....."])
	assert_true(PathPreview.find(m, Vector2i(0, 0), Vector2i(2, 0)).is_empty(), "goal on a wall")
	var p: Array[Vector2i] = PathPreview.find(m, Vector2i(0, 0), Vector2i(4, 0))
	assert_eq(p[p.size() - 1], Vector2i(4, 0))
	for c: Vector2i in p:
		assert_true(PathPreview.walkable(m, c), "path stays on floor: %s" % str(c))
	var enclosed: MapData = _feel_map([".#...", "##...", ".....", ".....", "....."])
	assert_true(PathPreview.find(enclosed, Vector2i(0, 0), Vector2i(4, 4)).is_empty(), "no way out")


func test_cap_and_trim() -> void:
	var rows: Array[String] = []
	for y: int in range(100):
		rows.append(".".repeat(100))
	var m: MapData = _feel_map(rows)
	var p: Array[Vector2i] = PathPreview.find(m, Vector2i(0, 0), Vector2i(99, 0))
	assert_eq(p.size(), PathPreview.MAX_CELLS, "capped")
	assert_eq(p[0], Vector2i(1, 0))
	var t: Array[Vector2i] = PathPreview.trim(p, Vector2i(3, 0))
	assert_eq(t[0], Vector2i(4, 0))
	assert_eq(PathPreview.trim(p, Vector2i(50, 50)).size(), p.size(), "off-path position keeps the path")


func test_heuristic_octile() -> void:
	assert_eq(PathPreview.heuristic(Vector2i(0, 0), Vector2i(3, 1)), 34)
