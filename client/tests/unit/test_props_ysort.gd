extends TestCase
## Y-sorted props (round 4, ro3-art): the Props layer of MapRenderer sorts its tiles by a base origin so that
## characters draw behind a tree they stand north of and in front of one they stand south of.

const TREE: int = 5
const TEMPLE: int = 3


func _map(size: int) -> MapData:
	var m := MapData.new()
	m.size = size
	m.terrain = PackedByteArray()
	m.terrain.resize(size * size)
	m.pz = PackedByteArray()
	m.pz.resize(size * size)
	return m


func test_sort_origin_arithmetic() -> void:
	# base 32 px below the cell centre (the cell bottom of a 64 px cell), feet 0.28 tiles = 18 px: origin 14 px
	assert_eq(MapRenderer.prop_sort_origin(32, 64), 14)
	# an entity on the tile south of the prop: its centre is 64 px below the prop's cell centre, well past 14
	assert_true(64 > MapRenderer.prop_sort_origin(32, 64))
	# an entity on the tile north: -64 px, before the origin
	assert_true(-64 < MapRenderer.prop_sort_origin(32, 64))
	# the origin compares feet with base: feet of an entity on the same row (+18) sit above the base (+32)
	assert_true(18 < 32)


func test_props_layer_is_y_sorted_with_base_origins() -> void:
	if not FileAccess.file_exists(ViewMath.TILESET_INDEX):
		assert_true(true, "tiles.json not present")
		return
	var r := MapRenderer.new()
	var m := _map(8)
	_put(m, 3, 3, TREE)
	for y: int in range(4, 6):
		for x: int in range(5, 7):
			_put(m, x, y, TEMPLE)
	r.build(m)
	assert_true(r.uses_tileset, "the tileset built")
	assert_true(r.props_layer != null and r.props_layer.tile_set != null, "props layer built")
	assert_true(r.props_layer.y_sort_enabled, "Props is y-sorted")
	assert_true(r.y_sort_enabled, "MapRenderer is y-sorted so the nested Props cells join the parent's sort")
	assert_false(r.props_layer.z_as_relative, "Props z is absolute")
	assert_eq(r.props_layer.z_index, 0)
	assert_eq(r.z_index, MapRenderer.TERRAIN_Z, "terrain stays below everything")
	assert_true(r.prop_tiles >= 2, "a tree and a fountain were placed")
	var src: TileSetAtlasSource = r.props_layer.tile_set.get_source(MapRenderer.SOURCE_ID) as TileSetAtlasSource
	assert_true(src != null)
	var seen: int = 0
	for i: int in range(src.get_tiles_count()):
		var at: Vector2i = src.get_tile_id(i)
		var td: TileData = src.get_tile_data(at, 0)
		assert_eq(td.y_sort_origin, MapRenderer.prop_sort_origin(32, r.cell_px), "every prop stands on its cell")
		seen += 1
	assert_true(seen > 0)
	r.free()


func test_roofs_cover_wall_blocks_by_neighbour_mask() -> void:
	if not FileAccess.file_exists(ViewMath.TILESET_INDEX) or not FileAccess.file_exists("res://assets/tiles/roofs.png"):
		assert_true(true, "tiles.json / roofs.png not present")
		return
	var r := MapRenderer.new()
	var m := _map(8)
	for y: int in range(2, 4):  # a 3x2 wall block at (2..4, 2..3)
		for x: int in range(2, 5):
			_put(m, x, y, 6)
	r.build(m)
	assert_true(r.props_layer != null and r.props_layer.tile_set != null)
	assert_true(r.props_layer.tile_set.has_source(MapRenderer.ROOF_SOURCE_ID), "roof atlas source")
	# every wall cell carries a roof tile whose atlas column is its wall-neighbour mask (N1 E2 S4 W8)
	var top_left: Vector2i = r.props_layer.get_cell_atlas_coords(Vector2i(2, 2))
	assert_eq(r.props_layer.get_cell_source_id(Vector2i(2, 2)), MapRenderer.ROOF_SOURCE_ID)
	assert_eq(top_left, Vector2i(2 | 4, 0), "top-left corner: wall to the E and S")
	assert_eq(r.props_layer.get_cell_atlas_coords(Vector2i(3, 2)), Vector2i(2 | 4 | 8, 0), "top edge")
	assert_eq(r.props_layer.get_cell_atlas_coords(Vector2i(4, 3)), Vector2i(1 | 8, 0), "bottom-right corner")
	assert_eq(r.props_layer.get_cell_atlas_coords(Vector2i(3, 3)), Vector2i(1 | 2 | 8, 0), "bottom edge")
	assert_eq(r.props_layer.get_cell_source_id(Vector2i(5, 2)), -1, "grass has no roof")
	assert_eq(MapRenderer.wall_mask(m.terrain, m.size, 3, 2, 6), 2 | 4 | 8)
	assert_eq(MapRenderer.wall_mask(m.terrain, m.size, 0, 0, 6), 0)
	r.free()


func test_roof_feature_rows_and_rule_props() -> void:
	# house fronts only on a straight south face (S open, W and E wall), picked by hash in 1 of `every`
	assert_eq(MapRenderer.roof_feature_row(2 | 8, 3, 3, 0), 0, "every 0: plain")
	assert_eq(MapRenderer.roof_feature_row(2 | 4 | 8, 3, 3, 3), 0, "S wall: no front")
	assert_eq(MapRenderer.roof_feature_row(2, 3, 3, 3), 0, "a corner gets no front")
	var seen: Dictionary = {}
	for x: int in range(40):
		seen[MapRenderer.roof_feature_row(2 | 8 | 1, x, 7, 3)] = true
	assert_true(seen.has(1) and seen.has(2), "doors and windows both appear along a long face")
	if not FileAccess.file_exists(ViewMath.TILESET_INDEX):
		return
	var r := MapRenderer.new()
	var m := _map(12)
	for x: int in range(12):
		_put(m, x, 5, 1)  # a road
		_put(m, x, 4, 3)  # the temple square north of it
		_put(m, x, 9, 12)  # a shop row
		_put(m, x, 10, 2)  # floor south of the shops
	r.build(m)
	var lamp: Vector2i = MapRenderer._vec2i((r._index["props"]["cells"]["lamp"] as Dictionary)["atlas"])
	var well: Vector2i = MapRenderer._vec2i((r._index["props"]["cells"]["well"] as Dictionary)["atlas"])
	var lamps: int = 0
	var wells: int = 0
	for x: int in range(12):
		if r.props_layer.get_cell_atlas_coords(Vector2i(x, 5)) == lamp:
			lamps += 1
		if r.props_layer.get_cell_atlas_coords(Vector2i(x, 10)) == well:
			wells += 1
	assert_eq(lamps, 2, "lamps every 6 along x at x = 0 and 6")
	assert_eq(wells, 2, "wells every 8 along x at x = 0 and 8")
	assert_eq(r.props_layer.get_cell_source_id(Vector2i(0, 0)), -1, "grass away from everything stays bare")
	r.free()


func _put(m: MapData, x: int, y: int, t: int) -> void:
	m.terrain[y * m.size + x] = t
