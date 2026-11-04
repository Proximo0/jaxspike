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


func _put(m: MapData, x: int, y: int, t: int) -> void:
	m.terrain[y * m.size + x] = t
