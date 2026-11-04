extends TestCase
## Terrain tileset: ViewMath autotile helpers (hash, weighted pick, blob4 masks, rows, block centres, prop variants)
## and MapRenderer drawing from client/assets/tiles (tileset, props layer, scale) or the flat-colour fallback.

const GRASS: int = 0
const ROAD: int = 1
const TEMPLE: int = 3
const WATER: int = 4
const TREE: int = 5
const WAYBELL: int = 9
const SHOP: int = 12


func _map(size: int) -> MapData:
	var m := MapData.new()
	m.size = size
	m.terrain = PackedByteArray()
	m.terrain.resize(size * size)
	m.pz = PackedByteArray()
	m.pz.resize(size * size)
	return m


func _put(m: MapData, x: int, y: int, t: int) -> void:
	m.terrain[y * m.size + x] = t


func test_hash_matches_preview_tool() -> void:
	# values from preview_tiles.py tile_hash: the offline review shows what Godot draws
	assert_eq(ViewMath.tile_hash(0, 0), 0)
	assert_eq(ViewMath.tile_hash(1, 2), 1107519889)
	assert_eq(ViewMath.tile_hash(255, 256), 766734439)
	assert_eq(ViewMath.tile_hash(511, 511), 906616237)


func test_weighted_pick() -> void:
	var w := PackedInt32Array([3, 1])
	assert_eq(ViewMath.weighted_pick(0, w), 0)
	assert_eq(ViewMath.weighted_pick(2, w), 0)
	assert_eq(ViewMath.weighted_pick(3, w), 1)
	assert_eq(ViewMath.weighted_pick(7, w), 1)
	assert_eq(ViewMath.weighted_pick(5, PackedInt32Array()), 0, "no weights")


func test_blob4_mask() -> void:
	var m := _map(4)
	for x: int in range(4):
		_put(m, x, 1, WATER)
	_put(m, 1, 0, WATER)
	var bits: int = 1 << WATER
	assert_eq(ViewMath.blob4_mask(m.terrain, 4, 1, 1, bits), ViewMath.MASK_N | ViewMath.MASK_E | ViewMath.MASK_W)
	assert_eq(ViewMath.blob4_mask(m.terrain, 4, 0, 1, bits), ViewMath.MASK_E, "map edge never connects")
	assert_eq(ViewMath.blob4_mask(m.terrain, 4, 1, 0, bits), ViewMath.MASK_S)
	assert_eq(ViewMath.blob4_mask(m.terrain, 4, 3, 2, bits), ViewMath.MASK_N, "grass cell reads water above")


func test_tile_rows_modes() -> void:
	var m := _map(3)
	_put(m, 1, 1, ROAD)
	_put(m, 1, 0, ROAD)
	_put(m, 2, 1, WAYBELL)
	var modes := PackedInt32Array([ViewMath.TILE_VARIANTS, ViewMath.TILE_BLOB4])
	modes.resize(13)
	var connects := PackedInt32Array([0, (1 << ROAD) | (1 << WAYBELL)])
	connects.resize(13)
	var weights: Array[PackedInt32Array] = [PackedInt32Array([1, 1, 1, 1])]
	for i: int in range(12):
		weights.append(PackedInt32Array())
	var rows: PackedByteArray = ViewMath.tile_rows(m.terrain, 3, modes, connects, weights)
	assert_eq(rows[1 * 3 + 1], ViewMath.MASK_N | ViewMath.MASK_E, "road joins road and the waybell")
	assert_eq(rows[0 * 3 + 1], ViewMath.MASK_S)
	assert_eq(rows[2 * 3 + 2], 0, "waybell is single")
	assert_eq(rows[0], ViewMath.weighted_pick(ViewMath.tile_hash(0, 0), weights[0]))
	for i: int in range(9):
		assert_true(rows[i] < 16)


func test_block_centers_and_prop_variant() -> void:
	var m := _map(20)
	for y: int in range(3, 15):
		for x: int in range(4, 16):
			_put(m, x, y, TEMPLE)
	_put(m, 0, 0, TEMPLE)
	var c: Array[Vector2i] = ViewMath.block_centers(m.terrain, 20, TEMPLE)
	assert_eq(c.size(), 2)
	assert_eq(c[0], Vector2i(0, 0), "a 1x1 block is its own centre")
	assert_eq(c[1], Vector2i(9, 8), "12x12 from (4,3): centre tile 4+5, 3+5")
	assert_eq(ViewMath.prop_variant(5, 9, 4, true), 1, "shops cycle by x")
	assert_eq(ViewMath.prop_variant(5, 9, 1, false), 0)
	var v: int = ViewMath.prop_variant(5, 9, 3, false)
	assert_true(v >= 0 and v < 3)


func test_renderer_draws_the_tileset() -> void:
	assert_true(FileAccess.file_exists(ViewMath.TILESET_INDEX), "tiles.json exists")
	var m := _map(16)
	for x: int in range(16):
		_put(m, x, 2, WATER)
	_put(m, 5, 5, TREE)
	for x: int in range(8, 12):
		_put(m, x, 9, SHOP)
	for y: int in range(10, 14):
		for x: int in range(2, 6):
			_put(m, x, y, TEMPLE)
	var r := MapRenderer.new()
	assert_eq(r.build(m), 256)
	assert_true(r.uses_tileset, "tileset loaded")
	assert_eq(r.cell_px, 64)
	assert_eq(r.get_used_cells().size(), 256)
	assert_eq(r.get_cell_atlas_coords(Vector2i(5, 5)), Vector2i(TREE, 0), "tree base")
	assert_eq(r.get_cell_atlas_coords(Vector2i(4, 2)), Vector2i(WATER, ViewMath.MASK_E | ViewMath.MASK_W))
	assert_eq(r.get_cell_atlas_coords(Vector2i(0, 2)), Vector2i(WATER, ViewMath.MASK_E))
	assert_eq(r.terrain_at_cell(Vector2i(9, 9)), SHOP)
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	assert_near(r.drawn_tile_px(), tile_px, 0.001, "64 px cells drawn at tile_px")
	assert_true(r.props_layer != null and r.props_layer.tile_set != null, "props layer")
	assert_true(r.props_layer.get_cell_source_id(Vector2i(5, 5)) >= 0, "tree prop")
	assert_true(r.props_layer.get_cell_source_id(Vector2i(8, 9)) >= 0, "stall prop")
	assert_true(r.props_layer.get_cell_atlas_coords(Vector2i(8, 9)) != r.props_layer.get_cell_atlas_coords(
			Vector2i(9, 9)), "neighbouring stalls differ")
	assert_true(r.props_layer.get_cell_source_id(Vector2i(3, 11)) >= 0, "fountain at the temple centre")
	assert_eq(r.props_layer.get_cell_source_id(Vector2i(0, 0)), -1, "no prop on grass")
	assert_eq(r.prop_tiles, 1 + 4 + 1)
	r.free()


func test_renderer_flat_fallback() -> void:
	var m := _map(8)
	_put(m, 3, 2, TREE)
	_put(m, 4, 2, 12)
	var r := MapRenderer.new()
	r.index_path = "res://assets/tiles/missing.json"
	assert_eq(r.build(m), 64)
	assert_false(r.uses_tileset)
	var px: int = r.built_tile_px
	assert_eq(r.atlas_size(), Vector2i(px * MapRenderer.TERRAIN_IDS, px))
	assert_eq(r.get_cell_atlas_coords(Vector2i(3, 2)), Vector2i(TREE, 0))
	assert_eq(r.get_cell_atlas_coords(Vector2i(4, 2)), Vector2i(12, 0), "shop id 12 has its own colour")
	assert_eq(r.scale, Vector2.ONE)
	r.free()


func test_full_map_build_time() -> void:
	# 512 x 512 like the live map: a field of grass with water, trees, roads and a temple square
	var size: int = 512
	var m := _map(size)
	var rng := RandomNumberGenerator.new()
	rng.seed = 7
	for i: int in range(size * size):
		var p: float = rng.randf()
		m.terrain[i] = TREE if p < 0.12 else (WATER if p < 0.2 else (ROAD if p < 0.24 else GRASS))
	for y: int in range(250, 262):
		for x: int in range(250, 262):
			_put(m, x, y, TEMPLE)
	var r := MapRenderer.new()
	var t0: int = Time.get_ticks_msec()
	r.build(m)
	var ms: int = Time.get_ticks_msec() - t0
	assert_eq(r.get_used_cells().size(), size * size)
	assert_true(ms < 2500, "512x512 build took %d ms" % ms)
	r.free()
