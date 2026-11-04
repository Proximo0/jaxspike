extends TestCase
## Terrain tileset: ViewMath autotile helpers (hash, weighted pick, blob4 and blob47 masks, rows, block centres,
## prop variants, node props)
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
	assert_eq(r.get_cell_atlas_coords(Vector2i(4, 2)), Vector2i(WATER,
			ViewMath.blob47_row(ViewMath.MASK_E | ViewMath.MASK_W)), "water is blob47")
	assert_eq(r.get_cell_atlas_coords(Vector2i(0, 2)), Vector2i(WATER, ViewMath.blob47_row(ViewMath.MASK_E)))
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


func test_blob47_masks_match_the_tileset() -> void:
	var masks: PackedInt32Array = ViewMath.blob47_masks()
	assert_eq(masks.size(), ViewMath.BLOB47_COUNT)
	assert_eq(masks[0], 0)
	assert_eq(masks[masks.size() - 1], 255)
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(ViewMath.TILESET_INDEX))
	assert_true(parsed is Dictionary, "tiles.json parses")
	var listed: Array = (parsed as Dictionary).get("blob47_masks", [])
	assert_eq(listed.size(), masks.size(), "gen_tiles.py wrote the same 47 masks")
	for i: int in range(mini(listed.size(), masks.size())):
		assert_eq(int(listed[i]), masks[i])
	var rows: int = int((parsed as Dictionary).get("rows", 0))
	assert_true(rows >= ViewMath.BLOB47_COUNT, "atlas has a row per mask")


func test_blob_reduce_and_rows() -> void:
	var n: int = ViewMath.MASK_N
	var e: int = ViewMath.MASK_E
	assert_eq(ViewMath.blob_reduce(ViewMath.MASK_NE), 0, "a diagonal alone does not count")
	assert_eq(ViewMath.blob_reduce(n | ViewMath.MASK_NE), n, "needs both sides")
	assert_eq(ViewMath.blob_reduce(n | e | ViewMath.MASK_NE), n | e | ViewMath.MASK_NE)
	assert_eq(ViewMath.blob47_row(255), ViewMath.BLOB47_COUNT - 1)
	assert_eq(ViewMath.blob47_row(0), 0)
	assert_eq(ViewMath.blob47_row(n | ViewMath.MASK_SW | ViewMath.MASK_SE), ViewMath.blob47_row(n),
			"dropped diagonals share the row")
	assert_true(ViewMath.blob47_row(n | e) != ViewMath.blob47_row(n | e | ViewMath.MASK_NE),
			"an inner corner has its own row")
	for m: int in range(256):
		var r: int = ViewMath.blob47_row(m)
		assert_true(r >= 0 and r < ViewMath.BLOB47_COUNT)


func test_blob8_mask_and_inner_corner_rows() -> void:
	# an L of water: (1,1) has water N and E but grass NE, so it is an inner corner
	var m := _map(4)
	for p: Vector2i in [Vector2i(1, 0), Vector2i(1, 1), Vector2i(2, 1), Vector2i(0, 0), Vector2i(0, 1)]:
		_put(m, p.x, p.y, WATER)
	var bits: int = 1 << WATER
	var raw: int = ViewMath.blob8_mask(m.terrain, 4, 1, 1, bits)
	assert_eq(raw, ViewMath.MASK_N | ViewMath.MASK_E | ViewMath.MASK_W | ViewMath.MASK_NW)
	assert_eq(ViewMath.blob8_mask(m.terrain, 4, 0, 0, bits), ViewMath.MASK_E | ViewMath.MASK_S | ViewMath.MASK_SE,
			"map edge never connects")
	var modes := PackedInt32Array()
	modes.resize(13)
	modes[WATER] = ViewMath.TILE_BLOB47
	var connects := PackedInt32Array()
	connects.resize(13)
	connects[WATER] = bits
	var weights: Array[PackedInt32Array] = []
	for i: int in range(13):
		weights.append(PackedInt32Array())
	var rows: PackedByteArray = ViewMath.tile_rows(m.terrain, 4, modes, connects, weights)
	assert_eq(rows[1 * 4 + 1], ViewMath.blob47_row(raw))
	assert_eq(rows[1 * 4 + 2], ViewMath.blob47_row(ViewMath.MASK_W), "the NE diagonal of (2,1) is not water")


func test_node_props_for_stations() -> void:
	var idx: Dictionary = {"cell_px": 64, "props": {"nodes": {"406": "f"}, "cells": {"f": {"atlas": [5, 2],
			"size": [1, 1]}}}}
	assert_eq(ViewMath.node_prop_region(idx, 406), Rect2i(320, 128, 64, 64))
	assert_eq(ViewMath.node_prop_region(idx, 407), Rect2i(), "no prop bound")
	for kind: int in [406, 407]:
		var tex: Texture2D = ViewMath.node_prop_texture(kind)
		assert_true(tex != null, "kind %d has a props-atlas texture" % kind)
		if tex != null:
			assert_eq(Vector2i(tex.get_size()), Vector2i(64, 64), "one tile, so the EntitySprite square fits")
	assert_true(ViewMath.node_prop_texture(400) == null, "gather nodes keep their shapes")


func test_near_rule_moves_the_row() -> void:
	# a tree beside a flagstone gets the planter row; boulders beside the cave floor get cave ground
	var m := _map(5)
	_put(m, 1, 1, TREE)
	_put(m, 2, 1, 2)  # floor
	_put(m, 3, 3, TREE)
	_put(m, 0, 4, 7)  # rock
	_put(m, 1, 4, 11)  # cave
	var modes := PackedInt32Array()
	modes.resize(13)
	modes[7] = ViewMath.TILE_VARIANTS
	var connects := PackedInt32Array()
	connects.resize(13)
	var weights: Array[PackedInt32Array] = []
	for i: int in range(13):
		weights.append(PackedInt32Array())
	weights[7] = PackedInt32Array([1, 1, 1])
	var near_bits := PackedInt32Array()
	near_bits.resize(13)
	var near_offset := PackedInt32Array()
	near_offset.resize(13)
	near_bits[TREE] = 1 << 2
	near_offset[TREE] = 1
	near_bits[7] = 1 << 11
	near_offset[7] = 3
	var rows: PackedByteArray = ViewMath.tile_rows(m.terrain, 5, modes, connects, weights, near_bits, near_offset)
	assert_eq(rows[1 * 5 + 1], 1, "town tree in a planter")
	assert_eq(rows[3 * 5 + 3], 0, "field tree on grass")
	var r: int = rows[4 * 5 + 0]
	assert_true(r >= 3 and r < 6, "rock by the cave uses the cave-ground rows (got %d)" % r)
	var plain: PackedByteArray = ViewMath.tile_rows(m.terrain, 5, modes, connects, weights)
	assert_eq(plain[1 * 5 + 1], 0, "without near arrays nothing moves")
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(ViewMath.TILESET_INDEX))
	var filled: Dictionary = (parsed as Dictionary).get("filled_rows", {})
	assert_true((filled.get("5", []) as Array).has(1.0), "the planter row is in the atlas")
	assert_true((filled.get("7", []) as Array).has(5.0), "the cave-ground rock rows are in the atlas")
