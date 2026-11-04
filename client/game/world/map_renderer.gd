class_name MapRenderer
extends TileMapLayer
## The whole map as a TileMapLayer, filled once from MapData at login.
## With the tileset of `ViewMath.TILESET_INDEX` (client/tools/art/tiles/gen_tiles.py): 64 px cells drawn at
## tile_px through the node scale, autotiled rows (blob4 masks, weighted variants) from ViewMath.tile_rows, and a
## child `Props` layer (trees, bell posts, stalls, the temple fountain) that may overhang the tile above.
## Without it: one flat colour per terrain id from ViewMath.TERRAIN_COLORS at tile_px (the old placeholder path).
## A changed Tuning.tile_px rescales in place (tileset) or swaps the colour atlas (fallback).

const TERRAIN_IDS: int = 13
const SOURCE_ID: int = 0
const ATLAS_ROWS: int = 16
const SHOP_ID: int = 12  ## game/data/terrain.py T_SHOP: its stalls cycle colours by x.

var built_tile_px: int = 0
var filled_tiles: int = 0
var prop_tiles: int = 0
var uses_tileset: bool = false
var cell_px: int = 0  ## Atlas cell size of the built tile set (64 for the tileset, tile_px for the fallback).
var props_layer: TileMapLayer = null
var index_path: String = ViewMath.TILESET_INDEX  ## Tests point this at a missing file to get the fallback.
var _source: TileSetAtlasSource = null
var _index: Dictionary = {}
var _index_loaded: bool = false


func _process(_delta: float) -> void:
	if built_tile_px == 0:
		return
	var px: int = _wanted_px()
	if px != built_tile_px:
		_apply_tile_px(px)


## Builds the tile set(s) and fills every cell from `map`. Returns the number of terrain cells set.
func build(map: MapData) -> int:
	clear()
	if props_layer != null:
		props_layer.clear()
	filled_tiles = 0
	prop_tiles = 0
	if not map.is_loaded():
		return 0
	var index: Dictionary = _load_index()
	uses_tileset = not index.is_empty() and _make_tileset(index)
	if not uses_tileset:
		_make_flat_tile_set(_wanted_px())
		_fill_flat(map)
	else:
		_fill_tileset(map, index)
	filled_tiles = map.size * map.size
	return filled_tiles


func tile_count() -> int:
	return filled_tiles


## Pixel size of the terrain atlas texture, or ZERO before the first build.
func atlas_size() -> Vector2i:
	if _source == null or _source.texture == null:
		return Vector2i.ZERO
	return Vector2i(_source.texture.get_size())


## Terrain id drawn at a cell (the atlas column), -1 for an empty cell.
func terrain_at_cell(cell: Vector2i) -> int:
	if get_cell_source_id(cell) < 0:
		return -1
	return get_cell_atlas_coords(cell).x


## World pixels per map tile as drawn (tile_set cell times the node scale).
func drawn_tile_px() -> float:
	if tile_set == null:
		return 0.0
	return float(tile_set.tile_size.x) * scale.x


func _wanted_px() -> int:
	return clampi(roundi(ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)), 4, 256)


func _apply_tile_px(px: int) -> void:
	if uses_tileset:
		scale = Vector2.ONE * (float(px) / float(cell_px))
	else:
		_source.texture = _make_flat_texture(px)
		_source.texture_region_size = Vector2i(px, px)
		tile_set.tile_size = Vector2i(px, px)
		cell_px = px
	built_tile_px = px


# ---------------------------------------------------------------- tileset path

func _load_index() -> Dictionary:
	if _index_loaded:
		return _index
	_index_loaded = true
	if not FileAccess.file_exists(index_path):
		return _index
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(index_path))
	if parsed is Dictionary:
		var d: Dictionary = parsed
		if d.has("terrains") and d.has("terrain_atlas") and int(d.get("cell_px", 0)) > 0:
			_index = d
	return _index


func _texture(path: String) -> Texture2D:
	if path == "" or not ResourceLoader.exists(path):
		return null
	return load(path) as Texture2D


func _make_tileset(index: Dictionary) -> bool:
	var tex: Texture2D = _texture(str(index.get("terrain_atlas", "")))
	if tex == null:
		return false
	cell_px = int(index["cell_px"])
	var ts := TileSet.new()
	ts.tile_size = Vector2i(cell_px, cell_px)
	_source = TileSetAtlasSource.new()
	_source.texture = tex
	_source.texture_region_size = Vector2i(cell_px, cell_px)
	var cols: int = mini(int(index.get("columns", TERRAIN_IDS)), floori(tex.get_width() / float(cell_px)))
	var rows: int = mini(int(index.get("rows", ATLAS_ROWS)), floori(tex.get_height() / float(cell_px)))
	var filled: Dictionary = index.get("filled_rows", {})
	for c: int in range(cols):
		var used: Array = filled.get(str(c), [0])
		for r: Variant in used:
			if int(r) < rows:
				_source.create_tile(Vector2i(c, int(r)))
	ts.add_source(_source, SOURCE_ID)
	tile_set = ts
	texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR
	_make_props(index)
	built_tile_px = _wanted_px()
	scale = Vector2.ONE * (float(built_tile_px) / float(cell_px))
	return true


func _make_props(index: Dictionary) -> void:
	if props_layer == null:
		props_layer = TileMapLayer.new()
		props_layer.name = "Props"
		add_child(props_layer)
	props_layer.tile_set = null
	var props: Dictionary = index.get("props", {})
	var tex: Texture2D = _texture(str(index.get("props_atlas", "")))
	if tex == null or props.is_empty():
		return
	var ts := TileSet.new()
	ts.tile_size = Vector2i(cell_px, cell_px)
	var src := TileSetAtlasSource.new()
	src.texture = tex
	src.texture_region_size = Vector2i(cell_px, cell_px)
	var cells: Dictionary = props.get("cells", {})
	for pname: Variant in cells:
		var cell: Dictionary = cells[pname]
		var at: Vector2i = _vec2i(cell.get("atlas", [0, 0]))
		var sz: Vector2i = _vec2i(cell.get("size", [1, 1]))
		if not src.has_room_for_tile(at, sz, 1, Vector2i.ZERO, 1):
			continue
		src.create_tile(at, sz)
		src.get_tile_data(at, 0).texture_origin = _vec2i(cell.get("origin", [0, 0]))
	ts.add_source(src, SOURCE_ID)
	props_layer.tile_set = ts


static func _vec2i(v: Variant) -> Vector2i:
	if v is Array and (v as Array).size() >= 2:
		var a: Array = v
		return Vector2i(int(a[0]), int(a[1]))
	return Vector2i.ZERO


func _fill_tileset(map: MapData, index: Dictionary) -> void:
	var size: int = map.size
	var terrain: PackedByteArray = map.terrain
	var modes := PackedInt32Array()
	var connects := PackedInt32Array()
	var weights: Array[PackedInt32Array] = []
	for t: int in range(TERRAIN_IDS):
		modes.append(ViewMath.TILE_SINGLE)
		connects.append(0)
		weights.append(PackedInt32Array())
	for e: Variant in index["terrains"]:
		var d: Dictionary = e
		var id: int = int(d.get("id", -1))
		if id < 0 or id >= TERRAIN_IDS:
			continue
		match str(d.get("mode", "single")):
			"blob4":
				modes[id] = ViewMath.TILE_BLOB4
				var bits: int = 0
				for c: Variant in d.get("connects", []):
					bits |= 1 << int(c)
				connects[id] = bits
			"variants":
				modes[id] = ViewMath.TILE_VARIANTS
				var wl: Array = d.get("weights", [])
				weights[id] = PackedInt32Array(wl)
	var rows: PackedByteArray = ViewMath.tile_rows(terrain, size, modes, connects, weights)
	# props chosen per terrain id: atlas coords of each variant
	var props: Dictionary = index.get("props", {})
	var cells: Dictionary = props.get("cells", {})
	var on_terrain: Dictionary = props.get("on_terrain", {})
	var prop_atlas: Array = []
	for t: int in range(TERRAIN_IDS):
		var list: Array[Vector2i] = []
		for pname: Variant in on_terrain.get(str(t), []):
			if cells.has(pname):
				list.append(_vec2i((cells[pname] as Dictionary).get("atlas", [0, 0])))
		prop_atlas.append(list)
	var shop_by_x: bool = str(props.get("shop_pick", "")) == "x"
	var with_props: bool = props_layer != null and props_layer.tile_set != null
	for y: int in range(size):
		var row: int = y * size
		for x: int in range(size):
			var t: int = terrain[row + x]
			if t >= TERRAIN_IDS:
				t = 0
			var cell := Vector2i(x, y)
			set_cell(cell, SOURCE_ID, Vector2i(t, rows[row + x]))
			if with_props:
				var list: Array[Vector2i] = prop_atlas[t]
				if not list.is_empty():
					var v: int = ViewMath.prop_variant(x, y, list.size(), shop_by_x and t == SHOP_ID)
					props_layer.set_cell(cell, SOURCE_ID, list[v])
					prop_tiles += 1
	if not with_props:
		return
	var centres: Dictionary = props.get("block_center", {})
	for key: Variant in centres:
		var pname: Variant = centres[key]
		if not cells.has(pname):
			continue
		var at: Vector2i = _vec2i((cells[pname] as Dictionary).get("atlas", [0, 0]))
		for c: Vector2i in ViewMath.block_centers(terrain, size, int(key)):
			props_layer.set_cell(c, SOURCE_ID, at)
			prop_tiles += 1


# ---------------------------------------------------------------- flat colour fallback

func _make_flat_tile_set(px: int) -> void:
	var ts := TileSet.new()
	_source = TileSetAtlasSource.new()
	_source.texture = _make_flat_texture(px)
	_source.texture_region_size = Vector2i(px, px)
	for i: int in range(TERRAIN_IDS):
		_source.create_tile(Vector2i(i, 0))
	ts.tile_size = Vector2i(px, px)
	ts.add_source(_source, SOURCE_ID)
	tile_set = ts
	scale = Vector2.ONE
	cell_px = px
	built_tile_px = px
	if props_layer != null:
		props_layer.tile_set = null


func _fill_flat(map: MapData) -> void:
	var size: int = map.size
	var terrain: PackedByteArray = map.terrain
	for y: int in range(size):
		var row: int = y * size
		for x: int in range(size):
			var t: int = terrain[row + x]
			if t >= TERRAIN_IDS:
				t = 0
			set_cell(Vector2i(x, y), SOURCE_ID, Vector2i(t, 0))


func _make_flat_texture(px: int) -> ImageTexture:
	var img: Image = Image.create(px * TERRAIN_IDS, px, false, Image.FORMAT_RGBA8)
	for i: int in range(TERRAIN_IDS):
		img.fill_rect(Rect2i(i * px, 0, px, px), ViewMath.terrain_color(i))
	return ImageTexture.create_from_image(img)
