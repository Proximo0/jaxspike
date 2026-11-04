class_name MapRenderer
extends TileMapLayer
## The whole map as a TileMapLayer, filled once from MapData at login.
## With the tileset of `ViewMath.TILESET_INDEX` (client/tools/art/tiles/gen_tiles.py): 64 px cells drawn at
## tile_px through the node scale, autotiled rows (blob47 / blob4 masks, weighted variants) from ViewMath.tile_rows, and a
## child `Props` layer (trees, bell posts, stalls, the temple fountain) that may overhang the tile above.
## Without it: one flat colour per terrain id from ViewMath.TERRAIN_COLORS at tile_px (the old placeholder path).
## A changed Tuning.tile_px rescales in place (tileset) or swaps the colour atlas (fallback).

const TERRAIN_IDS: int = 13
const SOURCE_ID: int = 0
const ATLAS_ROWS: int = 48  ## blob47 columns use 47 rows (tiles.json "rows" wins).
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


## Y-sort (round 4, ro3-art): the Props layer is y-sorted and sits at absolute z 0 while the terrain (this node)
## stays at z -1, so under a y-sorted common parent (GameView) with a y-sorted EntityLayer whose sprites are z 0,
## each prop sorts against the characters by its base: a character north of a tree draws behind it, south in
## front. The tile's y_sort_origin is the prop's `base_px` (tiles.json, px below the base cell's centre) minus the
## entity feet offset, because an entity node sits at its tile centre with the feet EntitySprite.FEET_Y lower.
const TERRAIN_Z: int = -1
## Roofs (round 4, ro3-art): the big decor wall blocks read as stone slabs, so every wall cell also gets a roof
## tile on the Props layer (tiles.json "roofs": a 16-column atlas of 1x2 tiles, column = 4-bit mask of wall
## neighbours N1 E2 S4 W8, same bits as tiles.json mask_bits), lifted above the wall and y-sorted by the wall's base.
const ROOF_SOURCE_ID: int = 1

var roof_terrain: int = -1  ## terrain id that gets roofs (-1: no roofs atlas)
var roof_feature_every: int = 0  ## tiles.json roofs.features.every (0: plain roofs only)
var roof_row_cells: int = 2  ## atlas cell rows per roof tile (its size.y): feature row r is at atlas row r * this
var roof_max_block: int = 0  ## tiles.json roofs.max_block: roofs only on wall blocks up to this wide and deep (0: all)
var roof_styles: int = 1  ## tiles.json roofs.styles count (tile red, slate blue, thatch)
var roof_feature_rows: int = 1  ## feature rows per style (plain, door, window)
var roof_style_by_hash: bool = false  ## roofs.style_mode "hash": one style per house from its corner hash
var _rules: Array[Dictionary] = []  ## prepared tiles.json "rules" rows: {on, near, every, by_x, atlas}
var _rules_by_terrain: Array = []  ## TERRAIN_IDS arrays of those rows, indexed by `on` (fast per-cell check)
var _rule_packs: Array[PackedInt32Array] = []  ## per terrain: [near, every, by_x, atlas_x, atlas_y] * n (the hot path)
var _rule_flags: PackedByteArray = PackedByteArray()  ## TERRAIN_IDS bytes: 1 when that terrain has a rule


static func prop_sort_origin(base_px: int, atlas_cell_px: int) -> int:
	return base_px - int(round(EntitySprite.FEET_Y * float(atlas_cell_px)))


## Roof atlas row for a wall cell: a house front (1 door, 2 window) goes on cells whose S side is open with wall
## neighbours W and E (a straight south face), one in `every` by tile_hash; everything else is the plain row 0.
static func roof_feature_row(mask: int, x: int, y: int, every: int) -> int:
	if every <= 0 or mask & 4 or not (mask & 2 and mask & 8):
		return 0
	var pick: int = posmod(ViewMath.tile_hash(x, y) >> 5, every)
	if pick == 1:
		return 1
	if pick == 0:
		return 2
	return 0


## tiles.json "rules" (data-driven placement, evaluated after on_terrain props): a cell of terrain `on` with a
## 4-neighbour of terrain `near` gets `prop` when (by_x ? x : tile_hash) % every == 0. Returns the prop's atlas
## coords or (-1, -1). `_rules` is prepared by _make_props from the index.
func rule_prop(terrain: PackedByteArray, size: int, x: int, y: int, t: int) -> Vector2i:
	if t < 0 or t >= _rule_packs.size():
		return Vector2i(-1, -1)
	# rules of this terrain packed as [near, every, by_x, atlas_x, atlas_y] * n: the cheap spacing test runs first
	# (most cells fail it), the four neighbour reads only for candidates, the hash at most once per cell
	var pack: PackedInt32Array = _rule_packs[t]
	var n: int = pack.size() / 5
	var hashed: int = -1
	for i: int in range(n):
		var base: int = i * 5
		var every: int = pack[base + 1]
		var pick: int = x
		if pack[base + 2] == 0:
			if hashed < 0:
				hashed = cell_hash(x, y)
			pick = hashed
		if posmod(pick, every) != 0:
			continue
		var near: int = pack[base]
		if (y > 0 and terrain[(y - 1) * size + x] == near) \
				or (x < size - 1 and terrain[y * size + x + 1] == near) \
				or (y < size - 1 and terrain[(y + 1) * size + x] == near) \
				or (x > 0 and terrain[y * size + x - 1] == near):
			return Vector2i(pack[base + 3], pack[base + 4])
	return Vector2i(-1, -1)


## True when the wall block through (x, y) is at most `max_block` cells wide and deep (its horizontal and vertical
## runs of `wall_id`); a bounded scan, at most 4 * max_block reads per wall cell. 0 or less: every block fits.
## A cheap deterministic per-cell hash for prop rules and roof styles (no call into ViewMath on the hot path):
## two large odd multipliers mixed and folded to a non-negative int.
static func cell_hash(x: int, y: int) -> int:
	var h: int = (x * 73856093) ^ (y * 19349663) ^ ((x + y) * 83492791)
	h = (h ^ (h >> 13)) * 1274126177
	return (h ^ (h >> 16)) & 0x7FFFFFFF


## Top-left corner of the wall block through (x, y): walk left along the row, then up along that column, each
## bounded by `max_block` (0: unbounded) so the scan stays cheap. Every cell of one rectangular block agrees.
static func block_origin(terrain: PackedByteArray, size: int, x: int, y: int, wall_id: int, max_block: int) -> Vector2i:
	var limit: int = max_block if max_block > 0 else size
	var ox: int = x
	var steps: int = 0
	while ox > 0 and steps < limit and terrain[y * size + ox - 1] == wall_id:
		ox -= 1
		steps += 1
	var oy: int = y
	steps = 0
	while oy > 0 and steps < limit and terrain[(oy - 1) * size + ox] == wall_id:
		oy -= 1
		steps += 1
	return Vector2i(ox, oy)


## Roof style index 0..styles-1 for a house: by tile_hash of the block's top-left corner when `by_hash`, else 0
## (tiles.json roofs.style_mode "hash" / absent). One style per house, stable across rebuilds.
static func roof_style(terrain: PackedByteArray, size: int, x: int, y: int, wall_id: int, max_block: int,
		styles: int, by_hash: bool) -> int:
	if styles <= 1 or not by_hash:
		return 0
	var o: Vector2i = block_origin(terrain, size, x, y, wall_id, max_block)
	return posmod(cell_hash(o.x, o.y) >> 3, styles)


static func block_fits(terrain: PackedByteArray, size: int, x: int, y: int, wall_id: int, max_block: int) -> bool:
	if max_block <= 0:
		return true
	var w: int = 1
	var i: int = x - 1
	while i >= 0 and w <= max_block and terrain[y * size + i] == wall_id:
		w += 1
		i -= 1
	i = x + 1
	while i < size and w <= max_block and terrain[y * size + i] == wall_id:
		w += 1
		i += 1
	if w > max_block:
		return false
	var h: int = 1
	var j: int = y - 1
	while j >= 0 and h <= max_block and terrain[j * size + x] == wall_id:
		h += 1
		j -= 1
	j = y + 1
	while j < size and h <= max_block and terrain[j * size + x] == wall_id:
		h += 1
		j += 1
	return h <= max_block


## 4-bit mask of the neighbours of (x, y) that are `wall_id`: N 1, E 2, S 4, W 8 (off-map counts as not wall).
static func wall_mask(terrain: PackedByteArray, size: int, x: int, y: int, wall_id: int) -> int:
	var m: int = 0
	if y > 0 and terrain[(y - 1) * size + x] == wall_id:
		m |= 1
	if x < size - 1 and terrain[y * size + x + 1] == wall_id:
		m |= 2
	if y < size - 1 and terrain[(y + 1) * size + x] == wall_id:
		m |= 4
	if x > 0 and terrain[y * size + x - 1] == wall_id:
		m |= 8
	return m


func _make_props(index: Dictionary) -> void:
	if props_layer == null:
		props_layer = TileMapLayer.new()
		props_layer.name = "Props"
		props_layer.y_sort_enabled = true
		props_layer.z_as_relative = false
		props_layer.z_index = 0
		add_child(props_layer)
	y_sort_enabled = true  # nested y-sort: the Props cells join the parent's sort; the terrain sorts among itself
	z_index = TERRAIN_Z
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
		var td: TileData = src.get_tile_data(at, 0)
		td.texture_origin = _vec2i(cell.get("origin", [0, 0]))
		td.y_sort_origin = prop_sort_origin(int(cell.get("base_px", cell_px / 2)), cell_px)
	ts.add_source(src, SOURCE_ID)
	roof_terrain = -1
	var roofs: Dictionary = index.get("roofs", {})
	var rtex: Texture2D = _texture(str(roofs.get("atlas", "")))
	if rtex != null:
		var rs := TileSetAtlasSource.new()
		rs.texture = rtex
		rs.texture_region_size = Vector2i(cell_px, cell_px)
		var rsz: Vector2i = _vec2i(roofs.get("size", [1, 2]))
		var rorg: Vector2i = _vec2i(roofs.get("origin", [0, 32]))
		var rbase: int = int(roofs.get("base_px", cell_px / 2))
		var made: int = 0
		var feats: Dictionary = roofs.get("features", {})
		var feat_rows: int = maxi(1, (feats.get("rows", ["plain"]) as Array).size())
		var n_styles: int = maxi(1, (roofs.get("styles", ["tile_red"]) as Array).size())
		for fr: int in range(feat_rows * n_styles):  # style s, feature f -> tile row s * feat_rows + f
			for c: int in range(int(roofs.get("columns", 16))):
				var at := Vector2i(c, fr * rsz.y)  # a 1x2 tile spans two atlas cells: tile row r sits at cell row r*2
				if not rs.has_room_for_tile(at, rsz, 1, Vector2i.ZERO, 1):
					continue
				rs.create_tile(at, rsz)
				var rtd: TileData = rs.get_tile_data(at, 0)
				rtd.texture_origin = rorg
				rtd.y_sort_origin = prop_sort_origin(rbase, cell_px)
				if fr == 0:
					made += 1
		if made == 16:
			ts.add_source(rs, ROOF_SOURCE_ID)
			roof_terrain = int(roofs.get("terrain", 6))
			roof_feature_every = int(feats.get("every", 0)) if feat_rows >= 3 else 0
			roof_row_cells = rsz.y
			roof_max_block = int(roofs.get("max_block", 0))
			roof_styles = n_styles
			roof_feature_rows = feat_rows
			roof_style_by_hash = str(roofs.get("style_mode", "fixed")) == "hash"
	_rules.clear()
	_rules_by_terrain.clear()
	_rule_packs.clear()
	_rule_flags.resize(TERRAIN_IDS)
	_rule_flags.fill(0)
	for t: int in range(TERRAIN_IDS):
		_rules_by_terrain.append([])
		_rule_packs.append(PackedInt32Array())
	for r: Variant in index.get("rules", []):
		var rd: Dictionary = r
		var pname: String = str(rd.get("prop", ""))
		var on: int = int(rd.get("on", -1))
		if not cells.has(pname) or on < 0 or on >= TERRAIN_IDS:
			continue
		var row: Dictionary = {"on": on, "near": int(rd.get("near", -1)), "every": int(rd.get("every", 1)),
				"by_x": bool(rd.get("by_x", false)), "atlas": _vec2i((cells[pname] as Dictionary).get("atlas", [0, 0]))}
		_rules.append(row)
		(_rules_by_terrain[on] as Array).append(row)
		var at2: Vector2i = row["atlas"]
		_rule_packs[on].append_array(PackedInt32Array([int(row["near"]), maxi(int(row["every"]), 1),
				1 if bool(row["by_x"]) else 0, at2.x, at2.y]))
		_rule_flags[on] = 1
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
	var near_bits := PackedInt32Array()
	var near_offset := PackedInt32Array()
	for t: int in range(TERRAIN_IDS):
		modes.append(ViewMath.TILE_SINGLE)
		connects.append(0)
		weights.append(PackedInt32Array())
		near_bits.append(0)
		near_offset.append(0)
	for e: Variant in index["terrains"]:
		var d: Dictionary = e
		var id: int = int(d.get("id", -1))
		if id < 0 or id >= TERRAIN_IDS:
			continue
		var near: Dictionary = d.get("near", {})
		for nid: Variant in near.get("ids", []):
			near_bits[id] |= 1 << int(nid)
		near_offset[id] = int(near.get("offset", 0))
		match str(d.get("mode", "single")):
			"blob4", "blob47":
				modes[id] = ViewMath.TILE_BLOB47 if str(d.get("mode")) == "blob47" else ViewMath.TILE_BLOB4
				var bits: int = 0
				for c: Variant in d.get("connects", []):
					bits |= 1 << int(c)
				connects[id] = bits
			"variants":
				modes[id] = ViewMath.TILE_VARIANTS
				var wl: Array = d.get("weights", [])
				weights[id] = PackedInt32Array(wl)
	var rows: PackedByteArray = ViewMath.tile_rows(terrain, size, modes, connects, weights, near_bits,
			near_offset)
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
				elif t == roof_terrain and block_fits(terrain, size, x, y, t, roof_max_block):
					var mask: int = wall_mask(terrain, size, x, y, t)
					props_layer.set_cell(cell, ROOF_SOURCE_ID,
							Vector2i(mask, (roof_style(terrain, size, x, y, t, roof_max_block, roof_styles, roof_style_by_hash)
									* roof_feature_rows + roof_feature_row(mask, x, y, roof_feature_every)) * roof_row_cells))
					prop_tiles += 1
				elif _rule_flags[t] == 1:  # only terrains with a rule pay the neighbour check
					var rp: Vector2i = rule_prop(terrain, size, x, y, t)
					if rp.x >= 0:
						props_layer.set_cell(cell, SOURCE_ID, rp)
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
