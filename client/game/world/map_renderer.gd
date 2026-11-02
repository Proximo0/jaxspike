class_name MapRenderer
extends TileMapLayer
## The whole map as a TileMapLayer with a runtime-made placeholder tile set: one flat colour per terrain id 0..7.
## Filled once from MapData; a changed Tuning.tile_px swaps the atlas texture and tile size in place.

const TERRAIN_IDS: int = 8
const SOURCE_ID: int = 0

var built_tile_px: int = 0
var filled_tiles: int = 0
var _source: TileSetAtlasSource = null


func _process(_delta: float) -> void:
	if built_tile_px == 0:
		return
	var px: int = _wanted_px()
	if px != built_tile_px:
		_apply_tile_px(px)


## Builds the tile set and fills every cell from `map`. Returns the number of cells set.
func build(map: MapData) -> int:
	clear()
	filled_tiles = 0
	if not map.is_loaded():
		return 0
	_make_tile_set(_wanted_px())
	var size: int = map.size
	var terrain: PackedByteArray = map.terrain
	for y: int in range(size):
		var row: int = y * size
		for x: int in range(size):
			var t: int = terrain[row + x]
			if t >= TERRAIN_IDS:
				t = 0
			set_cell(Vector2i(x, y), SOURCE_ID, Vector2i(t, 0))
	filled_tiles = size * size
	return filled_tiles


func tile_count() -> int:
	return filled_tiles


func _wanted_px() -> int:
	return clampi(roundi(ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)), 4, 256)


func _make_tile_set(px: int) -> void:
	var ts := TileSet.new()
	_source = TileSetAtlasSource.new()
	_source.texture = _make_texture(px)
	_source.texture_region_size = Vector2i(px, px)
	for i: int in range(TERRAIN_IDS):
		_source.create_tile(Vector2i(i, 0))
	ts.tile_size = Vector2i(px, px)
	ts.add_source(_source, SOURCE_ID)
	tile_set = ts
	built_tile_px = px


func _apply_tile_px(px: int) -> void:
	_source.texture = _make_texture(px)
	_source.texture_region_size = Vector2i(px, px)
	tile_set.tile_size = Vector2i(px, px)
	built_tile_px = px


func _make_texture(px: int) -> ImageTexture:
	var img: Image = Image.create(px * TERRAIN_IDS, px, false, Image.FORMAT_RGBA8)
	for i: int in range(TERRAIN_IDS):
		img.fill_rect(Rect2i(i * px, 0, px, px), ViewMath.terrain_color(i))
	return ImageTexture.create_from_image(img)
