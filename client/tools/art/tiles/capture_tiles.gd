extends SceneTree
## Windowed review renders of the tileset through the real MapRenderer, with a few chibi sprites for scale.
##   Godot_v4.7.2-stable_win64.exe --path client -s res://tools/art/tiles/capture_tiles.gd -- \
##       --terrain=<512x512 uint8 row-major terrain.bin> --out=<dir>
## Writes <out>/<shot>_zoom<z>.png for each shot at its zooms (towns, dungeons, a lake shore, a road bend, a wall
## corner), with a furnace and a fire node (406/407, art from ViewMath.node_prop_texture) beside the sprites.
## Quiet: prints one CAPTURE_TILES line (or the error).

const MAP: int = 512
const SHOTS: Array[Dictionary] = [
	{"name": "solmera_square", "tile": Vector2(256.0, 257.0), "zooms": [1.0, 2.0]},
	{"name": "lunavell_square", "tile": Vector2(102.0, 114.0), "zooms": [1.0, 2.0]},
	{"name": "pinehollow", "tile": Vector2(410.0, 103.0), "zooms": [1.0]},
	{"name": "tidewick", "tile": Vector2(399.0, 411.0), "zooms": [1.0]},
	{"name": "kharsand", "tile": Vector2(113.0, 400.0), "zooms": [1.0]},
	{"name": "cave", "tile": Vector2(256.0, 46.0), "zooms": [1.0, 2.0]},
	{"name": "cave_south", "tile": Vector2(256.0, 466.0), "zooms": [1.0]},
	{"name": "lake_shore", "tile": Vector2(432.0, 271.0), "zooms": [2.0]},
	{"name": "road_bend", "tile": Vector2(410.0, 255.0), "zooms": [2.0]},
	{"name": "wall_corner", "tile": Vector2(227.0, 227.0), "zooms": [2.0]},
]
const SPRITES: Array[Dictionary] = [
	{"act": "res://assets/anim/swordsman/act.json", "kind": 0, "label": "Swordsman", "dx": -2, "dy": 2},
	{"act": "res://assets/anim/mage/act.json", "kind": 0, "label": "Mage", "dx": 1, "dy": 3},
	{"act": "res://assets/anim/blobbin/act.json", "kind": 1, "label": "Blobbin", "dx": 3, "dy": 1},
	{"node": 406, "label": "Furnace", "dx": -1, "dy": -2},
	{"node": 407, "label": "Fire", "dx": 1, "dy": -2},
]

var _out: String = ""
var _renderer: MapRenderer = null
var _camera: Camera2D = null
var _sprites: Node2D = null


func _initialize() -> void:
	var terrain_path: String = ""
	for a: String in OS.get_cmdline_user_args():
		if a.begins_with("--terrain="):
			terrain_path = a.substr(10)
		elif a.begins_with("--out="):
			_out = a.substr(6)
	var bytes: PackedByteArray = FileAccess.get_file_as_bytes(terrain_path)
	if bytes.size() != MAP * MAP or _out == "":
		print("CAPTURE_TILES error: need --terrain=<%d bytes> and --out=<dir> (got %d bytes)" % [MAP * MAP, bytes.size()])
		quit(1)
		return
	DirAccess.make_dir_recursive_absolute(_out)
	var map := MapData.new()
	map.size = MAP
	map.terrain = bytes
	map.pz = PackedByteArray()
	map.pz.resize(MAP * MAP)
	_renderer = MapRenderer.new()
	root.add_child(_renderer)
	var t0: int = Time.get_ticks_msec()
	_renderer.build(map)
	var build_ms: int = Time.get_ticks_msec() - t0
	_sprites = Node2D.new()
	root.add_child(_sprites)
	_camera = Camera2D.new()
	root.add_child(_camera)  # the only Camera2D: current once it enters the tree
	_run.call_deferred(build_ms)


func _run(build_ms: int) -> void:
	var tile_px: float = ViewMath.tuned("tile_px", ViewMath.DEFAULT_TILE_PX)
	var files: int = 0
	for shot: Dictionary in SHOTS:
		var centre: Vector2 = shot["tile"]
		_place_sprites(centre, tile_px)
		for zv: Variant in shot["zooms"]:
			var z: float = float(zv)
			_camera.zoom = Vector2(z, z)
			_camera.position = ViewMath.tile_center(centre, tile_px)
			for i: int in range(4):
				await process_frame
			await RenderingServer.frame_post_draw
			var img: Image = root.get_texture().get_image()
			var path: String = _out.path_join("%s_zoom%d.png" % [shot["name"], int(z)])
			if img.save_png(path) == OK:
				files += 1
	print("CAPTURE_TILES done files=%d build_ms=%d tileset=%s cells=%d props=%d" % [files, build_ms,
			str(_renderer.uses_tileset), _renderer.tile_count(), _renderer.prop_tiles])
	quit(0)


func _place_sprites(centre: Vector2, tile_px: float) -> void:
	for c: Node in _sprites.get_children():
		c.free()
	for s: Dictionary in SPRITES:
		var sp := EntitySprite.new()
		_sprites.add_child(sp)
		sp.position = ViewMath.tile_center(centre + Vector2(float(s["dx"]), float(s["dy"])), tile_px)
		if s.has("node"):
			var kind: int = int(s["node"])
			sp.configure(kind, false, str(s["label"]), 1.0, tile_px, 12, ViewMath.node_prop_texture(kind), false,
					ViewMath.CAT_NODE)
			continue
		var sheet: ActSheet = ActSheet.load_act(str(s["act"]))
		sp.configure(int(s["kind"]), false, str(s["label"]), 1.0, tile_px, 12)
		if sheet != null:
			var clip: Dictionary = sheet.resolve(0, 1, int(s["kind"]) == 0)
			if not clip.is_empty():
				sp.set_clip(sheet, 0, str(clip["name"]), clip["anim"], 0, int(clip["frames"]), 1, bool(clip["flip"]),
						0.0)
