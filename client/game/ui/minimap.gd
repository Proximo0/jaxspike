class_name Minimap
extends PanelContainer
## RO minimap (top right): the map downscaled into one texture built once from MapData, the own position as a
## blinking white dot, portal (warp) and waybell markers from `welcome.world`, and optional entity dots (monsters
## red, players blue, drops yellow) behind the "Dots" toggle. "+/-" switches between the whole map and a local
## window around the player. Feed set_welcome(msg) and build(map); the view polls the session's store.

const VIEW_PX: int = 200  ## Drawn square size.
const MAX_TEX_PX: int = 512  ## The texture is at most this wide: maps above it are sampled every stride tiles.
const LOCAL_TILES: int = 96  ## Tiles across in the local view.
const WARP_COLOR: Color = Color(0.80, 0.45, 1.0)
const WAYBELL_COLOR: Color = Color(1.0, 0.85, 0.30)
const OWN_COLOR: Color = Color(1.0, 1.0, 1.0)
const DOT_COLORS: Array[Color] = [Color(0.35, 0.60, 1.0), Color(1.0, 0.30, 0.30), Color(1.0, 0.90, 0.30), Color(0.50, 0.95, 0.50)]
const MAX_DOTS: int = 256

## The drawn square: the texture and every marker, in local tile coordinates of the visible window.
class MapView extends Control:
	var mm: Minimap

	func _draw() -> void:
		if mm != null:
			mm.draw_into(self)

var session: NetSession = null
var welcome: Dictionary = {}
var map_size: int = 0
var stride: int = 1
var texture: ImageTexture = null
var markers: Array[Dictionary] = []  ## {pos: Vector2i, kind: "warp" | "waybell"}
var show_dots: bool = false
var local_view: bool = false
var blink_s: float = 0.0
var view: MapView
var dots_button: Button
var zoom_button: Button
var title: Label


func _ready() -> void:
	theme = load("res://game/ui/theme/ro_theme.tres") as Theme
	theme_type_variation = &"MinimapFrame"
	var box := VBoxContainer.new()
	add_child(box)
	var bar := HBoxContainer.new()
	box.add_child(bar)
	title = Label.new()
	title.text = "Map"
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	bar.add_child(title)
	zoom_button = _minimap_button(bar, "+", toggle_zoom)
	dots_button = _minimap_button(bar, "Dots", toggle_dots)
	dots_button.toggle_mode = true
	view = MapView.new()
	view.mm = self
	view.custom_minimum_size = Vector2(VIEW_PX, VIEW_PX)
	view.clip_contents = true
	view.mouse_filter = Control.MOUSE_FILTER_IGNORE
	box.add_child(view)


func _minimap_button(parent: Control, text: String, cb: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.focus_mode = Control.FOCUS_NONE
	b.pressed.connect(cb)
	parent.add_child(b)
	return b


func bind(s: NetSession) -> void:
	session = s


func set_welcome(msg: Dictionary) -> void:
	welcome = msg
	var world: Variant = msg.get("world")
	markers = world_markers(world if typeof(world) == TYPE_DICTIONARY else {})


func toggle_dots() -> void:
	show_dots = not show_dots
	if dots_button != null:
		dots_button.button_pressed = show_dots
	if view != null:
		view.queue_redraw()


func toggle_zoom() -> void:
	local_view = not local_view
	if zoom_button != null:
		zoom_button.text = "-" if local_view else "+"
	if view != null:
		view.queue_redraw()


## Builds the downscaled texture once per map. Returns the texture size in pixels.
func build(map: MapData) -> int:
	if map == null or not map.is_loaded():
		texture = null
		map_size = 0
		return 0
	map_size = map.size
	stride = stride_for(map.size)
	var px: int = (map.size + stride - 1) / stride
	var img: Image = Image.create_from_data(px, px, false, Image.FORMAT_RGB8, image_bytes(map, stride))
	texture = ImageTexture.create_from_image(img)
	if typeof(welcome.get("world")) != TYPE_DICTIONARY and map.size <= MAX_TEX_PX:
		markers = scan_markers(map)
	if view != null:
		view.queue_redraw()
	return px


func _process(delta: float) -> void:
	blink_s += delta
	if view != null and texture != null:
		view.queue_redraw()


## Tile -> pixel inside the drawn square for the current window (`origin` tile at pixel 0, `span` tiles across).
func window() -> Rect2:
	if map_size <= 0:
		return Rect2(0, 0, 1, 1)
	if not local_view:
		return Rect2(0, 0, map_size, map_size)
	var own: Vector2 = _own_tile()
	var span: float = float(mini(LOCAL_TILES, map_size))
	var o: Vector2 = (own - Vector2(span, span) * 0.5).clamp(Vector2.ZERO, Vector2(map_size - span, map_size - span))
	return Rect2(o, Vector2(span, span))


func draw_into(c: Control) -> void:
	c.draw_rect(Rect2(Vector2.ZERO, c.size), Color(0.05, 0.06, 0.08))
	if texture == null:
		return
	var w: Rect2 = window()
	var scale: float = float(VIEW_PX) / w.size.x
	var src := Rect2(w.position / float(stride), w.size / float(stride))
	c.draw_texture_rect_region(texture, Rect2(0, 0, VIEW_PX, VIEW_PX), src)
	for m: Dictionary in markers:
		var p: Vector2 = (Vector2(m["pos"] as Vector2i) - w.position + Vector2(0.5, 0.5)) * scale
		if not Rect2(Vector2.ZERO, Vector2(VIEW_PX, VIEW_PX)).has_point(p):
			continue
		var col: Color = WARP_COLOR if String(m["kind"]) == "warp" else WAYBELL_COLOR
		c.draw_circle(p, 4.0, Color(0, 0, 0, 0.8))
		c.draw_circle(p, 3.0, col)
	if show_dots and session != null:
		var n: int = 0
		for id: int in session.store.ids():
			if id == session.slot or n >= MAX_DOTS:
				continue
			var e: Dictionary = session.store.get_ent(id)
			var cat: int = ViewMath.entity_category(int(e.get("kind", 0)), e.has("item"))
			var p: Vector2 = (ViewMath.to_vec(e.get("pos")) - w.position + Vector2(0.5, 0.5)) * scale
			c.draw_rect(Rect2(p - Vector2(1.5, 1.5), Vector2(3, 3)), DOT_COLORS[clampi(cat, 0, DOT_COLORS.size() - 1)])
			n += 1
	var own: Vector2 = (_own_tile() - w.position + Vector2(0.5, 0.5)) * scale
	var on: bool = fmod(blink_s, 0.8) < 0.55
	c.draw_circle(own, 4.5, Color(0, 0, 0, 0.9))
	c.draw_circle(own, 3.0, OWN_COLOR if on else Color(1.0, 0.4, 0.4))


func _own_tile() -> Vector2:
	if session == null or session.store.you.is_empty():
		return Vector2(map_size, map_size) * 0.5
	return ViewMath.to_vec(session.store.you.get("pos"))


# ---- pure helpers ----

static func stride_for(size: int) -> int:
	return maxi(1, (size + MAX_TEX_PX - 1) / MAX_TEX_PX)


## RGB8 bytes of the map sampled every `stride` tiles (row-major, one pixel per sample), from the terrain colours;
## protection-zone tiles are lightened so towns read at a glance.
static func image_bytes(map: MapData, stride: int) -> PackedByteArray:
	var px: int = (map.size + stride - 1) / stride
	var lut := PackedByteArray()
	lut.resize(256 * 3)
	for t: int in range(256):
		var id: int = t - 256 if t > 127 else t
		var col: Color = ViewMath.terrain_color(id) if id >= 0 and id < ViewMath.TERRAIN_COLORS.size() else Color(0, 0, 0)
		lut[t * 3] = col.r8
		lut[t * 3 + 1] = col.g8
		lut[t * 3 + 2] = col.b8
	var out := PackedByteArray()
	out.resize(px * px * 3)
	var terrain: PackedByteArray = map.terrain
	var pz: PackedByteArray = map.pz
	var has_pz: bool = pz.size() == terrain.size()
	var o: int = 0
	for py: int in range(px):
		var row: int = py * stride * map.size
		for qx: int in range(px):
			var i: int = row + qx * stride
			var t: int = terrain[i]
			var r: int = lut[t * 3]
			var g: int = lut[t * 3 + 1]
			var b: int = lut[t * 3 + 2]
			if has_pz and pz[i] != 0:
				r = (r + 255) >> 1
				g = (g + 235) >> 1
				b = (b + 200) >> 1
			out[o] = r
			out[o + 1] = g
			out[o + 2] = b
			o += 3
	return out


## Warp sources and waybell tiles from welcome.world.
static func world_markers(world: Dictionary) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	var warps: Variant = world.get("warps")
	if typeof(warps) == TYPE_ARRAY:
		for w: Variant in warps as Array:
			if typeof(w) == TYPE_DICTIONARY and (w as Dictionary).has("from"):
				out.append({"pos": Vector2i(ViewMath.to_vec((w as Dictionary)["from"])), "kind": "warp"})
	var towns: Variant = world.get("towns")
	if typeof(towns) == TYPE_ARRAY:
		for t: Variant in towns as Array:
			if typeof(t) == TYPE_DICTIONARY and (t as Dictionary).has("waybell"):
				out.append({"pos": Vector2i(ViewMath.to_vec((t as Dictionary)["waybell"])), "kind": "waybell"})
	return out


## Fallback without welcome.world: every warp (8) and waybell (9) tile of a small map.
static func scan_markers(map: MapData) -> Array[Dictionary]:
	var out: Array[Dictionary] = []
	for i: int in range(map.terrain.size()):
		var t: int = map.terrain[i]
		if t == 8 or t == 9:
			out.append({"pos": Vector2i(i % map.size, i / map.size), "kind": "warp" if t == 8 else "waybell"})
	return out
